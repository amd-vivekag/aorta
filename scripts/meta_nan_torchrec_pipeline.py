"""
TorchRec TrainPipelineSparseDist faithful reproducer for AQL queue crash.

Tests whether TorchRec's record_stream-based memory management prevents the
AQL queue depth crash on AMD GPUs. Uses the EXACT same buffer lifecycle as
TorchRec's TrainPipelineSparseDist:

  1. Pipelineable batch with .to(device) and .record_stream() (like KJT)
  2. copy_batch_to_gpu: .to(device, non_blocking=True) on memcpy_stream
  3. _wait_for_batch: wait_stream + record_stream (TorchRec's pattern)
  4. deque-based batch lifecycle: append on enqueue, popleft on dequeue
  5. No pre-allocated fixed buffers — caching allocator recycles memory

The key question: does record_stream prevent the crash?
  - If YES → the issue_a reproducer simply had a missing record_stream
  - If NO  → the crash is below the caching allocator (kernarg recycling)

PIPELINE LAYOUT (matching TorchRec TrainPipelineSparseDist):

  batches = deque(maxlen=3)
  batches[0]: forward pass on default_stream     (iteration N)
  batches[1]: input_dist on datadist_stream       (iteration N+1)
  batches[2]: H2D copy on memcpy_stream           (iteration N+2)

  Each progress() step:
    1. fill_pipeline (if needed)
    2. _wait_for_batch(batches[0]) → wait_stream + record_stream
    3. start_sparse_data_dist(batches[1]) on datadist_stream
    4. enqueue_batch → .to(device) on memcpy_stream → append to deque
    5. forward(batches[0]) on default_stream
    6. wait_sparse_data_dist
    7. dequeue_batch → popleft (drops reference, allocator may reclaim)

Usage:
    # Test 1: Pipeline with record_stream (the key test)
    PYTHONPATH=scripts torchrun --nproc_per_node=2 \\
        scripts/meta_nan_torchrec_pipeline.py --pipelined --batch-size 4096 --no-compile

    # Test 2: Pipeline with sync-per-iter (should pass)
    PYTHONPATH=scripts torchrun --nproc_per_node=2 \\
        scripts/meta_nan_torchrec_pipeline.py --pipelined --sync-per-iter --batch-size 4096

    # Test 3: Non-pipelined baseline (should pass)
    PYTHONPATH=scripts torchrun --nproc_per_node=2 \\
        scripts/meta_nan_torchrec_pipeline.py --batch-size 4096

    # Test 4: Pipeline WITHOUT record_stream (should crash, like issue_a)
    PYTHONPATH=scripts torchrun --nproc_per_node=2 \\
        scripts/meta_nan_torchrec_pipeline.py --pipelined --no-record-stream --batch-size 4096

    # Single-GPU
    CUDA_VISIBLE_DEVICES=0 python scripts/meta_nan_torchrec_pipeline.py \\
        --pipelined --batch-size 4096 --no-compile
"""

import argparse
import logging
import os
import sys
import time
from collections import deque
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# =========================================================================
# Pipelineable batch — matches KJT's .to() and .record_stream() semantics
# =========================================================================


class PipelineableBatch:
    """Batch implementing TorchRec's Pipelineable / Multistreamable interface.

    .to(device, non_blocking) creates a NEW batch with freshly allocated
    device tensors (exactly like KeyedJaggedTensor.to()). The old batch's
    tensors are released, and the caching allocator may recycle their memory.

    .record_stream(stream) marks ALL tensor fields as used by the given
    stream, preventing the caching allocator from recycling them until
    that stream has consumed them.
    """

    __slots__ = ["dense", "indices_list", "offsets_list"]

    def __init__(
        self,
        dense: torch.Tensor,
        indices_list: List[torch.Tensor],
        offsets_list: List[torch.Tensor],
    ):
        self.dense = dense
        self.indices_list = indices_list
        self.offsets_list = offsets_list

    def to(self, device: torch.device, non_blocking: bool = False) -> "PipelineableBatch":
        return PipelineableBatch(
            dense=self.dense.to(device, non_blocking=non_blocking),
            indices_list=[t.to(device, non_blocking=non_blocking) for t in self.indices_list],
            offsets_list=[t.to(device, non_blocking=non_blocking) for t in self.offsets_list],
        )

    def record_stream(self, stream: torch.cuda.Stream) -> None:
        self.dense.record_stream(stream)
        for t in self.indices_list:
            t.record_stream(stream)
        for t in self.offsets_list:
            t.record_stream(stream)


# =========================================================================
# TorchRec pipeline utilities (copied from torchrec source, simplified)
# =========================================================================


def _to_device(
    batch: PipelineableBatch,
    device: torch.device,
    non_blocking: bool,
) -> PipelineableBatch:
    """Equivalent to torchrec.distributed.train_pipeline.utils._to_device."""
    return batch.to(device=device, non_blocking=non_blocking)


def _wait_for_batch(
    batch: PipelineableBatch,
    stream: Optional[torch.cuda.Stream],
    use_record_stream: bool = True,
) -> None:
    """Equivalent to torchrec.distributed.train_pipeline.utils._wait_for_batch.

    From TorchRec source:
      PyTorch uses the "caching allocator" for memory allocation for tensors.
      When a tensor is freed, its memory is likely to be reused by newly
      constructed tensors. By default, this allocator traces whether a tensor
      is still in use by only the CUDA stream where it was created. When a
      tensor is used by additional CUDA streams, we need to call
      record_stream to tell the allocator about these streams.
    """
    if stream is None:
        return

    curr_stream = torch.cuda.current_stream()
    curr_stream.wait_stream(stream)

    if use_record_stream:
        batch.record_stream(curr_stream)


# =========================================================================
# Model (reused from meta_nan_torchrec_sim.py)
# =========================================================================


class ShardedEmbeddingBag(nn.Module):
    """Simulates TorchRec's sharded EmbeddingBagCollection."""

    def __init__(self, num_tables: int, hash_size: int, dim: int,
                 rank: int, world_size: int):
        super().__init__()
        self.num_tables = num_tables
        self.hash_size = hash_size
        self.dim = dim

        tables_per_rank = num_tables // world_size
        self.local_tables = tables_per_rank
        self.tables = nn.ModuleList([
            nn.EmbeddingBag(hash_size, dim, mode="sum", sparse=False,
                            include_last_offset=True)
            for _ in range(tables_per_rank)
        ])
        for t in self.tables:
            nn.init.normal_(t.weight, std=0.01)

    def forward(self, indices_list: List[torch.Tensor],
                offsets_list: List[torch.Tensor]) -> torch.Tensor:
        outputs = []
        for i, table in enumerate(self.tables):
            out = table(indices_list[i], offsets_list[i])
            outputs.append(out)
        return torch.cat(outputs, dim=-1)


class InteractionArch(nn.Module):
    def __init__(self, num_features: int, dim: int):
        super().__init__()
        self.proj = nn.Linear(num_features * dim, dim * 4, bias=False)
        self.act = nn.GELU()
        self.out = nn.Linear(dim * 4, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.act(self.proj(x)))


class OverArch(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, num_layers: int = 3):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.TransformerEncoderLayer(
                d_model=dim, nhead=num_heads, dim_feedforward=dim * 4,
                batch_first=True, dropout=0.0, norm_first=True,
            ))
        self.head = nn.Linear(dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.head(x.mean(dim=1)).squeeze(-1)


class DLRMv3Model(nn.Module):
    def __init__(self, num_tables: int, hash_size: int, dim: int,
                 rank: int, world_size: int, num_dense: int = 13):
        super().__init__()
        self.embedding = ShardedEmbeddingBag(
            num_tables, hash_size, dim, rank, world_size,
        )
        self.dense_proj = nn.Linear(num_dense, dim, bias=False)
        tables_per_rank = num_tables // world_size
        total_sparse_dim = tables_per_rank * dim
        self.interaction = InteractionArch(num_features=2, dim=total_sparse_dim)
        self.over_arch = OverArch(dim=total_sparse_dim)

    def forward(self, batch: PipelineableBatch) -> torch.Tensor:
        sparse_out = self.embedding(batch.indices_list, batch.offsets_list)
        B = batch.dense.shape[0]
        dense_out = self.dense_proj(batch.dense)
        sparse_dim = sparse_out.shape[-1]
        dense_padded = F.pad(dense_out, (0, sparse_dim - dense_out.shape[-1]))
        combined = torch.stack([sparse_out, dense_padded], dim=1)
        interaction_out = self.interaction(combined.view(B, -1))
        return self.over_arch(interaction_out.unsqueeze(1))


# =========================================================================
# Data generation
# =========================================================================


def generate_host_pool(
    num_batches: int,
    batch_size: int,
    num_tables: int,
    pooling_factor: int,
    hash_size: int,
    num_dense: int,
) -> List[PipelineableBatch]:
    """Pre-generate pinned host batches for fast H2D."""
    rng_state = torch.random.get_rng_state()
    torch.manual_seed(42)
    pool = []
    for _ in range(num_batches):
        indices_list = []
        offsets_list = []
        for _ in range(num_tables):
            lengths = torch.randint(1, pooling_factor * 2 + 1, (batch_size,))
            total_indices = lengths.sum().item()
            indices = torch.randint(0, hash_size, (total_indices,), dtype=torch.long)
            offsets = torch.zeros(batch_size + 1, dtype=torch.long)
            offsets[1:] = torch.cumsum(lengths, 0)
            indices_list.append(indices.pin_memory())
            offsets_list.append(offsets.pin_memory())

        dense = torch.randn(batch_size, num_dense, dtype=torch.bfloat16).pin_memory()
        pool.append(PipelineableBatch(dense, indices_list, offsets_list))

    torch.random.set_rng_state(rng_state)
    return pool


# =========================================================================
# TrainPipelineSparseDist — faithful reimplementation
# =========================================================================


class SparsePipeline:
    """Faithful reimplementation of TorchRec's TrainPipelineSparseDist.

    Uses the exact same buffer lifecycle:
    - copy_batch_to_gpu: .to(device, non_blocking=True) on memcpy_stream
      → each batch gets FRESH device tensors from the caching allocator
    - _wait_for_batch: wait_stream + record_stream
      → tells the allocator that default_stream also uses these tensors
    - dequeue_batch: popleft from deque
      → drops the Python reference; allocator may reclaim the memory
      → record_stream's event prevents premature recycling (on CUDA)
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        host_pool: List[PipelineableBatch],
        use_record_stream: bool = True,
        world_size: int = 1,
        emb_dim: int = 0,
    ):
        self.model = model
        self.device = device
        self.host_pool = host_pool
        self.pool_idx = 0
        self.use_record_stream = use_record_stream

        self.memcpy_stream = torch.cuda.Stream()
        self.datadist_stream = torch.cuda.Stream()
        self.default_stream = torch.cuda.current_stream()

        self.batches: deque = deque()

        self.distributed = world_size > 1 and dist.is_initialized()
        if self.distributed:
            self.a2a_send = torch.empty(
                world_size, emb_dim, dtype=torch.bfloat16, device=device,
            )
            self.a2a_recv = torch.empty_like(self.a2a_send)

    def _next_host_batch(self) -> PipelineableBatch:
        batch = self.host_pool[self.pool_idx % len(self.host_pool)]
        self.pool_idx += 1
        return batch

    def copy_batch_to_gpu(self) -> Optional[PipelineableBatch]:
        """TorchRec: copy_batch_to_gpu — .to(device) on memcpy_stream."""
        host_batch = self._next_host_batch()
        with torch.cuda.stream(self.memcpy_stream):
            device_batch = _to_device(host_batch, self.device, non_blocking=True)
        return device_batch

    def enqueue_batch(self) -> bool:
        """TorchRec: enqueue_batch — copy to GPU, append to deque."""
        batch = self.copy_batch_to_gpu()
        if batch is None:
            return False
        self.batches.append(batch)
        return True

    def dequeue_batch(self) -> None:
        """TorchRec: dequeue_batch — popleft drops the reference."""
        self.batches.popleft()

    def wait_for_batch(self) -> None:
        """TorchRec: _wait_for_batch — wait_stream + record_stream."""
        if len(self.batches) == 0:
            return
        _wait_for_batch(
            self.batches[0],
            self.memcpy_stream,
            use_record_stream=self.use_record_stream,
        )

    def start_sparse_data_dist(self, iteration: int) -> Optional[dist.Work]:
        """Simulates input_dist (all_to_all) on datadist_stream."""
        if not self.distributed:
            return None
        with torch.cuda.stream(self.datadist_stream):
            self.a2a_send.fill_(float((iteration + 1) % 1000) / 1000.0)
            return dist.all_to_all_single(
                self.a2a_recv, self.a2a_send, async_op=True,
            )

    def wait_sparse_data_dist(self, work: Optional[dist.Work]) -> None:
        if work is not None:
            self.default_stream.wait_stream(self.datadist_stream)
            work.wait()

    def fill_pipeline(self) -> None:
        """TorchRec: fill_pipeline — prime with 2 batches."""
        while len(self.batches) < 2:
            if not self.enqueue_batch():
                return


# =========================================================================
# Eval loops
# =========================================================================


def run_pipelined(
    model: nn.Module,
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    device: torch.device,
    host_pool: List[PipelineableBatch],
) -> Tuple[int, Optional[int]]:
    tables_per_rank = args.num_tables // world_size
    emb_dim = args.batch_size * tables_per_rank * args.dim

    pipeline = SparsePipeline(
        model=model,
        device=device,
        host_pool=host_pool,
        use_record_stream=not args.no_record_stream,
        world_size=world_size,
        emb_dim=emb_dim,
    )

    if rank == 0:
        log.info(f"  record_stream: {'ON' if not args.no_record_stream else 'OFF'}")

    # Fill pipeline with initial batches (TorchRec: fill_pipeline)
    pipeline.fill_pipeline()

    nan_count = 0
    first_nan_iter = None
    start = time.time()
    last_log = start

    for it in range(args.iterations):
        # --- TorchRec progress() flow ---

        # 1. Wait for batches[0] to be ready on device
        pipeline.wait_for_batch()

        # 2. Start input_dist for batches[1] on datadist_stream
        dd_work = None
        if len(pipeline.batches) >= 2:
            dd_work = pipeline.start_sparse_data_dist(it)

        # 3. Enqueue next batch (H2D on memcpy_stream)
        pipeline.enqueue_batch()

        # 4. Forward pass on default_stream using batches[0]
        batch = pipeline.batches[0]
        with torch.no_grad():
            output = model(batch)

        # 5. Metrics on default_stream
        pred_f = output.float()

        # 6. Wait for datadist
        pipeline.wait_sparse_data_dist(dd_work)

        # 7. Dequeue consumed batch (popleft drops reference)
        pipeline.dequeue_batch()

        # Optional sync mitigation
        if args.sync_per_iter:
            torch.cuda.synchronize()

        # Periodic corruption check
        if (it + 1) % args.check_interval == 0:
            torch.cuda.synchronize()
            has_nan = torch.isnan(output).any().item()
            has_inf = torch.isinf(output).any().item()

            if has_nan or has_inf:
                nan_count += 1
                if first_nan_iter is None:
                    first_nan_iter = it
                if rank == 0:
                    nan_n = torch.isnan(output).sum().item()
                    inf_n = torch.isinf(output).sum().item()
                    log.error(
                        f"  iter={it + 1}: NaN={nan_n}, Inf={inf_n} "
                        f"[record_stream={'ON' if not args.no_record_stream else 'OFF'}]"
                    )

            now = time.time()
            if rank == 0 and (has_nan or has_inf or now - last_log >= 5):
                elapsed = now - start
                rate = (it + 1) / elapsed
                status = "NaN!" if (has_nan or has_inf) else "ok"
                log.info(
                    f"  [pipeline] iter={it + 1}/{args.iterations}  "
                    f"[{status}]  nans={nan_count}  rate={rate:.1f} it/s  "
                    f"batches_in_flight={len(pipeline.batches)}  "
                    f"record_stream={'ON' if not args.no_record_stream else 'OFF'}"
                )
                last_log = now

            if nan_count > 0 and args.stop_on_first:
                break

    torch.cuda.synchronize()
    return nan_count, first_nan_iter


def run_non_pipelined(
    model: nn.Module,
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    device: torch.device,
    host_pool: List[PipelineableBatch],
) -> Tuple[int, Optional[int]]:
    nan_count = 0
    first_nan_iter = None
    pool_idx = 0
    start = time.time()
    last_log = start

    for it in range(args.iterations):
        host_batch = host_pool[pool_idx % len(host_pool)]
        pool_idx += 1

        batch = _to_device(host_batch, device, non_blocking=False)
        with torch.no_grad():
            output = model(batch)
        torch.cuda.synchronize()

        if (it + 1) % args.check_interval == 0:
            has_nan = torch.isnan(output).any().item()
            has_inf = torch.isinf(output).any().item()
            if has_nan or has_inf:
                nan_count += 1
                if first_nan_iter is None:
                    first_nan_iter = it

            now = time.time()
            if rank == 0 and (has_nan or has_inf or now - last_log >= 5):
                rate = (it + 1) / (now - start)
                status = "NaN!" if (has_nan or has_inf) else "ok"
                log.info(
                    f"  [non-pipelined] iter={it + 1}/{args.iterations}  "
                    f"[{status}]  nans={nan_count}  rate={rate:.1f} it/s"
                )
                last_log = now

            if nan_count > 0 and args.stop_on_first:
                break

    return nan_count, first_nan_iter


# =========================================================================
# CLI and main
# =========================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TorchRec TrainPipelineSparseDist AQL crash reproducer",
    )
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--num-tables", type=int, default=16)
    parser.add_argument("--hash-size", type=int, default=1_000_000)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--pooling-factor", type=int, default=50)
    parser.add_argument("--num-dense", type=int, default=13)
    parser.add_argument("--pool-size", type=int, default=64)

    parser.add_argument("--pipelined", action="store_true",
                        help="Use 3-stage pipeline (default: non-pipelined)")
    parser.add_argument("--no-record-stream", action="store_true",
                        help="Disable record_stream (compare with/without)")
    parser.add_argument("--sync-per-iter", action="store_true",
                        help="torch.cuda.synchronize() every iteration")
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile")

    parser.add_argument("--check-interval", type=int, default=10)
    parser.add_argument("--stop-on-first", action="store_true", default=True)
    parser.add_argument("--no-stop-on-first", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    if args.no_stop_on_first:
        args.stop_on_first = False

    distributed = False
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        distributed = True
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

    rank = dist.get_rank() if distributed else 0
    world_size = dist.get_world_size() if distributed else 1
    device = torch.cuda.current_device()

    if args.num_tables % world_size != 0:
        args.num_tables = (args.num_tables // world_size) * world_size
        if rank == 0:
            log.warning(f"Adjusted num_tables to {args.num_tables}")

    tables_per_rank = args.num_tables // world_size

    if rank == 0:
        log.info("=" * 70)
        log.info("TORCHREC PIPELINE REPRODUCER: record_stream vs AQL queue depth")
        log.info("=" * 70)
        log.info(f"  mode: {'PIPELINED' if args.pipelined else 'NON-PIPELINED'}")
        log.info(f"  record_stream: {'OFF' if args.no_record_stream else 'ON'}")
        log.info(f"  sync_per_iter: {args.sync_per_iter}")
        log.info(f"  compile: {'disabled' if args.no_compile else 'enabled'}")
        log.info(f"  world_size={world_size}, iterations={args.iterations}")
        log.info(f"  batch_size={args.batch_size}")
        log.info(f"  tables={args.num_tables} (per_rank={tables_per_rank})")
        log.info(f"  hash_size={args.hash_size}, dim={args.dim}")
        log.info(f"  pooling_factor={args.pooling_factor}")
        log.info(f"  pool_size={args.pool_size}")
        log.info(f"  ROC_AQL_QUEUE_SIZE={os.environ.get('ROC_AQL_QUEUE_SIZE', '(not set)')}")
        log.info(f"  GPU_MAX_HW_QUEUES={os.environ.get('GPU_MAX_HW_QUEUES', '(not set)')}")
        log.info("=" * 70)

    # --- Model ---
    model = DLRMv3Model(
        num_tables=args.num_tables,
        hash_size=args.hash_size,
        dim=args.dim,
        rank=rank,
        world_size=world_size,
        num_dense=args.num_dense,
    ).to(device=device, dtype=torch.bfloat16)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    if rank == 0:
        log.info(f"Model parameters: {param_count:.1f}M")

    # --- Data ---
    if rank == 0:
        log.info(f"Generating {args.pool_size} host batches (pinned memory)...")
    host_pool = generate_host_pool(
        args.pool_size, args.batch_size, tables_per_rank,
        args.pooling_factor, args.hash_size, args.num_dense,
    )
    if rank == 0:
        log.info("Host pool ready.")

    # --- Warmup ---
    if rank == 0:
        log.info("Warmup (3 iters with sync)...")
    for i in range(3):
        batch = _to_device(host_pool[i], device, non_blocking=False)
        with torch.no_grad():
            model(batch)
        torch.cuda.synchronize()

    if not args.no_compile:
        if rank == 0:
            log.info("Compiling model with torch.compile...")
        model = torch.compile(model)
        for i in range(3):
            batch = _to_device(host_pool[i], device, non_blocking=False)
            with torch.no_grad():
                model(batch)
            torch.cuda.synchronize()
        if rank == 0:
            log.info("Compile warmup done.")

    # --- Run ---
    if rank == 0:
        log.info("")
        if args.pipelined:
            log.info("Starting PIPELINED eval loop "
                     f"(record_stream={'ON' if not args.no_record_stream else 'OFF'})...")
        else:
            log.info("Starting NON-PIPELINED eval loop...")
        log.info("")

    t_start = time.time()
    if args.pipelined:
        nan_count, first_nan = run_pipelined(
            model, args, rank, world_size, device, host_pool,
        )
    else:
        nan_count, first_nan = run_non_pipelined(
            model, args, rank, world_size, device, host_pool,
        )
    elapsed = time.time() - t_start

    # --- Results ---
    if rank == 0:
        log.info("")
        log.info("=" * 70)
        log.info("RESULTS")
        log.info("=" * 70)
        log.info(f"  Mode: {'PIPELINED' if args.pipelined else 'NON-PIPELINED'}")
        log.info(f"  record_stream: {'OFF' if args.no_record_stream else 'ON'}")
        log.info(f"  sync_per_iter: {args.sync_per_iter}")
        log.info(f"  Compile: {'disabled' if args.no_compile else 'enabled'}")
        log.info(f"  Elapsed: {elapsed:.1f}s ({args.iterations / max(elapsed, 0.1):.1f} it/s)")
        log.info(f"  NaN detections: {nan_count}")
        if first_nan is not None:
            log.info(f"  First NaN at iteration: {first_nan + 1}")

        if nan_count > 0:
            log.info("")
            if args.no_record_stream:
                log.info("VERDICT: CRASH without record_stream (expected, same as issue_a)")
            else:
                log.info("VERDICT: CRASH WITH record_stream!")
                log.info("  → record_stream does NOT prevent the crash on AMD")
                log.info("  → the issue is BELOW the caching allocator (kernarg recycling)")
        else:
            log.info("")
            if args.pipelined and not args.no_record_stream and not args.sync_per_iter:
                log.info("VERDICT: PASS with record_stream + pipelined + no sync")
                log.info("  → record_stream IS sufficient to prevent the crash")
                log.info("  → issue_a's crash was due to missing record_stream")
            else:
                log.info("VERDICT: PASS (expected for this configuration)")
        log.info("=" * 70)

    if distributed:
        dist.barrier()
        dist.destroy_process_group()

    sys.exit(1 if nan_count > 0 else 0)


if __name__ == "__main__":
    main()
