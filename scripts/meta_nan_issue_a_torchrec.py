"""
Meta NaN Issue A reproducer — TorchRec-style pipeline with record_stream.

Replicates the EXACT buffer management pattern of TorchRec's
TrainPipelineSparseDist without requiring fbgemm_gpu (which has ABI
incompatibility with our dev PyTorch).  The key mechanism we replicate:

  1.  copy_batch_to_gpu: `.to(device, non_blocking=True)` on memcpy_stream
      → caching allocator creates NEW device tensors each iteration.
  2.  _wait_for_batch:  `current_stream.wait_stream(memcpy_stream)` then
      `tensor.record_stream(current_stream)` to tell the allocator that
      the tensor is used on the default stream.
  3.  deque of 2 batches in the pipeline; oldest is popleft()-ed after
      forward, releasing its refcount.  The allocator may recycle the
      memory *immediately* if record_stream hasn't propagated.

WHY THIS SHOULD PRODUCE NaN (not crash) FOR ISSUE A:

  - Every iteration the caching allocator hands back memory for the SAME
    tensor types (int64 indices, bf16 embeddings) because the batch
    shape is identical across iterations.
  - When the GPU is far behind (AMD's 16K AQL queue), the allocator
    recycles memory from a *previous-but-not-yet-consumed* iteration.
  - The GPU reads stale-but-in-range indices from an earlier iteration.
    The embedding gather doesn't fault (indices are valid); it simply
    reads wrong rows, producing wrong-but-finite outputs.
  - Attention + MLP amplifies these wrong values into NaN via softmax
    overflow on wrong-magnitude dot products.

  Contrast with our `meta_nan_issue_a.py --alloc-mode alloc` which
  CRASHES because .to() for mixed dtypes can recycle memory from
  entirely different tensor types (e.g., bf16 recycled as int64),
  producing out-of-bounds indices.

3-STAGE PIPELINE (matching TrainPipelineSparseDist):

  At any point during steady state, up to 3 batches are in flight:

    batches[0]  →  default_stream:   forward + metrics  (iter N)
    batches[1]  →  data_dist_stream: all_to_all           (iter N+1)
    batches[2]  →  memcpy_stream:    H2D copy              (iter N+2)

Usage:
    # Default (should produce NaN on AMD with large AQL queue)
    torchrun --nproc_per_node=2 scripts/meta_nan_issue_a_torchrec.py

    # With AQL mitigation (should pass)
    ROC_AQL_QUEUE_SIZE=1024 torchrun --nproc_per_node=2 \\
        scripts/meta_nan_issue_a_torchrec.py --aql-queue-size 1024

    # Single-GPU
    python scripts/meta_nan_issue_a_torchrec.py --single-gpu

    # Disable torch.compile
    torchrun --nproc_per_node=2 scripts/meta_nan_issue_a_torchrec.py --no-compile
"""

import argparse
import logging
import math
import os
import sys
import time
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

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
# DLRMv3-style HSTU model (same as meta_nan_issue_a.py)
# =========================================================================


class HSTUAttentionLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, qk_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = qk_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, qk_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, qk_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4, bias=False),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim, bias=False),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        H = self.num_heads
        HD = self.head_dim

        residual = x
        x = self.norm1(x)

        q = self.q_proj(x).view(B, S, H, HD).transpose(1, 2)
        k = self.k_proj(x).view(B, S, H, HD).transpose(1, 2)
        v = self.v_proj(x).view(B, S, H, D // H).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, S, D)
        x = residual + self.out_proj(out)

        residual = x
        x = residual + self.ffn(self.norm2(x))
        return x


class HSTUModel(nn.Module):
    def __init__(
        self,
        item_hash_size: int = 1_000_000,
        user_hash_size: int = 100_000,
        category_size: int = 128,
        embedding_dim: int = 512,
        num_attention_layers: int = 5,
        num_heads: int = 4,
        attn_qk_dim: int = 128,
        preprocessor_dim: int = 256,
        max_seq_len: int = 256,
        max_candidates: int = 64,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.max_candidates = max_candidates

        self.item_embedding = nn.Embedding(item_hash_size, embedding_dim)
        self.user_embedding = nn.Embedding(user_hash_size, embedding_dim)
        self.category_embedding = nn.Embedding(category_size, embedding_dim)

        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.category_embedding.weight, std=0.01)

        self.preprocessor = nn.Sequential(
            nn.Linear(embedding_dim * 3, preprocessor_dim, bias=False),
            nn.GELU(),
            nn.Linear(preprocessor_dim, embedding_dim, bias=False),
        )

        self.attention_layers = nn.ModuleList([
            HSTUAttentionLayer(embedding_dim, num_heads, attn_qk_dim)
            for _ in range(num_attention_layers)
        ])

        self.output_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim, bias=False),
            nn.GELU(),
            nn.Linear(embedding_dim, 1, bias=False),
        )

    def forward(
        self,
        item_ids: torch.Tensor,
        user_ids: torch.Tensor,
        category_ids: torch.Tensor,
        candidate_item_ids: torch.Tensor,
        seq_lengths: torch.Tensor,
    ) -> torch.Tensor:
        item_emb = self.item_embedding(item_ids)
        user_emb = self.user_embedding(user_ids)
        cat_emb = self.category_embedding(category_ids)

        combined = torch.cat([item_emb, user_emb, cat_emb], dim=-1)
        x = self.preprocessor(combined)

        for layer in self.attention_layers:
            x = layer(x)

        cand_emb = self.item_embedding(candidate_item_ids)
        pooled = x.mean(dim=1, keepdim=True).expand_as(cand_emb)
        interaction = pooled * cand_emb

        predictions = self.output_head(interaction).squeeze(-1)
        return predictions


# =========================================================================
# Synthetic data: reuse DLRMv3 distributions
# =========================================================================

from dlrmv3_synthetic_data import (
    DLRMv3DataConfig,
    DLRMv3SyntheticBatchGenerator,
    ThreadedDataPipeline,
)


# =========================================================================
# TorchRec-style batch: a typed container with .to() and .record_stream()
#
# This replicates the Pipelineable / Multistreamable interface from TorchRec.
# The critical property: .to() creates NEW device tensors each call
# (through the caching allocator), and .record_stream() tells the
# allocator about cross-stream usage.
# =========================================================================


class RecBatch:
    """Typed batch container mimicking TorchRec's KJT/Pipelineable interface.

    All tensors share the same dtypes across iterations (int64 for indices,
    int64 for seq_lengths).  When the caching allocator recycles memory
    from a freed RecBatch, the new batch gets memory previously used by
    the SAME tensor types — so stale data is always valid-range indices
    from an earlier iteration, not garbage from a different dtype.

    This is what makes NaN (silent corruption) more likely than a crash.
    """
    __slots__ = ("item_ids", "user_ids", "category_ids",
                 "candidate_item_ids", "seq_lengths")

    def __init__(
        self,
        item_ids: torch.Tensor,
        user_ids: torch.Tensor,
        category_ids: torch.Tensor,
        candidate_item_ids: torch.Tensor,
        seq_lengths: torch.Tensor,
    ):
        self.item_ids = item_ids
        self.user_ids = user_ids
        self.category_ids = category_ids
        self.candidate_item_ids = candidate_item_ids
        self.seq_lengths = seq_lengths

    def to(self, device: torch.device, non_blocking: bool = False) -> "RecBatch":
        """Allocates fresh device tensors via the caching allocator."""
        return RecBatch(
            item_ids=self.item_ids.to(device, non_blocking=non_blocking),
            user_ids=self.user_ids.to(device, non_blocking=non_blocking),
            category_ids=self.category_ids.to(device, non_blocking=non_blocking),
            candidate_item_ids=self.candidate_item_ids.to(device, non_blocking=non_blocking),
            seq_lengths=self.seq_lengths.to(device, non_blocking=non_blocking),
        )

    def record_stream(self, stream: torch.cuda.Stream) -> None:
        """Tell the caching allocator these tensors are used on `stream`."""
        self.item_ids.record_stream(stream)
        self.user_ids.record_stream(stream)
        self.category_ids.record_stream(stream)
        self.candidate_item_ids.record_stream(stream)
        self.seq_lengths.record_stream(stream)

    def as_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "item_ids": self.item_ids,
            "user_ids": self.user_ids,
            "category_ids": self.category_ids,
            "candidate_item_ids": self.candidate_item_ids,
            "seq_lengths": self.seq_lengths,
        }


def dict_to_recbatch(d: Dict[str, torch.Tensor]) -> RecBatch:
    return RecBatch(
        item_ids=d["item_ids"],
        user_ids=d["user_ids"],
        category_ids=d["category_ids"],
        candidate_item_ids=d["candidate_item_ids"],
        seq_lengths=d["seq_lengths"],
    )


# =========================================================================
# TorchRec-style 3-stage pipeline
#
# Directly replicates TrainPipelineSparseDist.progress():
#   - batches deque (max capacity 3)
#   - copy_batch_to_gpu: .to(device) on memcpy_stream
#   - _wait_for_batch: wait_stream + record_stream
#   - start_sparse_data_dist: all_to_all on data_dist_stream
#   - wait_sparse_data_dist: wait for all_to_all
#   - forward on default_stream
#   - dequeue_batch: popleft
# =========================================================================


class TorchRecStylePipeline:
    """3-stage pipeline replicating TrainPipelineSparseDist's buffer lifecycle.

    The deque-based batches list with .to() allocation + record_stream is
    the EXACT pattern used by TorchRec.  When the AQL queue is deep (16K),
    record_stream's protection may be insufficient because the GPU hasn't
    actually consumed the tensor when the allocator checks liveness.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        data_pipeline: ThreadedDataPipeline,
        world_size: int,
        embedding_dim: int,
        batch_size: int,
        num_candidates: int,
        use_compile: bool = True,
        use_record_stream: bool = True,
        sync_interval: int = 0,
    ):
        self.model = model
        self.device = device
        self.data_pipeline = data_pipeline
        self.world_size = world_size
        self.use_record_stream = use_record_stream
        self.sync_interval = sync_interval

        self.memcpy_stream = torch.cuda.Stream()
        self.data_dist_stream = torch.cuda.Stream()
        self.default_stream = torch.cuda.current_stream()

        self.batches: Deque[RecBatch] = deque()
        self._dataloader_exhausted = False

        self._host_pool: Optional[List[RecBatch]] = None
        self._pool_idx = 0

        self.distributed = world_size > 1 and dist.is_initialized()
        datadist_size = batch_size * num_candidates * embedding_dim // 4
        if self.distributed:
            self.dd_send = torch.empty(world_size, datadist_size, dtype=torch.bfloat16, device=device)
            self.dd_recv = torch.empty_like(self.dd_send)
        else:
            self.dd_src = torch.empty(datadist_size, dtype=torch.bfloat16, device=device)
            self.dd_dst = torch.empty_like(self.dd_src)

        self.total_loss = torch.zeros(1, dtype=torch.float32, device=device)
        self.correct = torch.zeros(1, dtype=torch.int64, device=device)
        self.total = torch.zeros(1, dtype=torch.int64, device=device)
        self.pred_sum = torch.zeros(1, dtype=torch.bfloat16, device=device)

    def preload_host_pool(self, pool_size: int = 64) -> None:
        """Pre-generate batches for instant CPU dispatch (--fast-data)."""
        log.info(f"Pre-generating {pool_size} host batches...")
        self._host_pool = []
        for _ in range(pool_size):
            raw = self.data_pipeline.get_batch()
            if raw is None:
                break
            self._host_pool.append(dict_to_recbatch(raw))
        log.info(f"Host pool ready: {len(self._host_pool)} batches")

    def _next_batch_from_host(self) -> Optional[RecBatch]:
        if self._host_pool is not None:
            batch = self._host_pool[self._pool_idx % len(self._host_pool)]
            self._pool_idx += 1
            return batch
        raw = self.data_pipeline.get_batch()
        if raw is None:
            return None
        return dict_to_recbatch(raw)

    # --- TorchRec's copy_batch_to_gpu ---
    def copy_batch_to_gpu(self) -> Optional[RecBatch]:
        """H2D via .to(device) on memcpy_stream — caching allocator path."""
        host_batch = self._next_batch_from_host()
        if host_batch is None:
            self._dataloader_exhausted = True
            return None
        with torch.cuda.stream(self.memcpy_stream):
            device_batch = host_batch.to(self.device, non_blocking=True)
        return device_batch

    # --- TorchRec's _wait_for_batch ---
    def wait_for_batch(self) -> None:
        """Wait for batches[0] H2D to complete, then record_stream."""
        if not self.batches:
            return
        self.default_stream.wait_stream(self.memcpy_stream)
        if self.use_record_stream:
            self.batches[0].record_stream(self.default_stream)

    # --- TorchRec's start_sparse_data_dist ---
    def start_sparse_data_dist(self, iteration: int) -> Optional[dist.Work]:
        """all_to_all on data_dist_stream for batches[1]."""
        if len(self.batches) < 2:
            return None
        with torch.cuda.stream(self.data_dist_stream):
            self.data_dist_stream.wait_stream(self.memcpy_stream)
            if self.use_record_stream:
                self.batches[1].record_stream(self.data_dist_stream)
            if self.distributed:
                rank = dist.get_rank()
                self.dd_send.fill_(float(rank))
                return dist.all_to_all_single(self.dd_recv, self.dd_send, async_op=True)
            else:
                self.dd_src.fill_(float(iteration % 500))
                self.dd_dst.copy_(self.dd_src)
                return None

    # --- TorchRec's wait_sparse_data_dist ---
    def wait_sparse_data_dist(self, work: Optional[dist.Work]) -> None:
        if work is not None:
            self.default_stream.wait_stream(self.data_dist_stream)
            work.wait()

    def update_metrics(self, predictions: torch.Tensor) -> None:
        self.pred_sum += predictions.sum()
        labels = (predictions > 0).long()
        self.correct += labels.sum()
        self.total += predictions.numel()

    # --- TorchRec's dequeue_batch ---
    def dequeue_batch(self) -> None:
        """Pop the oldest batch. Its tensors' refcount drops; the caching
        allocator may recycle the underlying memory immediately."""
        if self.batches:
            self.batches.popleft()

    # --- TorchRec's enqueue_batch ---
    def enqueue_batch(self) -> bool:
        batch = self.copy_batch_to_gpu()
        if batch is None:
            return False
        self.batches.append(batch)
        return True

    # --- TorchRec's fill_pipeline ---
    def fill_pipeline(self) -> None:
        """Fill to capacity (2 batches) on first call — matches TorchRec."""
        if len(self.batches) >= 2:
            return
        self.enqueue_batch()
        if not self.batches:
            return
        # start data_dist for batch[0]
        self.wait_for_batch()
        self.start_sparse_data_dist(0)
        # batch i+1
        self.enqueue_batch()

    def progress(self, iteration: int) -> Optional[torch.Tensor]:
        """One pipeline step — mirrors TrainPipelineSparseDist.progress().

        Per-step order (from TorchRec source):
          1. fill_pipeline (only needed at start)
          2. _wait_for_batch (wait memcpy -> default, record_stream)
          3. start_sparse_data_dist for batches[1]
          4. enqueue_batch (copy next batch to GPU)  [= H2D for iter N+2]
          5. forward on batches[0]
          6. wait_sparse_data_dist for batches[1]
          7. metrics update
          8. dequeue_batch (popleft batches[0])
        """
        self.fill_pipeline()

        if not self.batches:
            return None

        # Step 2: wait for batches[0] H2D
        self.wait_for_batch()

        # Step 3: start data_dist for batches[1]
        dd_work = self.start_sparse_data_dist(iteration)

        # Step 4: enqueue next batch (H2D for iter N+2)
        if not self._dataloader_exhausted:
            self.enqueue_batch()

        # Step 5: forward on batches[0]
        batch = self.batches[0]
        with torch.no_grad():
            output = self.model(
                batch.item_ids, batch.user_ids, batch.category_ids,
                batch.candidate_item_ids, batch.seq_lengths,
            )

        # Step 6: wait for data_dist
        self.wait_sparse_data_dist(dd_work)

        # Step 7: metrics
        self.update_metrics(output)

        # Step 8: dequeue (free oldest batch → allocator may recycle)
        self.dequeue_batch()

        # Optional periodic sync
        if self.sync_interval > 0 and (iteration + 1) % self.sync_interval == 0:
            torch.cuda.synchronize()

        return output


# =========================================================================
# Corruption checker
# =========================================================================


class CorruptionChecker:
    def __init__(self):
        self.nan_count: int = 0
        self.inf_count: int = 0
        self.first_nan_iter: Optional[int] = None
        self.details: List[Dict] = []

    def check(self, tensor: torch.Tensor, iteration: int, label: str) -> bool:
        has_nan = torch.isnan(tensor).any().item()
        has_inf = torch.isinf(tensor).any().item()
        if has_nan:
            self.nan_count += 1
            if self.first_nan_iter is None:
                self.first_nan_iter = iteration
            self.details.append({
                "type": "nan", "label": label, "iteration": iteration,
                "count": torch.isnan(tensor).sum().item(),
            })
        if has_inf:
            self.inf_count += 1
            self.details.append({
                "type": "inf", "label": label, "iteration": iteration,
                "count": torch.isinf(tensor).sum().item(),
            })
        return has_nan or has_inf

    @property
    def any_issues(self) -> bool:
        return self.nan_count > 0 or self.inf_count > 0


# =========================================================================
# Main
# =========================================================================


def run_eval_loop(args: argparse.Namespace) -> CorruptionChecker:
    rank = 0
    world_size = 1
    if args.distributed:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

    device = torch.device(torch.cuda.current_device())
    dtype = torch.bfloat16

    if rank == 0:
        log.info("=" * 70)
        log.info("META NaN ISSUE A REPRODUCER: TorchRec-style Pipeline")
        log.info("=" * 70)
        log.info(f"  pipeline: TorchRec TrainPipelineSparseDist pattern")
        log.info(f"  buffer management: .to(device) + record_stream + deque popleft")
        log.info(f"  world_size={world_size}, iterations={args.iterations}")
        log.info(f"  batch_size={args.batch_size}, seq_len={args.seq_len}, candidates={args.num_candidates}")
        log.info(f"  embedding_dim={args.embedding_dim}, attention_layers={args.num_attention_layers}")
        log.info(f"  item_hash_size={args.item_hash_size}, user_hash_size={args.user_hash_size}")
        log.info(f"  torch.compile={'disabled' if args.no_compile else 'enabled'}")
        log.info(f"  record_stream={'disabled' if args.no_record_stream else 'enabled'}")
        log.info(f"  sync_interval={args.sync_interval}")
        log.info(f"  fast_data={args.fast_data}")
        log.info(f"  ROC_AQL_QUEUE_SIZE={os.environ.get('ROC_AQL_QUEUE_SIZE', '(not set, default ~16K)')}")
        log.info(f"  GPU_MAX_HW_QUEUES={os.environ.get('GPU_MAX_HW_QUEUES', '(not set)')}")
        log.info("=" * 70)

    # --- Model ---
    model = HSTUModel(
        item_hash_size=args.item_hash_size,
        user_hash_size=args.user_hash_size,
        category_size=128,
        embedding_dim=args.embedding_dim,
        num_attention_layers=args.num_attention_layers,
        num_heads=4,
        attn_qk_dim=128,
        max_seq_len=args.seq_len,
        max_candidates=args.num_candidates,
        dtype=dtype,
    ).to(device=device, dtype=dtype)
    model.eval()

    if not args.no_compile:
        if rank == 0:
            log.info("Compiling model with torch.compile...")
        model = torch.compile(model)

    # --- Data pipeline ---
    data_config = DLRMv3DataConfig(
        item_hash_size=args.item_hash_size,
        user_hash_size=args.user_hash_size,
        num_inference_candidates=args.num_candidates,
    )
    data_pipeline = ThreadedDataPipeline(
        config=data_config,
        batch_size=args.batch_size,
        max_seq_len=args.seq_len,
        queue_depth=64,
        total_batches=args.iterations + 200,
    )

    # --- Warmup ---
    warmup_gen = DLRMv3SyntheticBatchGenerator(
        config=data_config,
        batch_size=args.batch_size,
        max_seq_len=args.seq_len,
        device=device,
    )
    if rank == 0:
        log.info("Warmup: 10 iterations with sync...")
    for i in range(10):
        batch = warmup_gen.generate_batch(i)
        with torch.no_grad():
            out = model(
                batch["item_ids"], batch["user_ids"], batch["category_ids"],
                batch["candidate_item_ids"], batch["seq_lengths"],
            )
        torch.cuda.synchronize()
    if rank == 0:
        log.info("Warmup complete.")

    # --- TorchRec-style pipeline ---
    pipeline = TorchRecStylePipeline(
        model=model,
        device=device,
        data_pipeline=data_pipeline,
        world_size=world_size,
        embedding_dim=args.embedding_dim,
        batch_size=args.batch_size,
        num_candidates=args.num_candidates,
        use_compile=not args.no_compile,
        use_record_stream=not args.no_record_stream,
        sync_interval=args.sync_interval,
    )

    if args.fast_data:
        pipeline.preload_host_pool(64)

    checker = CorruptionChecker()
    gpu_events = [torch.cuda.Event(enable_timing=False) for _ in range(args.iterations)]

    if rank == 0:
        param_count = sum(p.numel() for p in model.parameters()) / 1e6
        log.info(f"Model parameters: {param_count:.1f}M")
        log.info("")
        log.info("Starting TorchRec-style pipelined eval loop...")
        log.info("  Pipeline layout per iteration (TrainPipelineSparseDist pattern):")
        log.info("    1. wait_for_batch  →  default_stream.wait_stream(memcpy_stream) + record_stream")
        log.info("    2. start_data_dist →  all_to_all on data_dist_stream")
        log.info("    3. enqueue_batch   →  .to(device) on memcpy_stream (H2D for iter N+2)")
        log.info("    4. forward         →  model(batches[0]) on default_stream")
        log.info("    5. wait_data_dist  →  wait all_to_all completion")
        log.info("    6. metrics         →  update on default_stream")
        log.info("    7. dequeue_batch   →  popleft() — allocator may recycle memory!")
        log.info("")

    start_time = time.time()
    last_log_time = start_time
    lag_samples = []

    for iteration in range(args.iterations):
        output = pipeline.progress(iteration)

        if output is None:
            if rank == 0:
                log.warning(f"Pipeline exhausted at iteration {iteration}")
            break

        gpu_events[iteration].record(torch.cuda.current_stream())

        if (iteration + 1) % args.check_interval == 0:
            torch.cuda.synchronize()

            nan_found = checker.check(output, iteration, "predictions")

            gpu_completed = -1
            for j in range(iteration, max(iteration - 200, -1), -1):
                if gpu_events[j].query():
                    gpu_completed = j
                    break
            lag = iteration - gpu_completed
            lag_samples.append(lag)

            now = time.time()
            if rank == 0 and (nan_found or now - last_log_time >= 5):
                elapsed = now - start_time
                iters_per_sec = (iteration + 1) / elapsed
                status = "NaN!" if nan_found else "ok"
                avg_lag = sum(lag_samples[-10:]) / min(len(lag_samples), 10)
                log.info(
                    f"  iter={iteration + 1}/{args.iterations}  "
                    f"[{status}]  nans={checker.nan_count}  "
                    f"lag={lag} (avg={avg_lag:.0f})  "
                    f"rate={iters_per_sec:.0f} it/s  "
                    f"batches_in_flight={len(pipeline.batches)}  "
                    f"[TorchRec-style: .to()+record_stream+deque]"
                )
                last_log_time = now

            if args.stop_on_first and checker.any_issues:
                if rank == 0:
                    log.error(f"Stopping at iteration {iteration + 1}")
                break

    torch.cuda.synchronize()
    data_pipeline.stop()
    elapsed = time.time() - start_time

    avg_lag = sum(lag_samples) / len(lag_samples) if lag_samples else 0
    max_lag = max(lag_samples) if lag_samples else 0

    if rank == 0:
        log.info("")
        log.info("=" * 70)
        log.info("RESULTS")
        log.info("=" * 70)
        log.info(f"  Pipeline: TorchRec-style (3-stage, deque-based)")
        log.info(f"  Buffer management: .to(device) + record_stream + popleft")
        log.info(f"  Iterations: {iteration + 1}")
        log.info(f"  Elapsed: {elapsed:.1f}s ({(iteration + 1)/elapsed:.0f} it/s)")
        log.info(f"  NaN detections: {checker.nan_count}")
        log.info(f"  Inf detections: {checker.inf_count}")
        log.info(f"  CPU-GPU lag -- avg: {avg_lag:.1f}, max: {max_lag}")
        if checker.first_nan_iter is not None:
            log.info(f"  First NaN at iteration: {checker.first_nan_iter + 1}")

        if checker.any_issues:
            log.info("")
            log.info("VERDICT: CORRUPTION DETECTED — NaN/Inf found!")
            log.info("  This matches Meta's Issue A symptom: silent corruption")
            log.info("  from caching allocator recycling same-type tensors.")
        else:
            log.info("")
            log.info("VERDICT: No corruption detected.")
            if max_lag < 5:
                log.info("  NOTE: CPU-GPU lag is small — the GPU is keeping up.")
                log.info("  Try: --fast-data --batch-size 512 for more AQL pressure.")

        log.info("=" * 70)

    return checker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Meta NaN Issue A: TorchRec-style pipeline reproducer",
    )
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--num-candidates", type=int, default=2048)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--num-attention-layers", type=int, default=5)
    parser.add_argument("--item-hash-size", type=int, default=1_000_000)
    parser.add_argument("--user-hash-size", type=int, default=100_000)

    parser.add_argument("--aql-queue-size", type=int, default=None)
    parser.add_argument("--hw-queues", type=int, default=None)
    parser.add_argument("--sync-interval", type=int, default=0,
                        help="Sync every N iters (0=never).")
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile")
    parser.add_argument("--no-record-stream", action="store_true",
                        help="Disable record_stream (makes corruption more likely)")
    parser.add_argument("--fast-data", action="store_true",
                        help="Pre-generate 64 host batches for instant CPU dispatch")
    parser.add_argument("--check-interval", type=int, default=50)
    parser.add_argument("--stop-on-first", action="store_true", default=True)
    parser.add_argument("--no-stop-on-first", action="store_true")
    parser.add_argument("--single-gpu", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    if args.no_stop_on_first:
        args.stop_on_first = False

    if args.aql_queue_size is not None:
        os.environ["ROC_AQL_QUEUE_SIZE"] = str(args.aql_queue_size)
    if args.hw_queues is not None:
        os.environ["GPU_MAX_HW_QUEUES"] = str(args.hw_queues)

    args.distributed = False
    if not args.single_gpu:
        if "RANK" in os.environ:
            dist.init_process_group(backend="nccl")
            args.distributed = True
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            torch.cuda.set_device(local_rank)
        else:
            log.warning("No RANK env var. Running single-GPU.")

    checker = run_eval_loop(args)

    if args.distributed:
        dist.barrier()
        dist.destroy_process_group()

    sys.exit(1 if checker.any_issues else 0)


if __name__ == "__main__":
    main()
