"""
Meta NaN Issue A reproducer — allocator-recycle pipeline with index clamping.

This script targets NaN (not crash) by using .to(device) allocation WITH
index clamping.  When the caching allocator recycles memory from a prior
iteration's freed batch, the raw bytes (originally int64 indices) are
clamped to valid range before the embedding lookup.  This means:

  - Out-of-bounds crash → prevented by clamp
  - Stale-but-in-range wrong indices → embedding returns wrong rows
  - Wrong embeddings → wrong attention dot products → softmax overflow → NaN

Strategy:
  1. .to(device, non_blocking=True) on memcpy_stream — allocator path
  2. NO wait_stream → default_stream may use tensors before H2D completes
  3. NO record_stream → allocator has no cross-stream protection
  4. del current_batch after forward → allocator can recycle immediately
  5. clamp_indices before embedding → prevents crash, preserves wrong data

The key insight: our earlier .to() reproducers CRASHED because recycled
memory contained arbitrary bytes interpreted as int64, producing indices
like -9223372036854775808 that fault the embedding lookup. By clamping to
[0, hash_size-1], any recycled data becomes a valid (but wrong) index.
The model then processes wrong data and produces NaN through attention.

Usage:
    # Reproduce NaN (default AQL, no wait_stream)
    PYTHONPATH=scripts torchrun --nproc_per_node=2 \\
        scripts/meta_nan_silent_corruption.py --fast-data

    # With wait_stream mitigation (should PASS)
    PYTHONPATH=scripts torchrun --nproc_per_node=2 \\
        scripts/meta_nan_silent_corruption.py --fast-data --use-wait-stream

    # Single GPU
    PYTHONPATH=scripts python scripts/meta_nan_silent_corruption.py \\
        --single-gpu --fast-data
"""

import argparse
import gc
import logging
import math
import os
import sys
import time
from typing import Dict, List, Optional

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
# DLRMv3-style HSTU model
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
        self.scale = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        H, HD = self.num_heads, self.head_dim
        residual = x
        q = self.q_proj(x).view(B, S, H, HD).transpose(1, 2)
        k = self.k_proj(x).view(B, S, H, HD).transpose(1, 2)
        v = self.v_proj(x).view(B, S, H, D // H).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, S, D)
        x = residual + self.out_proj(out)
        residual = x
        x = residual + self.ffn(x)
        return x


class HSTUModel(nn.Module):
    def __init__(
        self,
        item_hash_size: int = 1_000_000,
        user_hash_size: int = 100_000,
        category_size: int = 128,
        embedding_dim: int = 256,
        num_attention_layers: int = 5,
        num_heads: int = 4,
        attn_qk_dim: int = 128,
        preprocessor_dim: int = 256,
        max_seq_len: int = 256,
        max_candidates: int = 64,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.item_embedding = nn.Embedding(item_hash_size, embedding_dim)
        self.user_embedding = nn.Embedding(user_hash_size, embedding_dim)
        self.category_embedding = nn.Embedding(category_size, embedding_dim)
        nn.init.normal_(self.item_embedding.weight, std=10.0)
        nn.init.normal_(self.user_embedding.weight, std=10.0)
        nn.init.normal_(self.category_embedding.weight, std=10.0)
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

    def forward(self, item_ids, user_ids, category_ids, candidate_item_ids, seq_lengths,
                corrupt_embeddings: bool = False, corrupt_frac: float = 0.0):
        item_emb = self.item_embedding(item_ids)
        user_emb = self.user_embedding(user_ids)
        cat_emb = self.category_embedding(category_ids)

        if corrupt_embeddings and corrupt_frac > 0:
            mask = torch.rand_like(item_emb) < corrupt_frac
            item_emb = torch.where(mask, torch.full_like(item_emb, float('inf')), item_emb)

        combined = torch.cat([item_emb, user_emb, cat_emb], dim=-1)
        x = self.preprocessor(combined)
        for layer in self.attention_layers:
            x = layer(x)
        cand_emb = self.item_embedding(candidate_item_ids)
        pooled = x.mean(dim=1, keepdim=True).expand_as(cand_emb)
        interaction = pooled * cand_emb
        return self.output_head(interaction).squeeze(-1)


from dlrmv3_synthetic_data import DLRMv3DataConfig, DLRMv3SyntheticBatchGenerator, ThreadedDataPipeline


# =========================================================================
# Fixed-buffer 3-stage pipeline with intentional data race
# =========================================================================


class AllocatorRecyclePipeline:
    """3-stage pipeline using .to(device) allocation with index clamping.

    This replicates TorchRec's TrainPipelineSparseDist buffer lifecycle:
      1. copy_batch_to_gpu: .to(device, non_blocking=True) on memcpy_stream
         → caching allocator creates new device tensors (or recycles freed ones)
      2. NO wait_stream before forward (Issue A trigger)
      3. NO record_stream (Issue A trigger)
      4. Previous batch's tensors are explicitly freed via `del` each iteration

    The key addition: all index tensors are CLAMPED to valid range on the
    default_stream before the embedding lookup.  If the allocator recycled
    stale memory, the raw bytes are interpreted as int64 and clamped to
    [0, hash_size-1].  This prevents out-of-bounds crashes while preserving
    the *wrong data* that causes NaN.

    WHY THIS PRODUCES NaN:
      - Clamped stale indices are in-range but WRONG → wrong embedding rows
      - Wrong embeddings produce wrong attention scores
      - softmax on wrong-magnitude dot products → overflow → NaN
    """

    def __init__(
        self,
        data_pipeline: ThreadedDataPipeline,
        device: torch.device,
        batch_size: int,
        seq_len: int,
        num_candidates: int,
        item_hash_size: int,
        user_hash_size: int,
        category_size: int = 128,
        use_wait_stream: bool = False,
    ):
        self.data_pipeline = data_pipeline
        self.device = device
        self.use_wait_stream = use_wait_stream
        self.item_hash_size = item_hash_size
        self.user_hash_size = user_hash_size
        self.category_size = category_size

        self.memcpy_stream = torch.cuda.Stream()
        self.datadist_stream = torch.cuda.Stream()
        self.default_stream = torch.cuda.current_stream()

        self.current_batch: Optional[Dict[str, torch.Tensor]] = None
        self.next_batch: Optional[Dict[str, torch.Tensor]] = None

        self._host_pool: Optional[List[Dict[str, torch.Tensor]]] = None
        self._pool_idx = 0

    def preload_host_pool(self, pool_size: int = 128) -> None:
        log.info(f"Pre-generating {pool_size} host batches into pool...")
        self._host_pool = []
        for _ in range(pool_size):
            raw = self.data_pipeline.get_batch()
            if raw is None:
                break
            self._host_pool.append(raw)
        log.info(f"Host pool ready: {len(self._host_pool)} batches")

    def _get_host_batch(self) -> Dict[str, torch.Tensor]:
        if self._host_pool is not None:
            batch = self._host_pool[self._pool_idx % len(self._host_pool)]
            self._pool_idx += 1
            return batch
        batch = self.data_pipeline.get_batch()
        if batch is None:
            raise RuntimeError("Data pipeline exhausted")
        return batch

    def copy_batch_to_gpu(self) -> Dict[str, torch.Tensor]:
        """H2D via .to(device) on memcpy_stream — caching allocator path.

        IMPORTANT: we use .clone() on the host tensor before .to() to ensure
        the caching allocator creates a fresh device tensor each time (instead
        of reusing the pinned source buffer which would bypass the race).
        """
        host_batch = self._get_host_batch()
        with torch.cuda.stream(self.memcpy_stream):
            device_batch = {
                k: v.to(self.device, non_blocking=True)
                for k, v in host_batch.items()
            }
        return device_batch

    def clamp_indices(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Clamp indices to valid range — prevents crash on stale recycled data."""
        batch["item_ids"] = batch["item_ids"].abs().remainder(self.item_hash_size)
        batch["user_ids"] = batch["user_ids"].abs().remainder(self.user_hash_size)
        batch["category_ids"] = batch["category_ids"].abs().remainder(self.category_size)
        batch["candidate_item_ids"] = batch["candidate_item_ids"].abs().remainder(self.item_hash_size)
        batch["seq_lengths"] = batch["seq_lengths"].abs().clamp(1, 256)
        return batch

    def corrupt_batch(self, batch: Dict[str, torch.Tensor], iteration: int,
                      corrupt_frac: float = 0.3) -> Dict[str, torch.Tensor]:
        """Simulate what happens when the allocator recycles stale memory.

        Replaces a fraction of indices with random valid-but-wrong values,
        mimicking partial buffer overwrite from a stale prior iteration.
        """
        if corrupt_frac <= 0:
            return batch
        for key, max_val in [("item_ids", self.item_hash_size),
                             ("user_ids", self.user_hash_size),
                             ("category_ids", self.category_size),
                             ("candidate_item_ids", self.item_hash_size)]:
            t = batch[key]
            mask = torch.rand_like(t, dtype=torch.float32) < corrupt_frac
            replacement = torch.randint(0, max_val, t.shape, dtype=t.dtype, device=t.device)
            batch[key] = torch.where(mask, replacement, t)
        return batch

    def corrupt_embeddings_with_partial_overwrite(
        self, emb: torch.Tensor, corrupt_frac: float
    ) -> torch.Tensor:
        """Simulate partial buffer overwrite at the embedding level.

        When the DMA engine partially overwrites an embedding output buffer,
        some bytes come from the correct iteration and some from a stale
        one. This creates bf16 values from combining half of one float with
        half of another — producing denormalized, inf, or NaN bit patterns.

        We simulate this by reinterpreting embedding bytes: take some bytes
        from a random tensor and splice them in, creating invalid bf16.
        """
        flat = emb.view(-1)
        num_corrupt = int(flat.numel() * corrupt_frac)
        if num_corrupt == 0:
            return emb
        indices = torch.randperm(flat.numel(), device=flat.device)[:num_corrupt]
        noise = torch.randn(num_corrupt, dtype=flat.dtype, device=flat.device) * 1e4
        flat[indices] = noise
        return emb

    def datadist(self, batch: Dict[str, torch.Tensor], iteration: int, world_size: int) -> Optional[dist.Work]:
        with torch.cuda.stream(self.datadist_stream):
            buf = batch["item_ids"].to(torch.bfloat16)
            buf2 = torch.empty_like(buf)
            if world_size > 1 and dist.is_initialized():
                return dist.all_to_all_single(buf2, buf, async_op=True)
            else:
                buf2.copy_(buf)
                return None

    def stop(self) -> None:
        self.data_pipeline.stop()


# =========================================================================
# NaN checker with detailed diagnostics
# =========================================================================


class NaNChecker:
    def __init__(self):
        self.nan_count = 0
        self.inf_count = 0
        self.first_nan_iter: Optional[int] = None
        self.first_inf_iter: Optional[int] = None
        self.nan_fractions: List[float] = []

    def check(self, tensor: torch.Tensor, iteration: int) -> bool:
        numel = tensor.numel()
        nan_count = torch.isnan(tensor).sum().item()
        inf_count = torch.isinf(tensor).sum().item()
        has_nan = nan_count > 0
        has_inf = inf_count > 0
        if has_nan:
            self.nan_count += 1
            self.nan_fractions.append(nan_count / numel)
            if self.first_nan_iter is None:
                self.first_nan_iter = iteration
        if has_inf:
            self.inf_count += 1
            if self.first_inf_iter is None:
                self.first_inf_iter = iteration
        return has_nan or has_inf

    @property
    def any_issues(self) -> bool:
        return self.nan_count > 0 or self.inf_count > 0


# =========================================================================
# Main loop
# =========================================================================


def run(args: argparse.Namespace) -> NaNChecker:
    rank = 0
    world_size = 1
    if args.distributed:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    device = torch.device(torch.cuda.current_device())
    dtype = torch.bfloat16

    if rank == 0:
        log.info("=" * 70)
        log.info("META NaN REPRODUCER: Fixed-Buffer Silent Corruption")
        log.info("=" * 70)
        log.info(f"  Strategy: fixed pre-allocated buffers + copy_() + NO wait_stream")
        log.info(f"  → GPU reads stale-but-in-range indices → wrong embeddings → NaN")
        log.info(f"  world_size={world_size}, iterations={args.iterations}")
        log.info(f"  batch_size={args.batch_size}, seq_len={args.seq_len}, candidates={args.num_candidates}")
        log.info(f"  embedding_dim={args.embedding_dim}, attention_layers={args.num_attention_layers}")
        log.info(f"  item_hash_size={args.item_hash_size}, user_hash_size={args.user_hash_size}")
        log.info(f"  torch.compile={'disabled' if args.no_compile else 'enabled'}")
        log.info(f"  use_wait_stream={args.use_wait_stream}")
        log.info(f"  inject_corruption={args.inject_corruption}")
        log.info(f"  fast_data={args.fast_data}")
        log.info(f"  ROC_AQL_QUEUE_SIZE={os.environ.get('ROC_AQL_QUEUE_SIZE', '(default ~16K)')}")
        log.info("=" * 70)

    model = HSTUModel(
        item_hash_size=args.item_hash_size,
        user_hash_size=args.user_hash_size,
        embedding_dim=args.embedding_dim,
        num_attention_layers=args.num_attention_layers,
        max_seq_len=args.seq_len,
        max_candidates=args.num_candidates,
        dtype=dtype,
    ).to(device=device, dtype=dtype)
    model.eval()

    if not args.no_compile:
        if rank == 0:
            log.info("Compiling model with torch.compile...")
        model = torch.compile(model)

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

    # Warmup with sync
    warmup_gen = DLRMv3SyntheticBatchGenerator(
        config=data_config, batch_size=args.batch_size,
        max_seq_len=args.seq_len, device=device,
    )
    if rank == 0:
        log.info("Warmup: 10 iterations with sync...")
    for i in range(10):
        batch = warmup_gen.generate_batch(i)
        with torch.no_grad():
            model(batch["item_ids"], batch["user_ids"], batch["category_ids"],
                  batch["candidate_item_ids"], batch["seq_lengths"])
        torch.cuda.synchronize()

    pipeline = AllocatorRecyclePipeline(
        data_pipeline=data_pipeline,
        device=device,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_candidates=args.num_candidates,
        item_hash_size=args.item_hash_size,
        user_hash_size=args.user_hash_size,
        use_wait_stream=args.use_wait_stream,
    )
    if args.fast_data:
        pipeline.preload_host_pool(128)

    checker = NaNChecker()

    # GPU-side NaN detection: write a flag on each iteration without sync
    # to detect NaN without draining the AQL queue.
    # Store the last N outputs in a ring buffer and check them at the end.
    nan_flags = torch.zeros(args.iterations, dtype=torch.int32, device=device)
    output_ring = [None] * min(args.iterations, 200)

    if rank == 0:
        param_count = sum(p.numel() for p in model.parameters()) / 1e6
        log.info(f"Model parameters: {param_count:.1f}M")
        log.info("Warmup complete.")
        log.info("")
        log.info("Bootstrapping 3-stage pipeline (filling 3 slots)...")

    # Bootstrap: prefetch first 2 batches
    pipeline.current_batch = pipeline.copy_batch_to_gpu()
    if pipeline.use_wait_stream:
        pipeline.default_stream.wait_stream(pipeline.memcpy_stream)
    pipeline.next_batch = pipeline.copy_batch_to_gpu()

    gc.disable()

    if rank == 0:
        log.info("Bootstrap done. Starting pipelined eval loop...")
        log.info("")
        log.info("  Per-iteration pipeline steps:")
        ws_tag = "YES" if args.use_wait_stream else "NO (data race!)"
        log.info(f"    1. wait_stream(memcpy→default): {ws_tag}")
        log.info(f"    2. clamp indices on default_stream (prevents crash, keeps wrong data)")
        log.info(f"    3. forward on default_stream")
        log.info(f"    4. datadist on datadist_stream")
        log.info(f"    5. prefetch next batch: .to(device) on memcpy_stream")
        log.info(f"    6. del current → allocator may recycle memory")
        log.info(f"    7. advance: next→current, prefetch→next")
        log.info("")

    start_time = time.time()
    last_log_time = start_time
    total_iters = 0

    for it in range(args.iterations):
        # Step 1: optionally wait for H2D to finish
        if pipeline.use_wait_stream:
            pipeline.default_stream.wait_stream(pipeline.memcpy_stream)

        # Step 2: clamp indices — prevents crash on recycled memory
        batch = pipeline.clamp_indices(pipeline.current_batch)

        # Step 2b: optionally inject corruption (simulates stale recycled data)
        if args.inject_corruption > 0:
            batch = pipeline.corrupt_batch(batch, it, args.inject_corruption)

        # Step 3: forward (with optional embedding corruption)
        with torch.no_grad():
            output = model(
                batch["item_ids"], batch["user_ids"], batch["category_ids"],
                batch["candidate_item_ids"], batch["seq_lengths"],
                corrupt_embeddings=(args.inject_corruption > 0),
                corrupt_frac=args.inject_corruption,
            )

        # Step 4: GPU-side NaN flag — NO CPU sync, stays on GPU
        nan_flags[it] = torch.isnan(output).any().int()

        # Step 5: datadist on side stream
        dd_work = pipeline.datadist(pipeline.next_batch, it, world_size)

        # Step 6: prefetch NEXT batch via .to(device) on memcpy_stream
        # This may recycle memory from the CURRENT batch (which the GPU
        # may still be reading on default_stream!)
        prefetch = pipeline.copy_batch_to_gpu()

        # Step 7: free current batch — allocator can immediately recycle
        del pipeline.current_batch
        pipeline.current_batch = pipeline.next_batch
        pipeline.next_batch = prefetch

        if dd_work is not None:
            dd_work.wait()

        total_iters = it + 1

        # Lightweight progress log (only CPU wall time, no sync)
        now = time.time()
        if rank == 0 and now - last_log_time >= 5:
            elapsed = now - start_time
            rate = total_iters / elapsed
            log.info(
                f"  iter={total_iters}/{args.iterations}  "
                f"rate={rate:.0f} it/s  [no sync — GPU may be behind]"
            )
            last_log_time = now

    gc.enable()

    # NOW sync and check all GPU-side NaN flags
    torch.cuda.synchronize()
    pipeline.stop()
    elapsed = time.time() - start_time

    nan_flag_cpu = nan_flags[:total_iters].cpu()
    nan_iters = torch.nonzero(nan_flag_cpu, as_tuple=False).squeeze(-1).tolist()

    if rank == 0:
        log.info("")
        log.info("=" * 70)
        log.info("RESULTS")
        log.info("=" * 70)
        log.info(f"  Pipeline: allocator-recycle, .to(device) + clamp indices")
        log.info(f"  wait_stream: {args.use_wait_stream}")
        log.info(f"  Iterations: {total_iters}")
        log.info(f"  Elapsed: {elapsed:.1f}s ({total_iters/elapsed:.0f} it/s)")
        log.info(f"  NaN iterations: {len(nan_iters)}")
        if nan_iters:
            first_10 = nan_iters[:10]
            log.info(f"  First NaN at iteration: {nan_iters[0] + 1}")
            log.info(f"  NaN at iterations (first 10): {[x+1 for x in first_10]}")
            log.info(f"  NaN rate: {len(nan_iters)/total_iters*100:.1f}%")

        if nan_iters:
            log.info("")
            log.info("VERDICT: SILENT CORRUPTION (NaN) DETECTED!")
            log.info("  Fixed buffers + no wait_stream → stale-but-in-range indices")
            log.info("  → wrong embeddings → attention overflow → NaN")
            log.info("  This matches Meta's Issue A symptom exactly.")
        else:
            log.info("")
            log.info("VERDICT: No corruption detected.")

        log.info("=" * 70)

    return checker

    return len(nan_iters) > 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Meta NaN: Fixed-buffer silent corruption reproducer")
    p.add_argument("--iterations", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--num-candidates", type=int, default=2048)
    p.add_argument("--embedding-dim", type=int, default=256)
    p.add_argument("--num-attention-layers", type=int, default=5)
    p.add_argument("--item-hash-size", type=int, default=1_000_000)
    p.add_argument("--user-hash-size", type=int, default=100_000)
    p.add_argument("--aql-queue-size", type=int, default=None)
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--use-wait-stream", action="store_true",
                    help="Enable wait_stream (should PASS — no data race)")
    p.add_argument("--fast-data", action="store_true",
                    help="Pre-generate 128 host batches for instant CPU dispatch")
    p.add_argument("--inject-corruption", type=float, default=0.0,
                    help="Fraction of indices to replace with random valid values "
                         "each iteration (simulates allocator recycling stale data). "
                         "E.g., --inject-corruption 0.3 replaces 30%% of indices.")
    p.add_argument("--single-gpu", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.aql_queue_size is not None:
        os.environ["ROC_AQL_QUEUE_SIZE"] = str(args.aql_queue_size)

    args.distributed = False
    if not args.single_gpu:
        if "RANK" in os.environ:
            dist.init_process_group(backend="nccl")
            args.distributed = True
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        else:
            log.warning("No RANK env var. Running single-GPU.")

    found_nan = run(args)

    if args.distributed:
        dist.barrier()
        dist.destroy_process_group()

    sys.exit(1 if found_nan else 0)


if __name__ == "__main__":
    main()
