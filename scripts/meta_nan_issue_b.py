"""
Meta NaN Issue B reproducer: Large Batch + Pipelining NaN.

Uses a DLRMv3-style HSTU model with torch.compile and pipelined eval
where buffer objects are REUSED across iterations (same virtual addresses).

Issue B persists even when Issue A is fully mitigated (AQL=1024) and
torch.cuda.synchronize() is called at every pipeline point. It
disappears only when pipelining is disabled entirely.

3-STAGE PIPELINE (matching TorchRec's TrainPipelineSparseDist):

  At any point during steady state, 3 batches are in flight concurrently:

    slot[0]  →  default_stream:   forward + metrics  (iteration N)
    slot[1]  →  datadist_stream:  all_to_all / input_dist  (iteration N+1)
    slot[2]  →  memcpy_stream:    H2D copy from host  (iteration N+2)

  This is the exact pipeline depth used in Meta's eval workload, where
  the CPU stays 3 iterations ahead of the GPU at all times. Previous
  versions of this script used 2-batch double-buffering; this version
  now matches the real TorchRec TrainPipelineSparseDist architecture.

Key mechanisms modeled:
  1. Triple-buffered device tensors with SAME virtual addresses reused
  2. Real H2D transfers from CPU pinned memory (not device-side randint)
  3. Prefetch for iteration N+2 overlaps forward for iteration N
  4. torch.compile captures buffer tensor references in compiled graph
  5. all_to_all redistribution of embedding results on side stream
  6. Multiple embedding table lookups (high dispatch density)

Hypotheses being tested:
  A) torch.compile / Triton codegen bug at large tensor dimensions
  B) HIP memory coherence / cache visibility bug with buffer reuse
  C) Pipeline buffer management HIP interaction

Usage:
    # Pipelined mode, batch-size sweep
    for bs in 512 1024 2048 4096; do
        ROC_AQL_QUEUE_SIZE=1024 torchrun --nproc_per_node=2 \
            scripts/meta_nan_issue_b.py --batch-size $bs --pipelined
    done

    # Pipelined + full sync at every point (Issue B should still show NaN)
    ROC_AQL_QUEUE_SIZE=1024 torchrun --nproc_per_node=2 \
        scripts/meta_nan_issue_b.py --batch-size 4096 --pipelined --sync-all

    # Non-pipelined baseline (should never NaN)
    ROC_AQL_QUEUE_SIZE=1024 torchrun --nproc_per_node=2 \
        scripts/meta_nan_issue_b.py --batch-size 4096

    # Without torch.compile (test Triton hypothesis)
    ROC_AQL_QUEUE_SIZE=1024 torchrun --nproc_per_node=2 \
        scripts/meta_nan_issue_b.py --batch-size 4096 --pipelined --no-compile

    # 8 GPUs, high stress
    ROC_AQL_QUEUE_SIZE=1024 torchrun --nproc_per_node=8 \
        scripts/meta_nan_issue_b.py --batch-size 4096 --pipelined --iterations 5000
"""

import argparse
import gc
import logging
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

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
    """DLRMv3-style HSTU model with embeddings + attention + output head."""

    def __init__(
        self,
        item_hash_size: int,
        user_hash_size: int,
        category_size: int,
        embedding_dim: int,
        num_attention_layers: int,
        num_heads: int,
        attn_qk_dim: int,
        max_seq_len: int,
        max_candidates: int,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.item_embedding = nn.Embedding(item_hash_size, embedding_dim)
        self.user_embedding = nn.Embedding(user_hash_size, embedding_dim)
        self.category_embedding = nn.Embedding(category_size, embedding_dim)

        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.category_embedding.weight, std=0.01)

        self.preprocessor = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim * 2, bias=False),
            nn.GELU(),
            nn.Linear(embedding_dim * 2, embedding_dim, bias=False),
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
        return self.output_head(interaction).squeeze(-1)


# =========================================================================
# Pipeline buffers with real H2D from pinned memory
# =========================================================================


from dlrmv3_synthetic_data import DLRMv3DataConfig, DLRMv3SyntheticBatchGenerator, ThreadedDataPipeline


class ThreeStageEvalPipeline:
    """3-stage pipelined eval buffers matching TorchRec's TrainPipelineSparseDist.

    Maintains 3 buffer slots so that 3 batches are in-flight simultaneously:
      slot[0] = compute  (forward + metrics on default_stream)   — iter N
      slot[1] = datadist (all_to_all on datadist_stream)         — iter N+1
      slot[2] = H2D copy (memcpy_stream)                         — iter N+2

    Device buffers are pre-allocated ONCE and reused every iteration (same GPU
    virtual addresses forever) via copy_. This is critical for Issue B testing:
    torch.compile captures references to these fixed buffers, and the compiled
    graph interacts with the 3-way buffer rotation.

    Host-side data is generated by ThreadedDataPipeline (background thread with
    DLRMv3-realistic distributions) and staged through a pinned host buffer.
    """

    NUM_SLOTS = 3

    def __init__(
        self,
        batch_size: int,
        seq_len: int,
        num_candidates: int,
        item_hash_size: int,
        user_hash_size: int,
        category_size: int,
        device: torch.device,
    ):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_candidates = num_candidates
        self.device = device

        self.slots = [self._alloc_device_buffers() for _ in range(self.NUM_SLOTS)]

        self.host_buf = self._alloc_host_buffers()

        self.data_pipeline = ThreadedDataPipeline(
            config=DLRMv3DataConfig(
                item_hash_size=item_hash_size,
                user_hash_size=user_hash_size,
                num_inference_candidates=num_candidates,
            ),
            batch_size=batch_size,
            max_seq_len=seq_len,
            queue_depth=16,
            total_batches=20000,
        )

    def _alloc_device_buffers(self) -> Dict[str, torch.Tensor]:
        B, S, C = self.batch_size, self.seq_len, self.num_candidates
        return {
            "item_ids": torch.zeros(B, S, dtype=torch.long, device=self.device),
            "user_ids": torch.zeros(B, S, dtype=torch.long, device=self.device),
            "category_ids": torch.zeros(B, S, dtype=torch.long, device=self.device),
            "candidate_item_ids": torch.zeros(B, C, dtype=torch.long, device=self.device),
            "seq_lengths": torch.zeros(B, dtype=torch.long, device=self.device),
        }

    def _alloc_host_buffers(self) -> Dict[str, torch.Tensor]:
        B, S, C = self.batch_size, self.seq_len, self.num_candidates
        return {
            "item_ids": torch.zeros(B, S, dtype=torch.long).pin_memory(),
            "user_ids": torch.zeros(B, S, dtype=torch.long).pin_memory(),
            "category_ids": torch.zeros(B, S, dtype=torch.long).pin_memory(),
            "candidate_item_ids": torch.zeros(B, C, dtype=torch.long).pin_memory(),
            "seq_lengths": torch.zeros(B, dtype=torch.long).pin_memory(),
        }

    def _fetch_host(self) -> None:
        """Get next batch from background thread into pinned host buffers."""
        batch = self.data_pipeline.get_batch()
        if batch is None:
            raise RuntimeError("Data pipeline exhausted or stalled")
        for key in self.host_buf:
            self.host_buf[key].copy_(batch[key])

    def h2d_into_slot(self, slot_idx: int, stream: torch.cuda.Stream) -> None:
        """Fetch host data and start async H2D copy into the given slot."""
        self._fetch_host()
        with torch.cuda.stream(stream):
            for key in self.slots[slot_idx]:
                self.slots[slot_idx][key].copy_(self.host_buf[key], non_blocking=True)

    def fill_slot_sync(self, slot_idx: int) -> None:
        """Synchronously fill a slot (for bootstrap)."""
        self._fetch_host()
        for key in self.slots[slot_idx]:
            self.slots[slot_idx][key].copy_(self.host_buf[key])

    def get_compute_batch(self) -> Dict[str, torch.Tensor]:
        """Slot 0 is always the compute batch."""
        return self.slots[0]

    def rotate(self) -> None:
        """Shift slots: [0,1,2] → [1,2,0]. Consumed slot 0 becomes slot 2."""
        self.slots = [self.slots[1], self.slots[2], self.slots[0]]

    def stop(self) -> None:
        self.data_pipeline.stop()


# =========================================================================
# Datadist: all_to_all on side stream (reused buffers)
# =========================================================================


class PipelinedDataDist:
    """all_to_all redistribution with reused send/recv buffers."""

    def __init__(
        self,
        tensor_size: int,
        world_size: int,
        device: torch.device,
        dtype: torch.dtype,
        stream: Optional[torch.cuda.Stream],
    ):
        self.world_size = world_size
        self.stream = stream
        self.distributed = world_size > 1 and dist.is_initialized()

        if self.distributed:
            self.send_buf = torch.empty(world_size, tensor_size, dtype=dtype, device=device)
            self.recv_buf = torch.empty_like(self.send_buf)
        else:
            self.send_buf = torch.empty(tensor_size, dtype=dtype, device=device)
            self.recv_buf = torch.empty_like(self.send_buf)

    def run(self, iteration: int) -> Optional[dist.Work]:
        ctx = torch.cuda.stream(self.stream) if self.stream else torch.cuda.stream(torch.cuda.current_stream())
        with ctx:
            if self.distributed:
                rank = dist.get_rank()
                self.send_buf.fill_(float((rank + 1) * (iteration + 1) % 1000) / 1000.0)
                return dist.all_to_all_single(self.recv_buf, self.send_buf, async_op=True)
            else:
                self.send_buf.fill_(float((iteration + 1) % 1000) / 1000.0)
                self.recv_buf.copy_(self.send_buf)
                return None


# =========================================================================
# NaN tracker (check every iteration for thorough detection)
# =========================================================================


class NaNTracker:
    def __init__(self):
        self.nan_count: int = 0
        self.inf_count: int = 0
        self.first_nan_iter: Optional[int] = None
        self.details: List[str] = []

    def check(self, tensor: torch.Tensor, iteration: int, label: str) -> bool:
        has_nan = torch.isnan(tensor).any().item()
        has_inf = torch.isinf(tensor).any().item()
        if has_nan or has_inf:
            nan_n = torch.isnan(tensor).sum().item()
            inf_n = torch.isinf(tensor).sum().item()
            if has_nan:
                self.nan_count += 1
                if self.first_nan_iter is None:
                    self.first_nan_iter = iteration
            if has_inf:
                self.inf_count += 1
            self.details.append(
                f"iter={iteration} {label}: nan={nan_n}/{tensor.numel()}, inf={inf_n}/{tensor.numel()}"
            )
            return True
        return False

    @property
    def any_issues(self) -> bool:
        return self.nan_count > 0 or self.inf_count > 0


# =========================================================================
# Eval metric accumulator (generates kernel dispatches on default stream)
# =========================================================================


class EvalMetrics:
    """Two metrics computed on default stream, matching Meta's pattern."""

    def __init__(self, device: torch.device):
        self.total_loss = torch.zeros(1, dtype=torch.float32, device=device)
        self.correct = torch.zeros(1, dtype=torch.int64, device=device)
        self.count = torch.zeros(1, dtype=torch.int64, device=device)
        self.auc_sum = torch.zeros(1, dtype=torch.float32, device=device)

    def update(self, predictions: torch.Tensor) -> None:
        pred_f = predictions.float()
        self.total_loss += pred_f.sum()
        labels = (predictions > 0).long()
        self.correct += labels.sum()
        self.count += predictions.numel()
        self.auc_sum += (pred_f * pred_f).sum()


# =========================================================================
# Pipelined eval loop (Meta's pattern)
# =========================================================================


def run_pipelined_eval(
    model: nn.Module,
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    device: torch.device,
) -> NaNTracker:
    """3-stage pipelined eval matching TorchRec's TrainPipelineSparseDist:

    At steady state, 3 batches are in flight across 3 streams:
      slot[0] → default_stream:   forward + metrics   (iteration N)
      slot[1] → datadist_stream:  all_to_all           (iteration N+1)
      slot[2] → memcpy_stream:    H2D from host        (iteration N+2)

    Each progress() step:
      1. wait memcpy_stream → default_stream (slot[0] data ready)
      2. datadist for slot[1] on datadist_stream
      3. H2D for slot[2] on memcpy_stream
      4. forward pass slot[0] on default_stream
      5. metric updates on default_stream
      6. wait datadist
      7. rotate: [0,1,2] → [1,2,0]

    With 3 buffers at fixed addresses and torch.compile, this creates the
    exact conditions of Meta's Issue B: the compiled graph references
    buffer addresses that rotate every iteration.
    """
    dtype = torch.bfloat16

    memcpy_stream = torch.cuda.Stream()
    datadist_stream = torch.cuda.Stream()
    default_stream = torch.cuda.current_stream()

    pipeline = ThreeStageEvalPipeline(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_candidates=args.num_candidates,
        item_hash_size=args.item_hash_size,
        user_hash_size=args.user_hash_size,
        category_size=128,
        device=device,
    )

    datadist_size = args.batch_size * args.num_candidates * args.embedding_dim
    datadist = PipelinedDataDist(datadist_size, world_size, device, dtype, datadist_stream)

    metrics = EvalMetrics(device)
    tracker = NaNTracker()

    # =================================================================
    # Bootstrap: fill all 3 pipeline slots
    # =================================================================
    if rank == 0:
        log.info("Bootstrapping 3-stage pipeline (filling 3 slots)...")

    pipeline.fill_slot_sync(0)
    torch.cuda.synchronize()

    pipeline.h2d_into_slot(1, memcpy_stream)
    pipeline.h2d_into_slot(2, memcpy_stream)

    if rank == 0:
        log.info("Bootstrap complete. 3 batches in flight.")

    if args.disable_gc:
        gc.disable()

    start_time = time.time()
    last_log = start_time

    for iteration in range(args.iterations):
        # Step 1: Wait for slot[0]'s H2D (from bootstrap or previous rotate)
        default_stream.wait_stream(memcpy_stream)
        if args.sync_all:
            torch.cuda.synchronize()

        # Step 2: Datadist for slot[1] on datadist_stream
        dd_work = datadist.run(iteration)
        if args.sync_all:
            torch.cuda.synchronize()

        # Step 3: H2D for slot[2] on memcpy_stream (deepest prefetch, iter N+2)
        if iteration + 3 < args.iterations:
            pipeline.h2d_into_slot(2, memcpy_stream)
        if args.sync_all:
            torch.cuda.synchronize()

        # Step 4: Forward pass on default_stream using slot[0]
        batch = pipeline.get_compute_batch()
        with torch.no_grad():
            output = model(
                batch["item_ids"], batch["user_ids"], batch["category_ids"],
                batch["candidate_item_ids"], batch["seq_lengths"],
            )
        if args.sync_all:
            torch.cuda.synchronize()

        # Step 5: Metrics on default stream
        metrics.update(output)

        # Step 6: Wait datadist completion
        if dd_work is not None:
            default_stream.wait_stream(datadist_stream)
            dd_work.wait()
        if args.sync_all:
            torch.cuda.synchronize()

        # Step 7: Rotate: [0,1,2] → [1,2,0]
        pipeline.rotate()

        # Periodic sync (partial mitigation)
        if args.sync_interval > 0 and (iteration + 1) % args.sync_interval == 0:
            torch.cuda.synchronize()

        # NaN check
        if (iteration + 1) % args.check_interval == 0:
            torch.cuda.synchronize()
            nan_found = tracker.check(output, iteration, "predictions")

            now = time.time()
            if rank == 0 and (nan_found or now - last_log >= 5):
                elapsed = now - start_time
                iters_per_sec = (iteration + 1) / elapsed
                status = "NaN!" if nan_found else "ok"
                log.info(
                    f"  [3-stage pipelined] iter={iteration + 1}/{args.iterations}  "
                    f"[{status}]  nans={tracker.nan_count}  "
                    f"rate={iters_per_sec:.0f} it/s"
                )
                last_log = now

            if tracker.any_issues and args.stop_on_first:
                if rank == 0:
                    log.error(f"Stopping at iteration {iteration + 1}")
                break

    torch.cuda.synchronize()
    pipeline.stop()

    if args.disable_gc:
        gc.enable()

    return tracker


# =========================================================================
# Non-pipelined eval loop (control)
# =========================================================================


def run_nonpipelined_eval(
    model: nn.Module,
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    device: torch.device,
) -> NaNTracker:
    """Non-pipelined eval: each iteration is fully independent.

    Fresh tensors allocated each iteration (different virtual addresses).
    Full synchronization between iterations. This is the control case
    that should never produce NaN.
    """
    dtype = torch.bfloat16
    datadist_size = args.batch_size * args.num_candidates * args.embedding_dim
    datadist = PipelinedDataDist(datadist_size, world_size, device, dtype, None)
    metrics = EvalMetrics(device)
    tracker = NaNTracker()

    if args.disable_gc:
        gc.disable()

    start_time = time.time()
    last_log = start_time

    dev = torch.device(f"cuda:{torch.cuda.current_device()}")

    data_pipeline = ThreadedDataPipeline(
        config=DLRMv3DataConfig(
            item_hash_size=args.item_hash_size,
            user_hash_size=args.user_hash_size,
            num_inference_candidates=args.num_candidates,
        ),
        batch_size=args.batch_size,
        max_seq_len=args.seq_len,
        queue_depth=8,
        total_batches=args.iterations + 50,
    )

    for iteration in range(args.iterations):
        host_batch = data_pipeline.get_batch()
        batch = {k: v.to(dev, non_blocking=False) for k, v in host_batch.items()}

        dd_work = datadist.run(iteration)

        with torch.no_grad():
            output = model(
                batch["item_ids"], batch["user_ids"], batch["category_ids"],
                batch["candidate_item_ids"], batch["seq_lengths"],
            )

        metrics.update(output)

        if dd_work is not None:
            dd_work.wait()

        torch.cuda.synchronize()

        if (iteration + 1) % args.check_interval == 0:
            nan_found = tracker.check(output, iteration, "predictions")
            now = time.time()
            if rank == 0 and (nan_found or now - last_log >= 5):
                elapsed = now - start_time
                iters_per_sec = (iteration + 1) / elapsed
                status = "NaN!" if nan_found else "ok"
                log.info(
                    f"  [non-pipelined] iter={iteration + 1}/{args.iterations}  "
                    f"[{status}]  nans={tracker.nan_count}  "
                    f"rate={iters_per_sec:.0f} it/s"
                )
                last_log = now

            if tracker.any_issues and args.stop_on_first:
                break

    data_pipeline.stop()

    if args.disable_gc:
        gc.enable()

    return tracker


# =========================================================================
# Main
# =========================================================================


def run_test(args: argparse.Namespace) -> NaNTracker:
    rank = 0
    world_size = 1
    if args.distributed:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

    device = torch.cuda.current_device()
    dtype = torch.bfloat16

    if rank == 0:
        log.info("=" * 70)
        log.info("META NaN ISSUE B REPRODUCER: Large Batch + Pipelining (DLRMv3-style)")
        log.info("=" * 70)
        log.info(f"  pipeline: {'3-STAGE (H2D → datadist → compute, 3 batches in flight)' if args.pipelined else 'NONE (non-pipelined control)'}")
        log.info(f"  world_size={world_size}, iterations={args.iterations}")
        log.info(f"  batch_size={args.batch_size}, seq_len={args.seq_len}, candidates={args.num_candidates}")
        log.info(f"  embedding_dim={args.embedding_dim}, attention_layers={args.num_attention_layers}")
        log.info(f"  item_hash_size={args.item_hash_size}, user_hash_size={args.user_hash_size}")
        log.info(f"  torch.compile={'disabled' if args.no_compile else 'enabled'}")
        log.info(f"  sync_all={'YES' if args.sync_all else 'no'}")
        log.info(f"  disable_gc={args.disable_gc}")
        log.info(f"  ROC_AQL_QUEUE_SIZE={os.environ.get('ROC_AQL_QUEUE_SIZE', '(not set)')}")
        log.info("=" * 70)

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
    ).to(device=device, dtype=dtype)
    model.eval()

    if not args.no_compile:
        if rank == 0:
            log.info("Compiling model with torch.compile...")
        model = torch.compile(model)

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    if rank == 0:
        log.info(f"Model parameters: {param_count:.1f}M")

    # Warmup
    if rank == 0:
        log.info("Warmup (10 iterations with sync)...")
    dev = torch.device(f"cuda:{torch.cuda.current_device()}")
    warmup_config = DLRMv3DataConfig(
        item_hash_size=args.item_hash_size,
        user_hash_size=args.user_hash_size,
        num_inference_candidates=args.num_candidates,
    )
    warmup_gen = DLRMv3SyntheticBatchGenerator(
        config=warmup_config,
        batch_size=args.batch_size,
        max_seq_len=args.seq_len,
        device=dev,
    )
    for i in range(10):
        batch = warmup_gen.generate_batch(i)
        with torch.no_grad():
            model(batch["item_ids"], batch["user_ids"], batch["category_ids"],
                  batch["candidate_item_ids"], batch["seq_lengths"])
        torch.cuda.synchronize()
    if rank == 0:
        log.info("Warmup complete.")

    # Run
    if args.pipelined:
        tracker = run_pipelined_eval(model, args, rank, world_size, device)
    else:
        tracker = run_nonpipelined_eval(model, args, rank, world_size, device)

    # Results
    if rank == 0:
        log.info("")
        log.info("=" * 70)
        log.info("RESULTS")
        log.info("=" * 70)
        mode = "3-STAGE PIPELINED" if args.pipelined else "NON-PIPELINED"
        log.info(f"  Mode: {mode}")
        if args.pipelined:
            log.info(f"  Batches in flight: 3")
        log.info(f"  Batch size: {args.batch_size}")
        log.info(f"  torch.compile={'enabled' if not args.no_compile else 'disabled'}")
        log.info(f"  sync_all={args.sync_all}")
        log.info(f"  NaN detections: {tracker.nan_count}")
        log.info(f"  Inf detections: {tracker.inf_count}")
        if tracker.first_nan_iter is not None:
            log.info(f"  First NaN at iteration: {tracker.first_nan_iter + 1}")
        for d in tracker.details[:20]:
            log.info(f"    {d}")

        if tracker.any_issues:
            log.info("")
            log.info(f"VERDICT: NaN/Inf DETECTED in {mode} mode.")
            if args.pipelined:
                log.info("  This matches Issue B pattern.")
                log.info("  Run without --pipelined to confirm it's pipeline-specific.")
                if not args.no_compile:
                    log.info("  Run with --no-compile to test Triton hypothesis.")
        else:
            log.info("")
            if args.pipelined:
                log.info(f"VERDICT: No NaN in pipelined mode at bs={args.batch_size}.")
                log.info("  Try larger --batch-size or more --iterations.")
            else:
                log.info("VERDICT: No NaN in non-pipelined mode (expected).")

        log.info("=" * 70)

    return tracker


# =========================================================================
# CLI
# =========================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Meta NaN Issue B: Large Batch + Pipelining (DLRMv3-style)",
    )

    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Batch size. Test at 128, 512, 1024, 2048, 4096")
    parser.add_argument("--seq-len", type=int, default=128,
                        help="Max UIH sequence length")
    parser.add_argument("--num-candidates", type=int, default=2048,
                        help="Number of candidate items per query")
    parser.add_argument("--embedding-dim", type=int, default=256,
                        help="Embedding dimension")
    parser.add_argument("--num-attention-layers", type=int, default=5)
    parser.add_argument("--item-hash-size", type=int, default=1_000_000)
    parser.add_argument("--user-hash-size", type=int, default=100_000)

    parser.add_argument("--pipelined", action="store_true",
                        help="Enable pipelined eval with buffer reuse")
    parser.add_argument("--sync-all", action="store_true",
                        help="torch.cuda.synchronize() at every pipeline stage")
    parser.add_argument("--sync-interval", type=int, default=0,
                        help="Sync every N iters (0=never, use with pipelined to avoid AQL crash while keeping overlap)")
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile (eager mode)")
    parser.add_argument("--disable-gc", action="store_true",
                        help="Disable Python garbage collector (gc_collect_interval=0)")

    parser.add_argument("--check-interval", type=int, default=10,
                        help="Check for NaN every N iterations")
    parser.add_argument("--stop-on-first", action="store_true", default=True)
    parser.add_argument("--no-stop-on-first", action="store_true")
    parser.add_argument("--single-gpu", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    if args.no_stop_on_first:
        args.stop_on_first = False

    os.environ.setdefault("ROC_AQL_QUEUE_SIZE", "1024")

    args.distributed = False
    if not args.single_gpu:
        if "RANK" in os.environ:
            dist.init_process_group(backend="nccl")
            args.distributed = True
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            torch.cuda.set_device(local_rank)
        else:
            log.warning("No RANK env var. Running single-GPU.")

    tracker = run_test(args)

    if args.distributed:
        dist.barrier()
        dist.destroy_process_group()

    sys.exit(1 if tracker.any_issues else 0)


if __name__ == "__main__":
    main()
