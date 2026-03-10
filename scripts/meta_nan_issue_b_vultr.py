"""
Meta NaN Issue B reproducer — optimized for MI355X (Vultr).

Targets the exact conditions from Meta's problem statement:
  - NaN at bs>=1024 with AQL=1024 (Issue A mitigated)
  - NaN persists even with torch.cuda.synchronize() at every pipeline point
  - NaN disappears ONLY when pipelining is fully disabled
  - NaN appears ~340 iters at bs=1024, from the very beginning at bs=4096

Key mechanisms faithfully replicated:
  1. Triple-buffered device tensors at FIXED virtual addresses, rotated every iter
  2. torch.compile captures the model graph — the SAME compiled function operates
     on slots that rotate underneath it (slot[0] becomes slot[1] next iter)
  3. Real H2D from pinned memory on memcpy_stream (concurrent with forward)
  4. Real all_to_all on datadist_stream (concurrent with forward)
  5. Metric updates on default_stream AFTER forward (creates dispatch density)
  6. Cross-iteration verification: compare pipelined output against eager reference
     to detect SILENT corruption (NaN or wrong values), not just crashes

This script checks for corruption TWO ways:
  - NaN/Inf detection (standard)
  - Value divergence: runs the same input through an EAGER (non-compiled) model
    copy and checks if the compiled pipelined output differs beyond bf16 tolerance.
    This catches silent corruption that produces valid floats but wrong results.

Usage:
    # Default: 2 GPU, bs=1024, compiled, pipelined, stream-wait sync
    PYTHONPATH=scripts ROC_AQL_QUEUE_SIZE=1024 torchrun --nproc_per_node=2 \
        scripts/meta_nan_issue_b_vultr.py --batch-size 1024 --pipelined

    # With full sync at every stage (Issue B should still show NaN per Meta)
    PYTHONPATH=scripts ROC_AQL_QUEUE_SIZE=1024 torchrun --nproc_per_node=2 \
        scripts/meta_nan_issue_b_vultr.py --batch-size 4096 --pipelined --sync-all

    # With per-iteration sync (lighter than sync-all)
    PYTHONPATH=scripts ROC_AQL_QUEUE_SIZE=1024 torchrun --nproc_per_node=2 \
        scripts/meta_nan_issue_b_vultr.py --batch-size 4096 --pipelined --sync-per-iter

    # 8 GPU stress test
    PYTHONPATH=scripts ROC_AQL_QUEUE_SIZE=1024 torchrun --nproc_per_node=8 \
        scripts/meta_nan_issue_b_vultr.py --batch-size 4096 --pipelined --iterations 5000

    # Non-pipelined baseline (should NEVER produce NaN)
    PYTHONPATH=scripts ROC_AQL_QUEUE_SIZE=1024 torchrun --nproc_per_node=2 \
        scripts/meta_nan_issue_b_vultr.py --batch-size 4096

    # Batch size sweep
    for bs in 512 1024 2048 4096 8192; do
        PYTHONPATH=scripts ROC_AQL_QUEUE_SIZE=1024 torchrun --nproc_per_node=2 \
            scripts/meta_nan_issue_b_vultr.py --batch-size $bs --pipelined --iterations 2000
    done
"""

import argparse
import gc
import logging
import math
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
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
# Model — identical architecture to meta_nan_issue_b.py
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
# Host data pool — zero CPU stall
# =========================================================================


def generate_host_pool(
    num_batches: int,
    batch_size: int,
    seq_len: int,
    num_candidates: int,
    item_hash_size: int,
    user_hash_size: int,
    category_size: int,
) -> List[Dict[str, torch.Tensor]]:
    rng = np.random.RandomState(42)
    pool = []
    for _ in range(num_batches):
        pool.append({
            "item_ids": torch.from_numpy(
                rng.randint(0, item_hash_size, (batch_size, seq_len), dtype=np.int64)
            ).pin_memory(),
            "user_ids": torch.from_numpy(
                rng.randint(0, user_hash_size, (batch_size, seq_len), dtype=np.int64)
            ).pin_memory(),
            "category_ids": torch.from_numpy(
                rng.randint(0, category_size, (batch_size, seq_len), dtype=np.int64)
            ).pin_memory(),
            "candidate_item_ids": torch.from_numpy(
                rng.randint(0, item_hash_size, (batch_size, num_candidates), dtype=np.int64)
            ).pin_memory(),
            "seq_lengths": torch.full(
                (batch_size,), seq_len, dtype=torch.long
            ).pin_memory(),
        })
    return pool


# =========================================================================
# 3-stage pipeline with fixed device buffers + slot rotation
# =========================================================================


class ThreeStagePipeline:
    """3-stage pipeline matching TorchRec's TrainPipelineSparseDist.

    Critical for Issue B reproduction:
      - 3 device buffer slots allocated ONCE at fixed GPU virtual addresses
      - Slots rotate every iteration: [0,1,2] → [1,2,0]
      - torch.compile sees slot[0] each time, but the UNDERLYING physical
        buffer is different after rotation
      - H2D writes into slot[2] on memcpy_stream while forward reads slot[0]
        on default_stream (concurrent access to the same buffer pool)
    """

    NUM_SLOTS = 3

    def __init__(self, batch_size, seq_len, num_candidates, device, host_pool):
        self.device = device
        self.host_pool = host_pool
        self.pool_idx = 0
        self.slots = [self._alloc(batch_size, seq_len, num_candidates) for _ in range(self.NUM_SLOTS)]

        if log.isEnabledFor(logging.DEBUG):
            for i, s in enumerate(self.slots):
                ptrs = {k: v.data_ptr() for k, v in s.items()}
                log.debug(f"  slot[{i}] addresses: {ptrs}")

    def _alloc(self, B, S, C):
        return {
            "item_ids": torch.zeros(B, S, dtype=torch.long, device=self.device),
            "user_ids": torch.zeros(B, S, dtype=torch.long, device=self.device),
            "category_ids": torch.zeros(B, S, dtype=torch.long, device=self.device),
            "candidate_item_ids": torch.zeros(B, C, dtype=torch.long, device=self.device),
            "seq_lengths": torch.zeros(B, dtype=torch.long, device=self.device),
        }

    def _next_host(self):
        batch = self.host_pool[self.pool_idx % len(self.host_pool)]
        self.pool_idx += 1
        return batch

    def h2d_into_slot(self, slot_idx, stream):
        host = self._next_host()
        with torch.cuda.stream(stream):
            for key in self.slots[slot_idx]:
                self.slots[slot_idx][key].copy_(host[key], non_blocking=True)

    def fill_slot_sync(self, slot_idx):
        host = self._next_host()
        for key in self.slots[slot_idx]:
            self.slots[slot_idx][key].copy_(host[key])

    def get_compute_batch(self):
        return self.slots[0]

    def get_compute_host_batch(self):
        """Return the host batch corresponding to current slot[0]."""
        idx = (self.pool_idx - self.NUM_SLOTS) % len(self.host_pool)
        return self.host_pool[idx]

    def rotate(self):
        self.slots = [self.slots[1], self.slots[2], self.slots[0]]


# =========================================================================
# Datadist — all_to_all on side stream
# =========================================================================


class DataDist:
    def __init__(self, tensor_size, world_size, device, dtype, stream):
        self.world_size = world_size
        self.stream = stream
        self.distributed = world_size > 1 and dist.is_initialized()

        if self.distributed:
            self.send_buf = torch.empty(world_size, tensor_size, dtype=dtype, device=device)
            self.recv_buf = torch.empty_like(self.send_buf)
        else:
            self.send_buf = torch.empty(tensor_size, dtype=dtype, device=device)
            self.recv_buf = torch.empty_like(self.send_buf)

    def run(self, iteration):
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
# Corruption tracker — NaN + value divergence
# =========================================================================


class CorruptionTracker:
    def __init__(self):
        self.nan_count = 0
        self.inf_count = 0
        self.divergence_count = 0
        self.first_nan_iter = None
        self.first_divergence_iter = None
        self.details: List[str] = []

    def check_nan(self, tensor, iteration, label):
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

    def check_divergence(self, compiled_out, reference_out, iteration, atol=0.05):
        """Check if compiled output diverges from eager reference beyond bf16 tolerance."""
        diff = (compiled_out.float() - reference_out.float()).abs()
        max_diff = diff.max().item()
        if max_diff > atol:
            self.divergence_count += 1
            if self.first_divergence_iter is None:
                self.first_divergence_iter = iteration
            n_bad = (diff > atol).sum().item()
            self.details.append(
                f"iter={iteration} DIVERGENCE: max_diff={max_diff:.4f}, "
                f"bad_elements={n_bad}/{compiled_out.numel()}"
            )
            return True
        return False

    @property
    def any_issues(self):
        return self.nan_count > 0 or self.inf_count > 0 or self.divergence_count > 0


# =========================================================================
# Metrics — dispatch density on default stream
# =========================================================================


class EvalMetrics:
    def __init__(self, device):
        self.total_loss = torch.zeros(1, dtype=torch.float32, device=device)
        self.correct = torch.zeros(1, dtype=torch.int64, device=device)
        self.count = torch.zeros(1, dtype=torch.int64, device=device)
        self.auc_sum = torch.zeros(1, dtype=torch.float32, device=device)

    def update(self, predictions):
        pred_f = predictions.float()
        self.total_loss += pred_f.sum()
        labels = (predictions > 0).long()
        self.correct += labels.sum()
        self.count += predictions.numel()
        self.auc_sum += (pred_f * pred_f).sum()


# =========================================================================
# Pipelined eval loop
# =========================================================================


def run_pipelined(model, eager_model, args, rank, world_size, device, host_pool):
    dtype = torch.bfloat16
    memcpy_stream = torch.cuda.Stream()
    datadist_stream = torch.cuda.Stream()
    default_stream = torch.cuda.current_stream()

    pipeline = ThreeStagePipeline(
        args.batch_size, args.seq_len, args.num_candidates, device, host_pool,
    )

    datadist_size = args.batch_size * args.num_candidates * args.embedding_dim
    datadist = DataDist(datadist_size, world_size, device, dtype, datadist_stream)
    metrics = EvalMetrics(device)
    tracker = CorruptionTracker()

    # Bootstrap all 3 slots
    pipeline.fill_slot_sync(0)
    torch.cuda.synchronize()
    pipeline.h2d_into_slot(1, memcpy_stream)
    pipeline.h2d_into_slot(2, memcpy_stream)

    if args.disable_gc:
        gc.disable()

    start = time.time()
    last_log = start

    for it in range(args.iterations):
        # 1. Wait H2D → default
        default_stream.wait_stream(memcpy_stream)
        if args.sync_all:
            torch.cuda.synchronize()

        # 2. Datadist for slot[1]
        dd_work = datadist.run(it)
        if args.sync_all:
            torch.cuda.synchronize()

        # 3. H2D for slot[2] (deepest prefetch)
        if it + 3 < args.iterations:
            pipeline.h2d_into_slot(2, memcpy_stream)
        if args.sync_all:
            torch.cuda.synchronize()

        # 4. Forward on default_stream using slot[0]
        batch = pipeline.get_compute_batch()
        with torch.no_grad():
            output = model(
                batch["item_ids"], batch["user_ids"], batch["category_ids"],
                batch["candidate_item_ids"], batch["seq_lengths"],
            )
        if args.sync_all:
            torch.cuda.synchronize()

        # 5. Metrics on default stream
        metrics.update(output)

        # 6. Wait datadist
        if dd_work is not None:
            default_stream.wait_stream(datadist_stream)
            dd_work.wait()
        if args.sync_all:
            torch.cuda.synchronize()

        # 7. Rotate: [0,1,2] → [1,2,0]
        pipeline.rotate()

        if args.sync_per_iter:
            torch.cuda.synchronize()

        # Corruption check
        if (it + 1) % args.check_interval == 0:
            torch.cuda.synchronize()
            nan_found = tracker.check_nan(output, it, "predictions")

            # Cross-validate against eager model with same input
            if args.cross_validate and eager_model is not None:
                host_batch = pipeline.get_compute_host_batch()
                fresh = {k: v.to(device) for k, v in host_batch.items()}
                with torch.no_grad():
                    ref_output = eager_model(
                        fresh["item_ids"], fresh["user_ids"], fresh["category_ids"],
                        fresh["candidate_item_ids"], fresh["seq_lengths"],
                    )
                torch.cuda.synchronize()
                div_found = tracker.check_divergence(output, ref_output, it)
                if div_found and rank == 0:
                    log.warning(f"  iter={it+1}: DIVERGENCE detected!")

            now = time.time()
            if rank == 0 and (nan_found or now - last_log >= 5):
                elapsed = now - start
                rate = (it + 1) / elapsed
                status = "NaN!" if nan_found else "ok"
                extra = f"  div={tracker.divergence_count}" if args.cross_validate else ""
                log.info(
                    f"  [pipelined] iter={it+1}/{args.iterations}  "
                    f"[{status}]  nans={tracker.nan_count}{extra}  rate={rate:.1f} it/s"
                )
                last_log = now

            if tracker.any_issues and args.stop_on_first:
                if rank == 0:
                    log.error(f"Stopping at iteration {it + 1}")
                break

    torch.cuda.synchronize()
    if args.disable_gc:
        gc.enable()
    return tracker


# =========================================================================
# Non-pipelined control
# =========================================================================


def run_nonpipelined(model, eager_model, args, rank, world_size, device, host_pool):
    dtype = torch.bfloat16
    datadist_size = args.batch_size * args.num_candidates * args.embedding_dim
    datadist = DataDist(datadist_size, world_size, device, dtype, None)
    metrics = EvalMetrics(device)
    tracker = CorruptionTracker()

    if args.disable_gc:
        gc.disable()

    start = time.time()
    last_log = start

    for it in range(args.iterations):
        host = host_pool[it % len(host_pool)]
        batch = {k: v.to(device, non_blocking=False) for k, v in host.items()}

        dd_work = datadist.run(it)
        with torch.no_grad():
            output = model(
                batch["item_ids"], batch["user_ids"], batch["category_ids"],
                batch["candidate_item_ids"], batch["seq_lengths"],
            )
        metrics.update(output)
        if dd_work is not None:
            dd_work.wait()
        torch.cuda.synchronize()

        if (it + 1) % args.check_interval == 0:
            nan_found = tracker.check_nan(output, it, "predictions")

            if args.cross_validate and eager_model is not None:
                with torch.no_grad():
                    ref_output = eager_model(
                        batch["item_ids"], batch["user_ids"], batch["category_ids"],
                        batch["candidate_item_ids"], batch["seq_lengths"],
                    )
                torch.cuda.synchronize()
                tracker.check_divergence(output, ref_output, it)

            now = time.time()
            if rank == 0 and (nan_found or now - last_log >= 5):
                elapsed = now - start
                rate = (it + 1) / elapsed
                status = "NaN!" if nan_found else "ok"
                extra = f"  div={tracker.divergence_count}" if args.cross_validate else ""
                log.info(
                    f"  [non-pipelined] iter={it+1}/{args.iterations}  "
                    f"[{status}]  nans={tracker.nan_count}{extra}  rate={rate:.1f} it/s"
                )
                last_log = now
            if tracker.any_issues and args.stop_on_first:
                break

    if args.disable_gc:
        gc.enable()
    return tracker


# =========================================================================
# Main
# =========================================================================


def main():
    parser = argparse.ArgumentParser(description="Issue B reproducer (MI355X / Vultr)")
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--num-candidates", type=int, default=2048)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--num-attention-layers", type=int, default=5)
    parser.add_argument("--item-hash-size", type=int, default=1_000_000)
    parser.add_argument("--user-hash-size", type=int, default=100_000)
    parser.add_argument("--pool-size", type=int, default=64)

    parser.add_argument("--pipelined", action="store_true")
    parser.add_argument("--sync-all", action="store_true",
                        help="torch.cuda.synchronize() at every pipeline stage")
    parser.add_argument("--sync-per-iter", action="store_true",
                        help="torch.cuda.synchronize() once per iteration")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--disable-gc", action="store_true")
    parser.add_argument("--cross-validate", action="store_true", default=True,
                        help="Compare compiled output against eager reference (default: on)")
    parser.add_argument("--no-cross-validate", action="store_true")

    parser.add_argument("--check-interval", type=int, default=10)
    parser.add_argument("--stop-on-first", action="store_true", default=True)
    parser.add_argument("--no-stop-on-first", action="store_true")
    parser.add_argument("--single-gpu", action="store_true")

    args = parser.parse_args()
    if args.no_stop_on_first:
        args.stop_on_first = False
    if args.no_cross_validate:
        args.cross_validate = False

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

    rank = dist.get_rank() if args.distributed else 0
    world_size = dist.get_world_size() if args.distributed else 1
    device = torch.cuda.current_device()
    dtype = torch.bfloat16

    if rank == 0:
        log.info("=" * 70)
        log.info("ISSUE B REPRODUCER (MI355X / Vultr)")
        log.info("=" * 70)
        mode = "3-STAGE PIPELINED" if args.pipelined else "NON-PIPELINED"
        sync_mode = "sync-all" if args.sync_all else ("sync-per-iter" if args.sync_per_iter else "stream-wait only")
        log.info(f"  mode: {mode}")
        log.info(f"  sync: {sync_mode}")
        log.info(f"  cross_validate: {args.cross_validate}")
        log.info(f"  world_size={world_size}, iterations={args.iterations}")
        log.info(f"  batch_size={args.batch_size}, seq_len={args.seq_len}, candidates={args.num_candidates}")
        log.info(f"  embedding_dim={args.embedding_dim}, layers={args.num_attention_layers}")
        log.info(f"  item_hash={args.item_hash_size}, user_hash={args.user_hash_size}")
        log.info(f"  torch.compile={'disabled' if args.no_compile else 'enabled'}")
        log.info(f"  disable_gc={args.disable_gc}")
        log.info(f"  pool_size={args.pool_size}")
        log.info(f"  ROC_AQL_QUEUE_SIZE={os.environ.get('ROC_AQL_QUEUE_SIZE', 'not set')}")
        log.info("=" * 70)

    # Build model
    model_kwargs = dict(
        item_hash_size=args.item_hash_size,
        user_hash_size=args.user_hash_size,
        category_size=128,
        embedding_dim=args.embedding_dim,
        num_attention_layers=args.num_attention_layers,
        num_heads=4,
        attn_qk_dim=128,
        max_seq_len=args.seq_len,
        max_candidates=args.num_candidates,
    )

    model = HSTUModel(**model_kwargs).to(device=device, dtype=dtype)
    model.eval()

    # Eager reference model (shares weights — same parameters, no compile)
    eager_model = None
    if args.cross_validate:
        eager_model = model
        if rank == 0:
            log.info("Cross-validation enabled: will compare compiled vs eager output")

    if not args.no_compile:
        if rank == 0:
            log.info("Compiling model with torch.compile...")
        model = torch.compile(model)

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    if rank == 0:
        log.info(f"Model parameters: {param_count:.1f}M")

    # Pre-generate host data
    if rank == 0:
        log.info(f"Pre-generating {args.pool_size} host batches...")
    t0 = time.time()
    host_pool = generate_host_pool(
        args.pool_size, args.batch_size, args.seq_len, args.num_candidates,
        args.item_hash_size, args.user_hash_size, 128,
    )
    if rank == 0:
        log.info(f"Host pool ready in {time.time() - t0:.1f}s")

    # Warmup
    if rank == 0:
        log.info("Warmup (5 iterations)...")
    dev = torch.device(f"cuda:{torch.cuda.current_device()}")
    for i in range(5):
        b = {k: v.to(dev) for k, v in host_pool[i % len(host_pool)].items()}
        with torch.no_grad():
            model(b["item_ids"], b["user_ids"], b["category_ids"],
                  b["candidate_item_ids"], b["seq_lengths"])
        torch.cuda.synchronize()
    if rank == 0:
        log.info("Warmup complete.")

    # Run
    t_start = time.time()
    if args.pipelined:
        tracker = run_pipelined(model, eager_model, args, rank, world_size, device, host_pool)
    else:
        tracker = run_nonpipelined(model, eager_model, args, rank, world_size, device, host_pool)
    elapsed = time.time() - t_start

    # Results
    if rank == 0:
        log.info("")
        log.info("=" * 70)
        log.info("RESULTS")
        log.info("=" * 70)
        mode = "3-STAGE PIPELINED" if args.pipelined else "NON-PIPELINED"
        log.info(f"  Mode: {mode}")
        log.info(f"  Batch size: {args.batch_size}")
        log.info(f"  torch.compile={'enabled' if not args.no_compile else 'disabled'}")
        sync_mode = "sync-all" if args.sync_all else ("sync-per-iter" if args.sync_per_iter else "stream-wait only")
        log.info(f"  sync: {sync_mode}")
        log.info(f"  Elapsed: {elapsed:.1f}s ({args.iterations / max(elapsed, 0.1):.1f} it/s)")
        log.info(f"  NaN detections: {tracker.nan_count}")
        log.info(f"  Inf detections: {tracker.inf_count}")
        log.info(f"  Divergences: {tracker.divergence_count}")
        if tracker.first_nan_iter is not None:
            log.info(f"  First NaN at iteration: {tracker.first_nan_iter + 1}")
        if tracker.first_divergence_iter is not None:
            log.info(f"  First divergence at iteration: {tracker.first_divergence_iter + 1}")
        for d in tracker.details[:20]:
            log.info(f"    {d}")

        if tracker.any_issues:
            log.info("")
            log.info(f"VERDICT: CORRUPTION DETECTED in {mode} mode.")
            if tracker.nan_count > 0:
                log.info("  NaN detected — matches Issue B pattern.")
            if tracker.divergence_count > 0:
                log.info("  Value divergence — compiled pipelined output differs from eager.")
        else:
            log.info("")
            log.info(f"VERDICT: No corruption in {mode} mode at bs={args.batch_size}.")

        log.info("=" * 70)

    if args.distributed:
        dist.barrier()
        dist.destroy_process_group()

    sys.exit(1 if tracker.any_issues else 0)


if __name__ == "__main__":
    main()
