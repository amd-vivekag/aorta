"""
Meta NaN Issue A reproducer: AQL queue depth / tensor recycling corruption.

Uses a DLRMv3-style HSTU (Hierarchical Sequential Transduction Unit) model
with realistic embedding tables, multi-head attention, variable-length
sequences, and multi-stream pipelined eval -- closely matching Meta's
production RecSys workload.

The key mechanism: with zero CPU-GPU synchronization, the CPU submits
dispatch packets far ahead of the GPU. AMD's 16K AQL queue allows the CPU
to get thousands of dispatches ahead. Kernarg buffers and PyTorch tensor
memory get recycled before the GPU reads them, causing silent corruption.

3-STAGE PIPELINE (matching TorchRec's TrainPipelineSparseDist):

  At any point during steady state, 3 batches are in flight concurrently:

    batches[0]  →  default_stream:   forward + metrics  (iteration N)
    batches[1]  →  datadist_stream:  all_to_all / input_dist  (iteration N+1)
    batches[2]  →  memcpy_stream:    H2D copy from host  (iteration N+2)

  Each progress() step:
    1. Wait memcpy_stream → default_stream  (batch[0] data ready)
    2. Start datadist for batch[1] on datadist_stream
    3. Enqueue batch[2]: H2D copy on memcpy_stream
    4. Forward pass batch[0] on default_stream
    5. Metrics update on default_stream
    6. Wait datadist completion
    7. Dequeue batch[0], shift: [1,2] → [0,1], slot 2 ready for next H2D

  This creates 3x the AQL pressure vs a 2-batch pipeline because dispatches
  for 3 different iterations are submitted across 3 streams concurrently.

Architecture (matching DLRMv3 / MLCommons inference benchmark):
  - 3 embedding tables: item_id (hash_size x 512), user_id (10M x 512),
    item_category (128 x 512)
  - HSTU: 5 attention layers, 4 heads, 128-dim qk, bf16
  - Pipelined eval: H2D on memcpy_stream, datadist on datadist_stream,
    forward + metrics on default stream
  - Variable-length user interaction histories (jagged sequences)
  - torch.compile for Triton kernel generation

Usage:
    # Default (should produce NaN on AMD with large AQL queue)
    torchrun --nproc_per_node=2 scripts/meta_nan_issue_a.py

    # With AQL mitigation (should pass)
    ROC_AQL_QUEUE_SIZE=1024 torchrun --nproc_per_node=2 \
        scripts/meta_nan_issue_a.py --aql-queue-size 1024

    # Single-GPU
    python scripts/meta_nan_issue_a.py --single-gpu

    # Disable torch.compile (eager mode)
    torchrun --nproc_per_node=2 scripts/meta_nan_issue_a.py --no-compile
"""

import argparse
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
# DLRMv3-style HSTU model (standalone, no torchrec dependency)
# =========================================================================


class EmbeddingLookup(nn.Module):
    """Large embedding table with gather-based lookup.

    Generates many kernel dispatches per forward call: one gather per
    feature, plus index clamping and dtype casting. With multiple tables
    this creates the dispatch density seen in production RecSys.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, dtype: torch.dtype):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(
            torch.randn(num_embeddings, embedding_dim, dtype=dtype) * 0.01,
        )

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        idx = indices.clamp(0, self.num_embeddings - 1)
        return F.embedding(self.weight, idx.view(-1)).view(
            *indices.shape, self.embedding_dim
        )


class HSTUAttentionLayer(nn.Module):
    """Single HSTU attention layer (multi-head self-attention + FFN).

    Each layer generates ~10+ kernel dispatches: QKV projection, attention
    scores, softmax, attention output, FFN, layer norm, residual add.
    With 5 layers this creates 50+ dispatches just for the attention stack.
    """

    def __init__(self, embed_dim: int, num_heads: int, qk_dim: int, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.qk_dim = qk_dim
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
    """DLRMv3-style HSTU model for RecSys inference.

    Architecture matches the MLCommons DLRMv3 benchmark:
    - 3 embedding tables (item, user, category)
    - Preprocessor MLP to project concatenated embeddings
    - N-layer multi-head attention (HSTU)
    - Output MLP for prediction

    Each forward pass generates 100+ kernel dispatches from embeddings,
    attention layers, projections, and activations. With torch.compile,
    Triton fuses some ops but still produces many dispatch packets.
    """

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
        B = item_ids.shape[0]

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
# Synthetic data generator (uses DLRMv3 distributions from dlrmv3_synthetic_data.py)
# =========================================================================

from dlrmv3_synthetic_data import DLRMv3DataConfig, DLRMv3SyntheticBatchGenerator, ThreadedDataPipeline


# =========================================================================
# Metric accumulator (AUROC / NE / accuracy style)
# =========================================================================


class MetricAccumulator:
    """Simulates eval metric computation on the default stream.

    Real DLRMv3 inference computes NE, accuracy, and AUC. Each metric
    update generates several kernel dispatches (reductions, comparisons).
    """

    def __init__(self, device: torch.device, dtype: torch.dtype):
        self.total_loss = torch.zeros(1, dtype=torch.float32, device=device)
        self.correct = torch.zeros(1, dtype=torch.int64, device=device)
        self.total = torch.zeros(1, dtype=torch.int64, device=device)
        self.pred_sum = torch.zeros(1, dtype=dtype, device=device)

    def update(self, predictions: torch.Tensor) -> None:
        self.pred_sum += predictions.sum()
        labels = (predictions > 0).long()
        self.correct += labels.sum()
        self.total += predictions.numel()


# =========================================================================
# Datadist: all_to_all on side stream
# =========================================================================


class DataDistSimulator:
    """all_to_all embedding redistribution on datadist_stream.

    In the real workload, sharded embedding outputs are redistributed
    via all_to_all. This creates substantial dispatch density on a
    separate HW queue, filling the AQL queue faster.
    """

    def __init__(
        self,
        tensor_size: int,
        world_size: int,
        device: torch.device,
        dtype: torch.dtype,
        stream: Optional[torch.cuda.Stream] = None,
    ):
        self.world_size = world_size
        self.stream = stream
        self.distributed = world_size > 1 and dist.is_initialized()

        if self.distributed:
            self.send_buf = torch.empty(world_size, tensor_size, dtype=dtype, device=device)
            self.recv_buf = torch.empty_like(self.send_buf)
        else:
            self.src_buf = torch.empty(tensor_size, dtype=dtype, device=device)
            self.dst_buf = torch.empty_like(self.src_buf)

    def run(self, iteration: int) -> Optional[dist.Work]:
        ctx = torch.cuda.stream(self.stream) if self.stream else torch.cuda.stream(torch.cuda.current_stream())
        with ctx:
            if self.distributed:
                rank = dist.get_rank()
                self.send_buf.fill_(float(rank))
                return dist.all_to_all_single(self.recv_buf, self.send_buf, async_op=True)
            else:
                self.src_buf.fill_(float(iteration % 500))
                self.dst_buf.copy_(self.src_buf)
                return None


# =========================================================================
# Double-buffered H2D pipeline
# =========================================================================


class ThreeStageH2DPipeline:
    """3-stage pipelined H2D matching TorchRec's TrainPipelineSparseDist.

    Maintains 3 buffer slots so that 3 batches are in-flight simultaneously:
      slot[0] = compute (forward + metrics on default_stream)
      slot[1] = datadist (all_to_all on datadist_stream)
      slot[2] = H2D copy (memcpy_stream)

    Two modes controlled by --alloc-mode:
    - "fixed": 3 pre-allocated device buffer sets, reused via copy_
      (Meta's TorchRec pattern). Same GPU virtual addresses forever.
    - "alloc": Fresh .to(device) allocation each H2D via caching allocator.
      The allocator REUSES freed memory, creating the address-aliasing race
      when the GPU is far behind (Issue A trigger).
    """

    NUM_SLOTS = 3

    def __init__(
        self,
        data_pipeline: ThreadedDataPipeline,
        memcpy_stream: Optional[torch.cuda.Stream],
        device: torch.device,
        batch_size: int,
        seq_len: int,
        num_candidates: int,
        alloc_mode: str = "alloc",
        fast_data: bool = False,
    ):
        self.data_pipeline = data_pipeline
        self.memcpy_stream = memcpy_stream
        self.device = device
        self.alloc_mode = alloc_mode

        self._host_pool: Optional[List[Dict[str, torch.Tensor]]] = None
        self._pool_idx = 0
        if fast_data:
            log.info("Fast data mode: draining 64 batches from pipeline into pool...")
            self._host_pool = []
            for _ in range(64):
                b = data_pipeline.get_batch()
                if b is None:
                    break
                self._host_pool.append(b)
            log.info(f"Pool ready: {len(self._host_pool)} batches")

        if alloc_mode == "fixed":
            self.slots = [
                self._alloc_device_buffers(batch_size, seq_len, num_candidates)
                for _ in range(self.NUM_SLOTS)
            ]
        else:
            self.slots: List[Dict[str, torch.Tensor]] = [{} for _ in range(self.NUM_SLOTS)]

    def _alloc_device_buffers(self, B: int, S: int, C: int) -> Dict[str, torch.Tensor]:
        return {
            "item_ids": torch.zeros(B, S, dtype=torch.long, device=self.device),
            "user_ids": torch.zeros(B, S, dtype=torch.long, device=self.device),
            "category_ids": torch.zeros(B, S, dtype=torch.long, device=self.device),
            "candidate_item_ids": torch.zeros(B, C, dtype=torch.long, device=self.device),
            "seq_lengths": torch.zeros(B, dtype=torch.long, device=self.device),
        }

    def _get_host_batch(self) -> Dict[str, torch.Tensor]:
        if self._host_pool is not None:
            batch = self._host_pool[self._pool_idx % len(self._host_pool)]
            self._pool_idx += 1
            return batch
        batch = self.data_pipeline.get_batch()
        if batch is None:
            raise RuntimeError("Data pipeline exhausted or stalled")
        return batch

    def h2d_into_slot(self, slot_idx: int) -> None:
        """Start async H2D copy into the given slot on memcpy_stream."""
        stream = self.memcpy_stream or torch.cuda.current_stream()
        with torch.cuda.stream(stream):
            if self.alloc_mode == "fixed":
                host_batch = self._get_host_batch()
                for key in self.slots[slot_idx]:
                    self.slots[slot_idx][key].copy_(host_batch[key], non_blocking=True)
            else:
                self.slots[slot_idx] = {
                    k: v.to(self.device, non_blocking=True)
                    for k, v in self._get_host_batch().items()
                }

    def get_compute_batch(self) -> Dict[str, torch.Tensor]:
        """Slot 0 is always the compute batch."""
        return self.slots[0]

    def rotate(self) -> None:
        """Shift slots: [0,1,2] → [1,2,0]. Slot 0 (just consumed) becomes slot 2."""
        self.slots = [self.slots[1], self.slots[2], self.slots[0]]

    def stop(self) -> None:
        self.data_pipeline.stop()


# =========================================================================
# NaN / corruption checker
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
# Main eval loop
# =========================================================================


def run_eval_loop(args: argparse.Namespace) -> CorruptionChecker:
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
        log.info("META NaN ISSUE A REPRODUCER: AQL Queue Depth (DLRMv3-style)")
        log.info("=" * 70)
        log.info(f"  pipeline: 3-STAGE (H2D → datadist → compute, 3 batches in flight)")
        log.info(f"  world_size={world_size}, iterations={args.iterations}")
        log.info(f"  batch_size={args.batch_size}, seq_len={args.seq_len}, candidates={args.num_candidates}")
        log.info(f"  embedding_dim={args.embedding_dim}, attention_layers={args.num_attention_layers}")
        log.info(f"  item_hash_size={args.item_hash_size}, user_hash_size={args.user_hash_size}")
        log.info(f"  torch.compile={'disabled' if args.no_compile else 'enabled'}")
        log.info(f"  side_streams={'disabled' if args.no_side_streams else 'enabled'}")
        log.info(f"  alloc_mode={args.alloc_mode}")
        log.info(f"  sync_interval={args.sync_interval}")
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

    # --- Streams ---
    default_stream = torch.cuda.current_stream()
    if args.no_side_streams:
        memcpy_stream = None
        datadist_stream = None
    else:
        memcpy_stream = torch.cuda.Stream()
        datadist_stream = torch.cuda.Stream()

    # --- Data pipeline (threaded, DLRMv3-realistic distributions) ---
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

    # For warmup, generate directly on GPU (not from the pipeline)
    warmup_gen = DLRMv3SyntheticBatchGenerator(
        config=data_config,
        batch_size=args.batch_size,
        max_seq_len=args.seq_len,
        device=device,
    )

    # --- 3-stage pipeline ---
    h2d = ThreeStageH2DPipeline(
        data_pipeline, memcpy_stream, device,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_candidates=args.num_candidates,
        alloc_mode=args.alloc_mode,
        fast_data=args.fast_data,
    )

    # --- Datadist ---
    datadist_size = args.batch_size * args.num_candidates * args.embedding_dim // 4
    datadist = DataDistSimulator(datadist_size, world_size, device, dtype, datadist_stream)

    # --- Metrics ---
    metrics = MetricAccumulator(device, dtype)
    checker = CorruptionChecker()

    if rank == 0:
        param_count = sum(p.numel() for p in model.parameters()) / 1e6
        log.info(f"Model parameters: {param_count:.1f}M")

    # --- Warmup (with sync) ---
    if rank == 0:
        log.info("Warmup: 10 iterations with sync...")
    for i in range(10):
        batch = warmup_gen.generate_batch(i)
        with torch.no_grad():
            out = model(
                batch["item_ids"], batch["user_ids"], batch["category_ids"],
                batch["candidate_item_ids"], batch["seq_lengths"],
            )
        metrics.update(out)
        torch.cuda.synchronize()
    if rank == 0:
        log.info("Warmup complete.")

    # --- CPU-GPU lag measurement ---
    gpu_events = [torch.cuda.Event(enable_timing=False) for _ in range(args.iterations)]

    # ===================================================================
    # Bootstrap: fill all 3 pipeline slots before entering steady state.
    #   slot[0] = compute-ready  (H2D done, datadist done)
    #   slot[1] = datadist-ready (H2D done, datadist in-progress)
    #   slot[2] = h2d-ready      (H2D in-progress)
    # ===================================================================
    if rank == 0:
        log.info("")
        log.info("Bootstrapping 3-stage pipeline (filling 3 slots)...")

    # slot 0: H2D synchronously
    h2d.h2d_into_slot(0)
    if memcpy_stream:
        default_stream.wait_stream(memcpy_stream)

    # slot 1: H2D on memcpy_stream (will be used for datadist next)
    h2d.h2d_into_slot(1)

    # slot 2: H2D on memcpy_stream (deepest prefetch)
    h2d.h2d_into_slot(2)

    if rank == 0:
        log.info("Bootstrap complete. Starting 3-stage pipelined eval loop...")
        log.info("")
        log.info("  Pipeline layout per iteration:")
        log.info("    slot[0] → default_stream:   forward + metrics  (iter N)")
        log.info("    slot[1] → datadist_stream:   all_to_all         (iter N+1)")
        log.info("    slot[2] → memcpy_stream:     H2D copy           (iter N+2)")
        log.info("")

    start_time = time.time()
    last_log_time = start_time
    lag_samples = []

    for iteration in range(args.iterations):
        # ------------------------------------------------------------------
        # Step 1: Wait for slot[0]'s H2D to be done (from bootstrap or prev
        #         iteration's prefetch). Data for this iteration is ready.
        # ------------------------------------------------------------------
        if memcpy_stream:
            default_stream.wait_stream(memcpy_stream)

        # ------------------------------------------------------------------
        # Step 2: Kick off datadist for slot[1] on datadist_stream.
        #         This processes iter N+1's embedding redistribution while
        #         iter N's forward pass runs on default_stream.
        # ------------------------------------------------------------------
        dd_work = datadist.run(iteration)

        # ------------------------------------------------------------------
        # Step 3: Start H2D for the NEXT slot[2] on memcpy_stream.
        #         After rotate(), the current slot[2] will become slot[0]
        #         in 2 iterations. This is the deepest prefetch (iter N+2).
        # ------------------------------------------------------------------
        if iteration + 3 < args.iterations:
            h2d.h2d_into_slot(2)

        # ------------------------------------------------------------------
        # Step 4: Forward pass on default_stream using slot[0]'s data.
        #         Generates 100+ dispatch packets from the HSTU model.
        # ------------------------------------------------------------------
        batch = h2d.get_compute_batch()
        with torch.no_grad():
            output = model(
                batch["item_ids"], batch["user_ids"], batch["category_ids"],
                batch["candidate_item_ids"], batch["seq_lengths"],
            )

        # ------------------------------------------------------------------
        # Step 5: Metric updates on default_stream.
        # ------------------------------------------------------------------
        metrics.update(output)

        # ------------------------------------------------------------------
        # Step 6: Wait for datadist to complete.
        # ------------------------------------------------------------------
        if dd_work is not None:
            if datadist_stream:
                default_stream.wait_stream(datadist_stream)
            dd_work.wait()

        # ------------------------------------------------------------------
        # Step 7: Rotate slots: [0,1,2] → [1,2,0].
        #         Slot 0 (just consumed) moves to position 2 for reuse
        #         as the next H2D target. Previous slot[1] (datadist done)
        #         becomes the new compute slot[0].
        # ------------------------------------------------------------------
        h2d.rotate()

        # Record GPU event for lag measurement
        gpu_events[iteration].record(default_stream)

        # Optional sync mitigation
        if args.sync_interval > 0 and (iteration + 1) % args.sync_interval == 0:
            torch.cuda.synchronize()

        # Periodic check
        if (iteration + 1) % args.check_interval == 0:
            torch.cuda.synchronize()

            nan_found = checker.check(output, iteration, "predictions")

            # Measure CPU-GPU lag
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
                    f"[3-stage: H2D/datadist/compute]"
                )
                last_log_time = now

            if args.stop_on_first and checker.any_issues:
                if rank == 0:
                    log.error(f"Stopping at iteration {iteration + 1}")
                break

    torch.cuda.synchronize()
    h2d.stop()
    elapsed = time.time() - start_time

    avg_lag = sum(lag_samples) / len(lag_samples) if lag_samples else 0
    max_lag = max(lag_samples) if lag_samples else 0

    if rank == 0:
        log.info("")
        log.info("=" * 70)
        log.info("RESULTS")
        log.info("=" * 70)
        log.info(f"  Pipeline: 3-stage (H2D → datadist → compute)")
        log.info(f"  Batches in flight: 3")
        log.info(f"  Iterations: {iteration + 1}")
        log.info(f"  Elapsed: {elapsed:.1f}s ({(iteration + 1)/elapsed:.0f} it/s)")
        log.info(f"  NaN detections: {checker.nan_count}")
        log.info(f"  Inf detections: {checker.inf_count}")
        log.info(f"  CPU-GPU lag -- avg: {avg_lag:.1f}, max: {max_lag}")
        if checker.first_nan_iter is not None:
            log.info(f"  First NaN at iteration: {checker.first_nan_iter + 1}")

        if checker.any_issues:
            log.info("")
            log.info("VERDICT: CORRUPTION DETECTED.")
        else:
            log.info("")
            log.info("VERDICT: No corruption detected.")
            if max_lag < 5:
                log.info("  NOTE: CPU-GPU lag is small. The GPU is keeping up.")
                log.info("  Try --batch-size 512 --seq-len 256 for more stress.")

        log.info("=" * 70)

    return checker


# =========================================================================
# CLI
# =========================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Meta NaN Issue A: AQL queue depth reproducer (DLRMv3-style)",
    )

    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size (DLRMv3 uses 10-128). Default: 64")
    parser.add_argument("--seq-len", type=int, default=128,
                        help="Max sequence length for UIH. Default: 128")
    parser.add_argument("--num-candidates", type=int, default=2048,
                        help="Number of candidate items per query. Default: 2048 (DLRMv3 inference)")
    parser.add_argument("--embedding-dim", type=int, default=256,
                        help="Embedding dimension (DLRMv3 uses 512). Default: 256")
    parser.add_argument("--num-attention-layers", type=int, default=5,
                        help="Number of HSTU attention layers. Default: 5")
    parser.add_argument("--item-hash-size", type=int, default=1_000_000,
                        help="Item embedding table size (DLRMv3: 1B). Default: 1M")
    parser.add_argument("--user-hash-size", type=int, default=100_000,
                        help="User embedding table size (DLRMv3: 10M). Default: 100K")
    parser.add_argument("--datadist-size", type=int, default=500_000,
                        help="Datadist tensor size for all_to_all. Default: 500K")

    # Mitigation flags
    parser.add_argument("--aql-queue-size", type=int, default=None)
    parser.add_argument("--no-side-streams", action="store_true")
    parser.add_argument("--hw-queues", type=int, default=None)
    parser.add_argument("--sync-interval", type=int, default=0,
                        help="Sync every N iters (0=never). Default: 0")
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile (eager mode)")
    parser.add_argument("--alloc-mode", choices=["alloc", "fixed"], default="alloc",
                        help="H2D buffer strategy: 'alloc' = fresh .to() each iter "
                             "(caching allocator recycles, triggers Issue A); "
                             "'fixed' = pre-allocated copy_ (Meta TorchRec pattern). Default: alloc")
    parser.add_argument("--fast-data", action="store_true",
                        help="Pre-generate data pool (64 batches) so CPU dispatch path is instant. "
                             "Required for Issue A crash repro -- without it, CPU blocks on data gen.")
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
