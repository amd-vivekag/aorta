"""
Pipelined eval mode reproducer.

Replicates a pipelined eval loop to investigate two NaN phenomena:

Experiment A (queue depth race):
    CPU races ahead of GPU by 3-4 iterations when AQL queue is large (16K).
    Kernarg or tensor recycling causes silent data corruption -> NaN.
    Fixed by: ROC_AQL_QUEUE_SIZE=1024, GPU_MAX_HW_QUEUES=2, or any CPU-GPU sync.

Experiment B (large batch + pipelining NaN):
    At batch_size >= 1024, NaN persists even with AQL=1024 and full sync.
    Fixed by: disabling pipelining or reducing batch size.
    Hypotheses: torch.compile codegen bug, HIP cache coherence, allocator recycling.

Pipeline structure (pipelined, steady state):

    memcpy_stream:   ... H2D batch N was prefetched during iter N-1 ...
    datadist_stream: ... datadist for iter N was prefetched during iter N-1 ...

    default_stream:
        wait_stream(memcpy_stream)
        wait_stream(datadist_stream)
        compiled_forward(batch_N)       # torch.compile'd
        update_metrics(output_N)        # NE + MAE
        update_reg_metrics(output_N)    # calibration

    memcpy_stream:   H2D batch N+1 (prefetch)
    datadist_stream: datadist iter N+1 (prefetch)

    -- NO CPU-GPU sync, CPU races ahead --

Unpipelined mode (Experiment B control):
    Each iteration is fully independent: H2D -> sync -> datadist -> sync ->
    forward -> metrics -> sync. No cross-iteration buffer sharing.
"""

import logging
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile as torch_profile, ProfilerActivity, schedule

from ..base import BaseReproducer
from ..config import ReproducerConfig, ReproducerResult

log = logging.getLogger(__name__)


# =============================================================================
# Models
# =============================================================================


class EvalModel(nn.Module):
    """
    Simple MLP for the eval forward pass.

    Input: (batch_size, feature_dim) -> Output: (batch_size, 1).
    Applied with torch.compile when config.use_compile is True.
    """

    def __init__(self, feature_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(feature_dim, hidden_dim, bias=False))
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
        self.head = nn.Linear(hidden_dim, 1, bias=False)
        self.needs_sparse = False

    def forward(self, x: torch.Tensor, sparse_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = F.gelu(layer(x))
        return self.head(x)


class DLRMModel(nn.Module):
    """
    TorchRec-style DLRM model for realistic eval workload.

    Architecture:
        dense -> bottom_mlp -> embedding_dim
        sparse -> N embedding tables (lookup + sum pool) -> N x embedding_dim
        [dense_out, embed_0, ..., embed_N-1] -> concat -> over_arch -> (batch, 1)

    This generates GPU work matching production TorchRec eval: large embedding
    gathers (memory-BW bound) + MLP compute (FLOPs bound).
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_tables: int,
        embedding_rows: int,
        embedding_dim: int,
        sparse_pooling_factor: int,
        over_arch_layers: int,
    ):
        super().__init__()
        self.num_tables = num_tables
        self.pooling_factor = sparse_pooling_factor
        self.needs_sparse = True

        # Bottom MLP: feature_dim -> embedding_dim
        bottom = [nn.Linear(feature_dim, hidden_dim, bias=False)]
        for _ in range(num_layers - 1):
            bottom.extend([nn.ReLU(), nn.Linear(hidden_dim, hidden_dim, bias=False)])
        bottom.extend([nn.ReLU(), nn.Linear(hidden_dim, embedding_dim, bias=False)])
        self.bottom_mlp = nn.Sequential(*bottom)

        # Embedding tables
        self.embeddings = nn.ModuleList([
            nn.EmbeddingBag(embedding_rows, embedding_dim, mode="sum", sparse=False)
            for _ in range(num_tables)
        ])

        # Over-arch MLP: (num_tables + 1) * embedding_dim -> 1
        over_in = (num_tables + 1) * embedding_dim
        over = [nn.Linear(over_in, hidden_dim, bias=False)]
        for _ in range(over_arch_layers - 1):
            over.extend([nn.ReLU(), nn.Linear(hidden_dim, hidden_dim, bias=False)])
        over.extend([nn.ReLU(), nn.Linear(hidden_dim, 1, bias=False)])
        self.over_arch = nn.Sequential(*over)

        total_params = sum(p.numel() for p in self.parameters())
        emb_params = sum(p.numel() for emb in self.embeddings for p in emb.parameters())
        mlp_params = total_params - emb_params
        log.info(
            f"DLRMModel: {num_tables} tables x {embedding_rows} rows x {embedding_dim} dim, "
            f"params: {total_params/1e6:.1f}M total ({emb_params/1e6:.1f}M embed + {mlp_params/1e6:.1f}M MLP)"
        )

    def forward(self, dense: torch.Tensor, sparse_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dense: (batch, feature_dim) float32
            sparse_ids: (num_tables, batch * pooling_factor) int64
        """
        batch_size = dense.shape[0]

        # Bottom MLP
        dense_out = self.bottom_mlp(dense)

        # Embedding lookups with sum pooling
        offsets = torch.arange(
            0, batch_size * self.pooling_factor,
            self.pooling_factor, device=dense.device, dtype=torch.long,
        )
        embeds = []
        for i in range(self.num_tables):
            embeds.append(self.embeddings[i](sparse_ids[i], offsets))

        # Feature interaction (concat)
        interaction = torch.cat([dense_out] + embeds, dim=1)

        # Over-arch
        return self.over_arch(interaction)


class _HSTUAttentionLayer(nn.Module):
    """Single HSTU-style multi-head self-attention layer with pre-norm residual.

    Uses fused ``F.scaled_dot_product_attention`` -- no CPU-GPU sync, fully
    torch.compile-friendly.
    """

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        residual = x
        x = self.norm(x)
        qkv = self.qkv_proj(x).reshape(B, S, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(B, S, D)
        return residual + self.out_proj(out)


class HSTUModel(nn.Module):
    """
    DLRMv3-inspired model with HSTU-style causal attention on realistic-length
    sequences.

    Simulates the production dense forward pass: pre-computed sequence embeddings
    (from CPU sparse arch) are projected, run through multi-head causal attention
    layers, pooled, and fed to a prediction head that outputs raw logits.

    No GPU-side embedding tables -- sparse lookups happen on CPU and
    the results are transferred to GPU. We simulate this by accepting pre-computed
    seq_embeddings of shape (B, seq_len, feature_dim) as input.

    When datadist_proj is set, the model also consumes the datadist shard inside
    the compiled forward region, matching the production trace structure where the
    CompiledFullGraph block includes all post-wait work.

    GPU work is dominated by O(seq_len^2 * embed_dim) attention -- the only way
    to make the GPU heavy enough to fall behind the CPU for Experiment A.

    Key properties:
        - Zero CPU-GPU synchronization (no .item(), no Python loops over batch)
        - torch.compile compatible (standard PyTorch ops only)
        - Compute-bound via attention (not memory-BW bound like MLP/EmbeddingBag)
        - Output is raw logits for maximum corruption sensitivity
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        num_layers: int,
        embedding_dim: int,
        over_arch_layers: int,
        num_attn_layers: int = 5,
        num_attn_heads: int = 4,
    ):
        super().__init__()
        self.needs_sparse = False
        self.needs_seq_embeddings = True
        self.datadist_proj: Optional[nn.Linear] = None
        self._datadist_batch_size: int = 0

        # Bottom MLP: projects dense features -> embed_dim (injected into sequence)
        bottom = [nn.Linear(feature_dim, hidden_dim, bias=False)]
        for _ in range(num_layers - 1):
            bottom.extend([nn.ReLU(), nn.Linear(hidden_dim, hidden_dim, bias=False)])
        bottom.extend([nn.ReLU(), nn.Linear(hidden_dim, embedding_dim, bias=False)])
        self.bottom_mlp = nn.Sequential(*bottom)

        # Preprocessor: projects seq_embeddings from feature_dim -> embed_dim
        self.pre_attn_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, embedding_dim, bias=False),
            nn.LayerNorm(embedding_dim),
        )

        # HSTU attention stack
        self.attn_layers = nn.ModuleList([
            _HSTUAttentionLayer(embedding_dim, num_attn_heads)
            for _ in range(num_attn_layers)
        ])

        # Post-attention norm
        self.post_attn_norm = nn.LayerNorm(embedding_dim)

        # Prediction head: embed_dim -> 1, output is raw logit
        pred = [nn.Linear(embedding_dim, hidden_dim, bias=False)]
        for _ in range(over_arch_layers - 1):
            pred.extend([nn.ReLU(), nn.Linear(hidden_dim, hidden_dim, bias=False)])
        pred.extend([nn.ReLU(), nn.Linear(hidden_dim, 1, bias=False)])
        self.prediction_head = nn.Sequential(*pred)

        total_params = sum(p.numel() for p in self.parameters())
        log.info(
            f"HSTUModel: embed_dim={embedding_dim}, "
            f"{num_attn_layers} attn layers x {num_attn_heads} heads, "
            f"params: {total_params/1e6:.1f}M total"
        )

    def set_datadist_proj(self, proj: nn.Linear, batch_size: int) -> None:
        """Attach datadist projection so it runs inside the compiled forward."""
        self.datadist_proj = proj
        self._datadist_batch_size = batch_size

    def forward(self, seq_embeddings: torch.Tensor,
                dense_features: torch.Tensor,
                datadist_shard: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            seq_embeddings: (B, seq_len, feature_dim) from CPU sparse arch
            dense_features: (B, feature_dim) user/context features
            datadist_shard: optional (shard_size,) flat tensor from datadist stream
        Returns:
            (B, 1) raw logits
        """
        if datadist_shard is not None and self.datadist_proj is not None:
            shard_2d = datadist_shard.view(self._datadist_batch_size, -1)
            dense_features = dense_features + self.datadist_proj(shard_2d)

        seq = self.pre_attn_proj(seq_embeddings)

        dense_out = self.bottom_mlp(dense_features)
        seq[:, 0, :] = seq[:, 0, :] + dense_out

        for layer in self.attn_layers:
            seq = layer(seq)

        seq = self.post_attn_norm(seq)
        pooled = seq.mean(dim=1)
        return self.prediction_head(pooled)


# =============================================================================
# Reproducer
# =============================================================================


class EvalPipelinedReproducer(BaseReproducer):
    """
    Pipelined eval reproducer matching a production eval pipeline.

    Overrides the base run() to implement:
    - Zero inter-iteration sync (Experiment A) or configurable sync
    - torch.compile'd forward pass
    - NE / MAE / calibration metric accumulation
    - Double-buffered prefetch with datadist (reduce_scatter + send/recv)
    - DDP wrapper
    - NaN detection via accumulated metrics
    """

    def __init__(self, config: ReproducerConfig, rank: int, world_size: int):
        super().__init__(config, rank, world_size)

        self.model: Optional[nn.Module] = None
        self.compiled_model = None

        # Streams
        self.datadist_stream: Optional[torch.cuda.Stream] = None

        # Double-buffered pipeline tensors (dense)
        self.batch_gpu_current: Optional[torch.Tensor] = None
        self.batch_gpu_next: Optional[torch.Tensor] = None
        self.labels_gpu_current: Optional[torch.Tensor] = None
        self.labels_gpu_next: Optional[torch.Tensor] = None

        # Double-buffered sparse feature tensors
        self.sparse_gpu_current: Optional[torch.Tensor] = None
        self.sparse_gpu_next: Optional[torch.Tensor] = None

        # Double-buffered sequence embedding tensors (dlrm_v3)
        self.seq_emb_gpu_current: Optional[torch.Tensor] = None
        self.seq_emb_gpu_next: Optional[torch.Tensor] = None

        # Pre-generated CPU data
        self.all_batches_cpu: List[torch.Tensor] = []
        self.all_labels_cpu: List[torch.Tensor] = []
        self.all_sparse_cpu: List[torch.Tensor] = []
        self.all_seq_emb_cpu: List[torch.Tensor] = []

        # Datadist buffers (double-buffered)
        self.embed_full_current: Optional[torch.Tensor] = None
        self.embed_shard_current: Optional[torch.Tensor] = None
        self.embed_full_next: Optional[torch.Tensor] = None
        self.embed_shard_next: Optional[torch.Tensor] = None

        # Point-to-point buffers
        self.p2p_send_buf: Optional[torch.Tensor] = None
        self.p2p_recv_buf: Optional[torch.Tensor] = None

        # All-gather buffers (ReduceGather-like leg)
        self.gather_input: Optional[torch.Tensor] = None
        self.gather_output: Optional[torch.Tensor] = None

        # AllToAll buffers (CSAN shows alltoall_base_ as the racing collective)
        self.a2a_recv_buf: Optional[torch.Tensor] = None
        # Forward-phase AllToAll output (default stream, matches model_fwd
        # ShardedEmbeddingBagCollection AllToAll pattern from CSAN)
        self.fwd_a2a_output: Optional[torch.Tensor] = None

        # All-reduce buffer for embedding norm sync
        self.embed_norm_buf: Optional[torch.Tensor] = None

        # Metric accumulators (GPU tensors)
        self.ne_sum: Optional[torch.Tensor] = None
        self.mae_sum: Optional[torch.Tensor] = None
        self.cal_pred_sum: Optional[torch.Tensor] = None
        self.cal_label_sum: Optional[torch.Tensor] = None
        self.metric_count: int = 0

        # GPU padding buffer
        self._padding_buf: Optional[torch.Tensor] = None

        # CCA cross-stream alloc: cached shard sizes for dynamic allocation
        self._cca_shard_size: int = 0
        self._cca_full_size: int = 0

        # Datadist -> forward projection (creates real data dependency)
        self.datadist_proj: Optional[nn.Linear] = None

        # Whether model uses sparse features or seq_embeddings
        self._uses_sparse: bool = False
        self._uses_seq_embeddings: bool = False
        self._datadist_in_compiled: bool = False

        # CCA address tracking: maps data_ptr -> (alloc_iter, tensor_name, stream)
        self._cca_alloc_log: dict = {}
        # Reuse events: list of (iter, tensor_name, addr, prev_alloc_iter, prev_stream)
        self._cca_reuse_events: list = []
        self._cca_reuse_count: int = 0

        # Data integrity verification: GPU-side inline detection (no CPU-GPU sync).
        # Two detection phases per iteration:
        #   Phase A (pre-forward): compares datadist write-time checksum with
        #       default_stream read-time checksum.  Detects corruption that
        #       happened BEFORE the forward pass.
        #   Phase B (post-forward): compares pre-forward and post-forward
        #       checksums on default_stream.  Detects corruption that happened
        #       DURING the forward pass (the primary CCA race window).
        self._integrity_write_buf: Optional[torch.Tensor] = None
        self._integrity_pre_fwd_buf: Optional[torch.Tensor] = None
        self._integrity_corruption_gpu: Optional[torch.Tensor] = None
        self._integrity_mismatch_log_gpu: Optional[torch.Tensor] = None

    # =========================================================================
    # Setup
    # =========================================================================

    def setup(self) -> None:
        self._setup_env()
        self._setup_streams()
        self._setup_model()
        self._setup_pipeline_buffers()
        self._setup_metrics()
        if self.config.pre_generate_data:
            self._pre_generate_data()
        if self.config.gpu_padding_dispatches > 0:
            self._padding_buf = torch.empty(1024, device="cuda", dtype=torch.float32)

        if self.config.cca_cross_stream_alloc and self.world_size > 1:
            shard_size = self.config.embed_tensor_size // self.world_size
            shard_per_sample = shard_size // self.config.batch_size
            self._cca_shard_size = shard_per_sample * self.config.batch_size
            self._cca_full_size = self._cca_shard_size * self.world_size
            log.info(
                f"CCA cross-stream alloc enabled: shard_size={self._cca_shard_size}, "
                f"full_size={self._cca_full_size}, "
                f"record_stream={self.config.cca_record_stream}, "
                f"pressure_tensors={self.config.cca_num_pressure_tensors}"
            )

        if self.config.cca_integrity_check and self.world_size > 1:
            device = torch.device("cuda")
            self._integrity_write_buf = torch.zeros(1, dtype=torch.float64, device=device)
            self._integrity_pre_fwd_buf = torch.zeros(1, dtype=torch.float64, device=device)
            self._integrity_corruption_gpu = torch.zeros(1, dtype=torch.int64, device=device)
            max_log = 64
            self._integrity_mismatch_log_gpu = torch.full(
                (max_log,), -1, dtype=torch.int64, device=device
            )
            log.info(f"CCA integrity check enabled: inline GPU detection, {max_log} mismatch slots")

        model_info = f"model_type={self.config.model_type}"
        if self._uses_seq_embeddings:
            model_info += (
                f", seq_len={self.config.seq_len}"
                f", emb_dim={self.config.embedding_dim}"
                f", hstu_layers={self.config.hstu_attn_num_layers}"
                f", hstu_heads={self.config.hstu_num_heads}"
                f", bfloat16={self.config.use_bfloat16}"
            )
        elif self._uses_sparse:
            model_info += (
                f", tables={self.config.num_embedding_tables}"
                f", rows={self.config.embedding_rows}"
                f", emb_dim={self.config.embedding_dim}"
                f", pool={self.config.sparse_pooling_factor}"
            )
        log.info(
            f"EvalPipelined setup: rank={self.rank}, world_size={self.world_size}, "
            f"batch_size={self.config.batch_size}, hidden_dim={self.config.hidden_dim}, "
            f"model_layers={self.config.model_layers}, compile={self.config.use_compile}, "
            f"pipelining={self.config.enable_pipelining}, "
            f"sync_policy={self.config.sync_policy}, {model_info}"
        )

    def _setup_streams(self) -> None:
        self.default_stream = torch.cuda.current_stream()
        self.memcpy_stream = torch.cuda.Stream()
        if self.world_size > 1 and self.config.use_datadist_stream:
            if self.config.same_stream_mode:
                self.datadist_stream = self.memcpy_stream
                log.info("Using SAME stream for H2D and datadist")
            else:
                self.datadist_stream = torch.cuda.Stream()
                log.info("Created separate datadist_stream")
        else:
            self.datadist_stream = None
        log.info("Created memcpy_stream and default_stream")

    def _setup_model(self) -> None:
        cfg = self.config
        device = torch.device("cuda")

        if cfg.model_type == "dlrm_v3":
            self.model = HSTUModel(
                feature_dim=cfg.feature_dim,
                hidden_dim=cfg.hidden_dim,
                num_layers=cfg.model_layers,
                embedding_dim=cfg.embedding_dim,
                over_arch_layers=cfg.over_arch_layers,
                num_attn_layers=cfg.hstu_attn_num_layers,
                num_attn_heads=cfg.hstu_num_heads,
            ).to(device=device, dtype=torch.float32)
        elif cfg.model_type == "dlrm":
            self.model = DLRMModel(
                feature_dim=cfg.feature_dim,
                hidden_dim=cfg.hidden_dim,
                num_layers=cfg.model_layers,
                num_tables=cfg.num_embedding_tables,
                embedding_rows=cfg.embedding_rows,
                embedding_dim=cfg.embedding_dim,
                sparse_pooling_factor=cfg.sparse_pooling_factor,
                over_arch_layers=cfg.over_arch_layers,
            ).to(device=device, dtype=torch.float32)
        else:
            self.model = EvalModel(
                cfg.feature_dim, cfg.hidden_dim, cfg.model_layers
            ).to(device=device, dtype=torch.float32)

        self._uses_sparse = getattr(self.model, "needs_sparse", False)
        self._uses_seq_embeddings = getattr(self.model, "needs_seq_embeddings", False)
        self._datadist_in_compiled = False
        self.model.eval()

        # For dlrm_v3 with multi-GPU: attach datadist projection to the model
        # BEFORE DDP/compile so it becomes part of the compiled graph.  This
        # matches the production trace where CompiledFullGraph includes all
        # post-wait work (datadist consumption + attention + metrics preamble).
        if (cfg.model_type == "dlrm_v3" and self.world_size > 1
                and hasattr(self.model, "set_datadist_proj")):
            shard_size = cfg.embed_tensor_size // self.world_size
            shard_per_sample = shard_size // cfg.batch_size
            shard_size = shard_per_sample * cfg.batch_size
            proj = nn.Linear(
                shard_per_sample, cfg.feature_dim, bias=False,
            ).to(device=device, dtype=torch.float32)
            proj.eval()
            for p in proj.parameters():
                p.requires_grad_(False)
            self.model.set_datadist_proj(proj, cfg.batch_size)
            self._datadist_in_compiled = True
            log.info(
                f"Datadist projection attached to model (inside compiled forward): "
                f"shard ({shard_size},) -> ({cfg.batch_size}, {shard_per_sample}) "
                f"-> ({cfg.batch_size}, {cfg.feature_dim})"
            )

        if self.world_size > 1 and cfg.use_ddp_wrapper:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=[local_rank]
            )
            log.info("Wrapped model in DDP")

        if cfg.use_compile:
            self.compiled_model = torch.compile(self.model)
            log.info("Applied torch.compile to model")
        else:
            self.compiled_model = self.model
            log.info("torch.compile disabled")

    def _setup_pipeline_buffers(self) -> None:
        cfg = self.config
        device = torch.device("cuda")

        self.batch_gpu_current = torch.empty(
            cfg.batch_size, cfg.feature_dim, dtype=torch.float32, device=device
        )
        self.labels_gpu_current = torch.empty(
            cfg.batch_size, 1, dtype=torch.float32, device=device
        )

        if self._uses_sparse:
            sparse_shape = (cfg.num_embedding_tables, cfg.batch_size * cfg.sparse_pooling_factor)
            self.sparse_gpu_current = torch.zeros(sparse_shape, dtype=torch.long, device=device)

        if self._uses_seq_embeddings:
            seq_shape = (cfg.batch_size, cfg.seq_len, cfg.feature_dim)
            self.seq_emb_gpu_current = torch.empty(
                *seq_shape, dtype=torch.float32, device=device
            )

        if cfg.enable_pipelining and not cfg.cca_cross_stream_alloc:
            self.batch_gpu_next = torch.empty(
                cfg.batch_size, cfg.feature_dim, dtype=torch.float32, device=device
            )
            self.labels_gpu_next = torch.empty(
                cfg.batch_size, 1, dtype=torch.float32, device=device
            )
            if self._uses_sparse:
                self.sparse_gpu_next = torch.zeros(sparse_shape, dtype=torch.long, device=device)
            if self._uses_seq_embeddings:
                self.seq_emb_gpu_next = torch.empty(
                    *seq_shape, dtype=torch.float32, device=device
                )

        if self.world_size > 1:
            shard_size = cfg.embed_tensor_size // self.world_size
            shard_per_sample = shard_size // cfg.batch_size
            shard_size = shard_per_sample * cfg.batch_size
            full_size = shard_size * self.world_size
            self.embed_full_current = torch.randn(
                full_size, dtype=torch.float32, device=device
            )
            self.embed_shard_current = torch.empty(
                shard_size, dtype=torch.float32, device=device
            )
            self.p2p_send_buf = torch.randn(
                cfg.p2p_tensor_size, dtype=torch.float32, device=device
            )
            self.p2p_recv_buf = torch.empty(
                cfg.p2p_tensor_size, dtype=torch.float32, device=device
            )
            # All-gather buffers: simulates the ReduceGather-like leg visible in
            # production traces.  Uses shard_size per rank, gathered into full_size.
            self.gather_input = torch.randn(
                shard_size, dtype=torch.float32, device=device
            )
            self.gather_output = torch.empty(
                full_size, dtype=torch.float32, device=device
            )
            # All-reduce buffer: simulates embedding norm sync visible in production traces
            self.embed_norm_buf = torch.randn(
                shard_size, dtype=torch.float32, device=device
            )
            # AllToAll recv buffer for datadist leg (matches CSAN collective type)
            self.a2a_recv_buf = torch.empty(
                full_size, dtype=torch.float32, device=device
            )
            # Default-stream AllToAll output: simulates model's embedding
            # redistribution (ShardedEmbeddingBagCollection AllToAll on
            # default stream during model_fwd, per CSAN trace).
            if shard_size % self.world_size == 0:
                self.fwd_a2a_output = torch.empty(
                    shard_size, dtype=torch.float32, device=device
                )
            else:
                log.warning(
                    f"fwd_a2a_output NOT allocated: shard_size={shard_size} is not "
                    f"divisible by world_size={self.world_size}. "
                    f"_model_fwd_alltoall() will be a no-op -- the default-stream "
                    f"AllToAll pattern from CSAN will NOT be reproduced. "
                    f"Adjust embed_tensor_size or batch_size so that "
                    f"(embed_tensor_size // world_size) %% world_size == 0."
                )
            if cfg.enable_pipelining and not cfg.cca_cross_stream_alloc:
                self.embed_full_next = torch.randn(
                    full_size, dtype=torch.float32, device=device
                )
                self.embed_shard_next = torch.empty(
                    shard_size, dtype=torch.float32, device=device
                )

            if not self._datadist_in_compiled:
                self.datadist_proj = nn.Linear(
                    shard_per_sample, cfg.feature_dim, bias=False,
                ).to(device=device, dtype=torch.float32)
                self.datadist_proj.eval()
                for p in self.datadist_proj.parameters():
                    p.requires_grad_(False)
                log.info(
                    f"Datadist->forward projection (eager): shard ({shard_size},) -> "
                    f"({cfg.batch_size}, {shard_per_sample}) -> "
                    f"({cfg.batch_size}, {cfg.feature_dim})"
                )

        seq_info = ""
        if self._uses_seq_embeddings:
            seq_bytes = cfg.batch_size * cfg.seq_len * cfg.feature_dim * 4
            bufs = 2 if cfg.enable_pipelining else 1
            seq_info = f", seq_emb={bufs}x{seq_bytes/1e6:.1f}MB"
        log.info(
            f"Pipeline buffers allocated: pipelining={cfg.enable_pipelining}, "
            f"model_type={cfg.model_type}{seq_info}, "
            f"datadist={'enabled' if self.world_size > 1 else 'disabled (single GPU)'}"
        )

    def _setup_metrics(self) -> None:
        device = torch.device("cuda")
        self.ne_sum = torch.zeros(1, dtype=torch.float32, device=device)
        self.mae_sum = torch.zeros(1, dtype=torch.float32, device=device)
        self.cal_pred_sum = torch.zeros(1, dtype=torch.float32, device=device)
        self.cal_label_sum = torch.zeros(1, dtype=torch.float32, device=device)
        self.metric_count = 0

    def _pre_generate_data(self) -> None:
        cfg = self.config
        total_iters = cfg.warmup_iterations + cfg.verify_iterations + 2

        if cfg.pre_generate_pool_size is not None:
            pool_size = cfg.pre_generate_pool_size
        elif self._uses_seq_embeddings:
            pool_size = 20
        else:
            pool_size = total_iters

        total = min(pool_size, total_iters)
        log.info(
            f"Pre-generating {total} CPU batches "
            f"(model_type={cfg.model_type}, pool for {total_iters} iters)..."
        )
        gen = torch.Generator()
        gen.manual_seed(42 + self.rank)

        for _ in range(total):
            self.all_batches_cpu.append(
                torch.randn(
                    cfg.batch_size, cfg.feature_dim,
                    dtype=torch.float32, generator=gen,
                ).pin_memory()
            )
            self.all_labels_cpu.append(
                torch.randint(
                    0, 2, (cfg.batch_size, 1),
                    dtype=torch.float32, generator=gen,
                ).pin_memory()
            )
            if self._uses_sparse:
                self.all_sparse_cpu.append(
                    torch.randint(
                        0, cfg.embedding_rows,
                        (cfg.num_embedding_tables, cfg.batch_size * cfg.sparse_pooling_factor),
                        dtype=torch.long,
                    ).pin_memory()
                )
            if self._uses_seq_embeddings:
                self.all_seq_emb_cpu.append(
                    torch.randn(
                        cfg.batch_size, cfg.seq_len, cfg.feature_dim,
                        dtype=torch.float32, generator=gen,
                    ).pin_memory()
                )

        if self._uses_seq_embeddings:
            seq_bytes = cfg.batch_size * cfg.seq_len * cfg.feature_dim * 4
            log.info(
                f"Pre-generated {total} CPU batches "
                f"(seq_emb: {total}x{seq_bytes/1e6:.1f}MB = "
                f"{total * seq_bytes / 1e6:.0f}MB pinned)"
            )
        else:
            log.info(f"Pre-generated {total} CPU batches")

    # =========================================================================
    # Pipeline primitives
    # =========================================================================

    def _get_cpu_batch(
        self, iteration: int
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Returns (dense_batch, labels, sparse_ids_or_None, seq_emb_or_None)."""
        if self.config.pre_generate_data:
            idx = iteration % len(self.all_batches_cpu)
            sparse = self.all_sparse_cpu[idx] if self._uses_sparse else None
            seq_emb = self.all_seq_emb_cpu[idx] if self._uses_seq_embeddings else None
            return self.all_batches_cpu[idx], self.all_labels_cpu[idx], sparse, seq_emb
        cfg = self.config
        batch = torch.randn(cfg.batch_size, cfg.feature_dim, dtype=torch.float32)
        labels = torch.randint(0, 2, (cfg.batch_size, 1), dtype=torch.float32)
        sparse = None
        seq_emb = None
        if self._uses_sparse:
            sparse = torch.randint(
                0, cfg.embedding_rows,
                (cfg.num_embedding_tables, cfg.batch_size * cfg.sparse_pooling_factor),
                dtype=torch.long,
            )
        if self._uses_seq_embeddings:
            seq_emb = torch.randn(
                cfg.batch_size, cfg.seq_len, cfg.feature_dim, dtype=torch.float32,
            )
        return batch, labels, sparse, seq_emb

    def _prefetch_h2d(self, iteration: int) -> None:
        batch_cpu, labels_cpu, sparse_cpu, seq_emb_cpu = self._get_cpu_batch(iteration)
        cfg = self.config
        pipelined = cfg.enable_pipelining
        cca = cfg.cca_cross_stream_alloc and pipelined

        with torch.cuda.stream(self.memcpy_stream):
            if cca:
                target_batch = torch.empty(
                    cfg.batch_size, cfg.feature_dim,
                    dtype=torch.float32, device="cuda",
                )
                target_labels = torch.empty(
                    cfg.batch_size, 1,
                    dtype=torch.float32, device="cuda",
                )
            else:
                target_batch = self.batch_gpu_next if pipelined else self.batch_gpu_current
                target_labels = self.labels_gpu_next if pipelined else self.labels_gpu_current

            target_batch.copy_(batch_cpu, non_blocking=True)
            target_labels.copy_(labels_cpu, non_blocking=True)

            if sparse_cpu is not None:
                target_sparse = self.sparse_gpu_next if pipelined else self.sparse_gpu_current
                if cca:
                    target_sparse = torch.zeros(
                        cfg.num_embedding_tables,
                        cfg.batch_size * cfg.sparse_pooling_factor,
                        dtype=torch.long, device="cuda",
                    )
                target_sparse.copy_(sparse_cpu, non_blocking=True)

            if seq_emb_cpu is not None:
                if cca:
                    target_seq = torch.empty(
                        cfg.batch_size, cfg.seq_len, cfg.feature_dim,
                        dtype=torch.float32, device="cuda",
                    )
                else:
                    target_seq = self.seq_emb_gpu_next if pipelined else self.seq_emb_gpu_current
                target_seq.copy_(seq_emb_cpu, non_blocking=True)

        if cca:
            self._cca_track_alloc(target_batch, "batch", "memcpy_stream", iteration)
            self._cca_track_alloc(target_labels, "labels", "memcpy_stream", iteration)
            self.batch_gpu_next = target_batch
            self.labels_gpu_next = target_labels
            if sparse_cpu is not None:
                self._cca_track_alloc(target_sparse, "sparse", "memcpy_stream", iteration)
                self.sparse_gpu_next = target_sparse
            if seq_emb_cpu is not None:
                self._cca_track_alloc(target_seq, "seq_emb", "memcpy_stream", iteration)
                self.seq_emb_gpu_next = target_seq

    def _prefetch_datadist(self, iteration: int = -1) -> None:
        if self.world_size <= 1:
            return
        cfg = self.config
        cca = cfg.cca_cross_stream_alloc and cfg.enable_pipelining

        # When datadist_stream is None (use_datadist_stream=False), run
        # datadist work on the default stream (serialized) rather than
        # skipping it -- embed_shard_current must be populated before
        # _inject_datadist / _forward reads it.
        stream_ctx = (
            torch.cuda.stream(self.datadist_stream)
            if self.datadist_stream is not None
            else nullcontext()
        )
        with stream_ctx:
            if cca:
                if self.config.cca_event_sync:
                    # Mitigation: wait for default_stream's forward to
                    # complete before dropping references.  If corruption
                    # disappears with this flag, it confirms the CCA
                    # recycling race is the root cause.
                    ev = torch.cuda.Event()
                    ev.record(self.default_stream)
                    ev.synchronize()

                # Drop old embed refs BEFORE allocating new ones.
                # With --no-cca-record-stream, stream_uses is empty, so
                # CCA returns these blocks to the free pool immediately
                # (no events inserted -- see CUDACachingAllocator free()).
                # The subsequent torch.empty() on the same stream can then
                # reuse these blocks while default_stream's forward is
                # still reading them.
                self.embed_full_current = None
                self.embed_shard_current = None

                target_full = torch.randn(
                    self._cca_full_size, dtype=torch.float32, device="cuda",
                )
                target_shard = torch.empty(
                    self._cca_shard_size, dtype=torch.float32, device="cuda",
                )
            else:
                target_full = self.embed_full_next if cfg.enable_pipelining else self.embed_full_current
                target_shard = self.embed_shard_next if cfg.enable_pipelining else self.embed_shard_current

            # Leg 1: reduce_scatter (embedding redistribution)
            dist.reduce_scatter_tensor(target_shard, target_full)
            # Post-scatter normalization (real GPU work on datadist stream)
            target_shard.mul_(1.0 / self.world_size)

            # Integrity check: record checksum on datadist_stream right after write
            if self.config.cca_integrity_check:
                self._integrity_write_checksum(target_shard, iteration)

            # Leg 2: point-to-point send/recv (batched to avoid separate
            # communicator creation per op which hangs on some RCCL versions)
            if self.world_size > 1:
                send_peer = (self.rank + 1) % self.world_size
                recv_peer = (self.rank - 1 + self.world_size) % self.world_size
                p2p_ops = [
                    dist.P2POp(dist.isend, self.p2p_send_buf, send_peer),
                    dist.P2POp(dist.irecv, self.p2p_recv_buf, recv_peer),
                ]
                reqs = dist.batch_isend_irecv(p2p_ops)
                # No .wait() -- GPU stream ordering on datadist_stream
                # guarantees the recv completes before subsequent ops.
                # Matches the production pipeline which has zero CPU-GPU sync.
                self.p2p_recv_buf.clamp_(-10.0, 10.0)

            # Leg 3: all_gather (ReduceGather-like collection)
            if self.gather_input is not None:
                dist.all_gather_into_tensor(
                    self.gather_output, self.gather_input
                )

            # Leg 4: all_reduce for embedding norm sync
            if self.embed_norm_buf is not None:
                dist.all_reduce(self.embed_norm_buf)

            # Leg 5: all_to_all_single (CSAN shows alltoall_base_ as the
            # racing collective in KJTAllToAllTensorsAwaitable)
            if self.a2a_recv_buf is not None:
                if cca:
                    a2a_recv = torch.empty(
                        self._cca_full_size, dtype=torch.float32, device="cuda",
                    )
                else:
                    a2a_recv = self.a2a_recv_buf
                dist.all_to_all_single(a2a_recv, target_full)

        if cca:
            self._cca_track_alloc(target_full, "embed_full", "datadist_stream", iteration)
            self._cca_track_alloc(target_shard, "embed_shard", "datadist_stream", iteration)
            self.embed_full_next = target_full
            self.embed_shard_next = target_shard

    def _swap_buffers(self) -> None:
        if not self.config.enable_pipelining:
            return

        if self.config.cca_cross_stream_alloc:
            # embed_full/embed_shard_current were already dropped inside
            # _prefetch_datadist, BEFORE the next torch.empty() call.
            # With --no-cca-record-stream (stream_uses empty), CCA
            # returns freed blocks to the pool immediately, so the
            # subsequent allocation can reuse them while default_stream
            # is still reading.  The remaining tensors (batch, labels,
            # sparse) are dropped here.

            self.batch_gpu_current = self.batch_gpu_next
            self.labels_gpu_current = self.labels_gpu_next
            self.batch_gpu_next = None
            self.labels_gpu_next = None

            if self._uses_sparse:
                self.sparse_gpu_current = self.sparse_gpu_next
                self.sparse_gpu_next = None

            if self._uses_seq_embeddings and self.seq_emb_gpu_next is not None:
                self.seq_emb_gpu_current = self.seq_emb_gpu_next
                self.seq_emb_gpu_next = None

            if self.world_size > 1 and self.embed_shard_next is not None:
                self.embed_full_current = self.embed_full_next
                self.embed_shard_current = self.embed_shard_next
                self.embed_full_next = None
                self.embed_shard_next = None
        else:
            self.batch_gpu_current, self.batch_gpu_next = (
                self.batch_gpu_next, self.batch_gpu_current
            )
            self.labels_gpu_current, self.labels_gpu_next = (
                self.labels_gpu_next, self.labels_gpu_current
            )
            if self._uses_sparse and self.sparse_gpu_next is not None:
                self.sparse_gpu_current, self.sparse_gpu_next = (
                    self.sparse_gpu_next, self.sparse_gpu_current
                )
            if self._uses_seq_embeddings and self.seq_emb_gpu_next is not None:
                self.seq_emb_gpu_current, self.seq_emb_gpu_next = (
                    self.seq_emb_gpu_next, self.seq_emb_gpu_current
                )
            if self.world_size > 1 and self.embed_full_next is not None:
                self.embed_full_current, self.embed_full_next = (
                    self.embed_full_next, self.embed_full_current
                )
                self.embed_shard_current, self.embed_shard_next = (
                    self.embed_shard_next, self.embed_shard_current
                )

    def _update_metrics(self, output: torch.Tensor, labels: torch.Tensor) -> None:
        if not self.config.simulate_metrics:
            return
        self.ne_sum += F.binary_cross_entropy_with_logits(
            output, labels, reduction="sum"
        )
        self.mae_sum += (output.sigmoid() - labels).abs().sum()
        self.cal_pred_sum += output.sigmoid().sum()
        self.cal_label_sum += labels.sum()
        self.metric_count += self.config.batch_size

    def _inject_datadist(self, batch: torch.Tensor) -> None:
        """Add datadist output into dense features, creating a real data dependency.

        After this call, forward reads data that was produced by the datadist
        stream.  Any RCCL signaling bug, stale cache line, or premature stream
        completion will directly poison the forward input and surface as NaN.
        """
        if self.datadist_proj is None or self.world_size <= 1:
            return
        shard = self.embed_shard_current.view(self.config.batch_size, -1)
        batch.add_(self.datadist_proj(shard))

    def _gpu_padding(self) -> None:
        if self.config.gpu_padding_dispatches <= 0:
            return
        for _ in range(self.config.gpu_padding_dispatches):
            self._padding_buf.fill_(1.0)

    def _cca_record_streams(self) -> None:
        """Tell CCA that the default stream also uses cross-stream tensors.

        Without this, block->stream_uses is empty.  On free, CCA returns
        the block to the pool immediately (no events inserted at all).
        The next torch.empty() on the allocation stream can reuse the
        block while the default stream's forward is still reading it.

        Calling record_stream(default_stream) populates stream_uses.
        On free, CCA inserts events on those extra streams via
        insert_events().  The block is only recycled once all events
        complete -- ensuring the default stream has finished reading.
        """
        if not self.config.cca_record_stream:
            return
        ds = self.default_stream
        if self.batch_gpu_current is not None:
            self.batch_gpu_current.record_stream(ds)
        if self.labels_gpu_current is not None:
            self.labels_gpu_current.record_stream(ds)
        if self._uses_seq_embeddings and self.seq_emb_gpu_current is not None:
            self.seq_emb_gpu_current.record_stream(ds)
        if self._uses_sparse and self.sparse_gpu_current is not None:
            self.sparse_gpu_current.record_stream(ds)
        if self.embed_shard_current is not None:
            self.embed_shard_current.record_stream(ds)
        if self.embed_full_current is not None:
            self.embed_full_current.record_stream(ds)

    def _cca_track_alloc(self, tensor: torch.Tensor, name: str, stream: str, iteration: int) -> None:
        """Track a CCA allocation and detect address reuse."""
        addr = tensor.data_ptr()
        prev = self._cca_alloc_log.get(addr)
        if prev is not None:
            prev_iter, prev_name, prev_stream = prev
            self._cca_reuse_count += 1
            # Log cross-stream reuses (the dangerous ones) and first few same-stream
            is_cross_stream = stream != prev_stream
            if self._cca_reuse_count <= 5 or (is_cross_stream and len(self._cca_reuse_events) < 20):
                self._cca_reuse_events.append(
                    (iteration, name, addr, prev_iter, prev_name, prev_stream)
                )
                tag = "CROSS-STREAM" if is_cross_stream else "same-stream"
                log.info(
                    f"[rank{self.rank}] CCA REUSE ({tag}): {name} @{addr:#x} on {stream} "
                    f"(iter {iteration}) <- was {prev_name} on {prev_stream} (iter {prev_iter})"
                )
        self._cca_alloc_log[addr] = (iteration, name, stream)

    def _cca_log_summary(self) -> None:
        """Log summary of CCA address reuse."""
        log.info(
            f"[rank{self.rank}] CCA address reuse summary: "
            f"{self._cca_reuse_count} reuses across {len(self._cca_alloc_log)} unique addresses"
        )

    def _integrity_write_checksum(self, tensor: torch.Tensor, iteration: int) -> None:
        """Compute checksum on datadist_stream right after reduce_scatter.

        Stores a single scalar checksum in ``_integrity_write_buf`` (GPU-side).
        The default_stream will compare against this value next iteration
        (after wait_stream ensures visibility).
        """
        if self._integrity_write_buf is None:
            return
        self._integrity_write_buf[0] = tensor.to(torch.float64).sum()

    def _integrity_compare(
        self, tensor: torch.Tensor, ref_buf: torch.Tensor, iteration: int,
    ) -> None:
        """Checksum tensor on current stream and compare with ref_buf (GPU-only)."""
        cur_sum = tensor.to(torch.float64).sum()
        mismatch = ((cur_sum - ref_buf[0]).abs() > 1e-6).to(torch.int64)

        log_buf = self._integrity_mismatch_log_gpu
        assert log_buf is not None
        count_before = self._integrity_corruption_gpu[0].clone()
        log_idx = count_before % log_buf.shape[0]
        iter_t = torch.tensor(iteration, dtype=torch.int64, device=tensor.device)
        log_buf[log_idx] = torch.where(mismatch.bool(), iter_t, log_buf[log_idx])
        self._integrity_corruption_gpu += mismatch

    def _integrity_pre_forward(self, tensor: torch.Tensor, iteration: int) -> None:
        """Phase A: checksum on default_stream before forward.

        Compares with write-time checksum from datadist_stream. Detects
        corruption that happened between datadist write and forward start.
        Also captures the value in pre_fwd_buf for phase B comparison.
        """
        if self._integrity_write_buf is None:
            return
        cur_sum = tensor.to(torch.float64).sum()
        self._integrity_pre_fwd_buf[0] = cur_sum

        mismatch = ((cur_sum - self._integrity_write_buf[0]).abs() > 1e-6).to(torch.int64)
        log_buf = self._integrity_mismatch_log_gpu
        assert log_buf is not None
        count_before = self._integrity_corruption_gpu[0].clone()
        log_idx = count_before % log_buf.shape[0]
        iter_t = torch.tensor(iteration, dtype=torch.int64, device=tensor.device)
        log_buf[log_idx] = torch.where(mismatch.bool(), iter_t, log_buf[log_idx])
        self._integrity_corruption_gpu += mismatch

    def _integrity_post_forward(self, tensor: torch.Tensor, iteration: int) -> None:
        """Phase B: checksum on default_stream after forward.

        Compares with pre-forward checksum. Detects corruption that happened
        DURING the forward pass -- the primary CCA race window where the
        datadist_stream recycles embed_shard memory while default_stream reads.
        """
        if self._integrity_pre_fwd_buf is None:
            return
        self._integrity_compare(tensor, self._integrity_pre_fwd_buf, iteration)

    def _integrity_read_corruption_count(self) -> int:
        """Non-blocking-ish read of the corruption counter (one scalar D2H)."""
        if self._integrity_corruption_gpu is None:
            return 0
        return int(self._integrity_corruption_gpu.item())

    def _integrity_report(self) -> int:
        """Final report after sync.  Returns corruption count."""
        if self._integrity_corruption_gpu is None:
            return 0
        count = int(self._integrity_corruption_gpu.item())
        if count > 0:
            logged = self._integrity_mismatch_log_gpu
            assert logged is not None
            n_logged = min(count, logged.shape[0])
            iters = logged[:n_logged].tolist()
            log.error(
                f"[rank{self.rank}] CCA INTEGRITY VIOLATION: {count} mismatches detected. "
                f"First {n_logged} at iterations: {iters}"
            )
            log.error(
                f"[rank{self.rank}] CCA recycled embed_shard block (stream_uses empty, "
                f"no events) while default_stream forward was still reading it."
            )
        else:
            log.info(
                f"[rank{self.rank}] CCA integrity check PASSED: "
                f"all checksums matched (no memory recycling detected)"
            )
        return count

    def _cca_pressure(self, iteration: int) -> None:
        """Stress CCA's cross-stream recycling with two pressure patterns.

        Pattern 1 (default-stream): allocate and free shard-sized tensors on
        the default stream.  Depletes the CCA free pool, forcing aggressive
        recycling when datadist_stream calls torch.empty().

        Pattern 2 (cross-stream): allocate tensors on datadist_stream, read
        them on default_stream (via wait_stream), then drop refs.  Without
        record_stream(), CCA records events on datadist_stream only.  This
        mirrors the exact race path of the pipeline buffers: datadist alloc
        -> default read -> CCA recycles to datadist while default still reads.
        More cross-stream pressure = higher recycling rate = wider race window.

        Half the budget goes to each pattern.
        """
        if self.config.cca_num_pressure_tensors <= 0 or self._cca_shard_size <= 0:
            return

        n_default = self.config.cca_num_pressure_tensors // 2
        n_cross = self.config.cca_num_pressure_tensors - n_default

        for pi in range(n_default):
            t = torch.empty(self._cca_shard_size, dtype=torch.float32, device="cuda")
            self._cca_track_alloc(t, f"pressure_def_{pi}", "default_stream", iteration)
            t.fill_(float(iteration % 100))

        if self.datadist_stream is not None and n_cross > 0:
            cross_tensors = []
            with torch.cuda.stream(self.datadist_stream):
                for pi in range(n_cross):
                    t = torch.empty(self._cca_shard_size, dtype=torch.float32, device="cuda")
                    t.fill_(float(iteration % 100 + pi))
                    self._cca_track_alloc(t, f"pressure_cross_{pi}", "datadist_stream", iteration)
                    cross_tensors.append(t)

            self.default_stream.wait_stream(self.datadist_stream)
            for t in cross_tensors:
                _ = t.sum()
                if self.config.cca_record_stream:
                    t.record_stream(self.default_stream)

    def _model_fwd_alltoall(self) -> None:
        """Run AllToAll on default stream, simulating the model's internal
        embedding redistribution (ShardedEmbeddingBagCollection AllToAll).

        Uses embed_shard_current as input -- allocated on datadist_stream,
        read here on default_stream.  Without record_stream, the block's
        stream_uses is empty, so CCA returns it to the pool immediately
        on free.  The next allocation on datadist_stream can reuse it
        while default_stream is still reading.

        We use sync (async_op=False) because async_op=True returns a Work
        object that stashes tensor refs, keeping Python refcount > 0 and
        preventing CCA from freeing the block.
        """
        if self.fwd_a2a_output is None or self.embed_shard_current is None:
            return
        if self.world_size <= 1:
            return
        dist.all_to_all_single(self.fwd_a2a_output, self.embed_shard_current)

    # =========================================================================
    # Dispatch measurement
    # =========================================================================

    def _measure_dispatches_per_stream(
        self,
        iter_fn,
        start_iter: int,
        num_iters: int = 3,
    ) -> dict:
        """Profile a few iterations and count GPU dispatches per stream.

        Returns dict mapping stream_id -> dispatches_per_iter (averaged).
        All ranks participate (collectives require it), but only rank 0
        inspects the trace.
        """
        import json as _json
        import tempfile

        torch.cuda.synchronize()

        tmp = None
        if self.rank == 0:
            tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
            tmp.close()

        with torch_profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,
            with_stack=False,
        ) as prof:
            for ci in range(num_iters):
                iter_fn(start_iter + ci)
                torch.cuda.synchronize()

        if self.rank != 0:
            return {}

        prof.export_chrome_trace(tmp.name)
        try:
            with open(tmp.name) as f:
                trace = _json.load(f)
        finally:
            os.unlink(tmp.name)

        events = trace.get("traceEvents", trace) if isinstance(trace, dict) else trace

        stream_counts: dict = {}
        for ev in events:
            cat = ev.get("cat", "")
            if cat not in ("kernel", "gpu_memcpy"):
                continue
            sid = str(ev.get("args", {}).get("stream", "?"))
            stream_counts[sid] = stream_counts.get(sid, 0) + 1

        per_iter: dict = {}
        for sid, count in stream_counts.items():
            per_iter[sid] = round(count / num_iters)

        return per_iter

    # =========================================================================
    # NaN detection
    # =========================================================================

    def _check_nan(self) -> bool:
        """Return True if any accumulated metric contains NaN or Inf."""
        for name, tensor in [
            ("ne_sum", self.ne_sum),
            ("mae_sum", self.mae_sum),
            ("cal_pred_sum", self.cal_pred_sum),
            ("cal_label_sum", self.cal_label_sum),
        ]:
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                return True
        return False

    # =========================================================================
    # Iteration implementations
    # =========================================================================

    def _forward(
        self,
        batch: torch.Tensor,
        sparse: Optional[torch.Tensor] = None,
        seq_emb: Optional[torch.Tensor] = None,
        datadist_shard: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run forward pass, dispatching to the right model input format."""
        if self._uses_seq_embeddings and seq_emb is not None:
            if datadist_shard is not None:
                return self.compiled_model(seq_emb, batch, datadist_shard)
            return self.compiled_model(seq_emb, batch)
        if self._uses_sparse and sparse is not None:
            return self.compiled_model(batch, sparse)
        return self.compiled_model(batch)

    def _run_iteration_pipelined(self, iteration: int) -> None:
        """Steady-state pipelined iteration (Layer 1)."""
        self.default_stream.wait_stream(self.memcpy_stream)
        if self.datadist_stream is not None:
            self.default_stream.wait_stream(self.datadist_stream)

        # Phase A: pre-forward integrity check on default_stream.
        # Compares write-time checksum (datadist_stream, iter N-1) with
        # current data. Detects corruption before the forward starts.
        if (self.config.cca_integrity_check and self.embed_shard_current is not None
                and iteration > 0):
            self._integrity_pre_forward(self.embed_shard_current, iteration - 1)

        if self.config.cca_cross_stream_alloc:
            self._cca_record_streams()

        if not self._datadist_in_compiled:
            self._inject_datadist(self.batch_gpu_current)

        autocast_ctx = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if self.config.use_bfloat16 else nullcontext()
        )
        shard = self.embed_shard_current if self._datadist_in_compiled else None
        with torch.no_grad(), autocast_ctx:
            output = self._forward(
                self.batch_gpu_current, self.sparse_gpu_current,
                self.seq_emb_gpu_current, shard,
            )
            labels = self.labels_gpu_current

        self._model_fwd_alltoall()

        # Phase B: post-forward integrity check on default_stream.
        # Compares with pre-forward checksum. Detects corruption DURING
        # the forward -- the primary CCA race window where datadist_stream
        # recycles embed_shard memory while default_stream reads it.
        if (self.config.cca_integrity_check and self.embed_shard_current is not None
                and iteration > 0):
            self._integrity_post_forward(self.embed_shard_current, iteration - 1)

        if self.config.cca_cross_stream_alloc:
            self._cca_pressure(iteration)

        self._prefetch_h2d(iteration + 1)
        self._prefetch_datadist(iteration)
        self._update_metrics(output, labels)
        self._gpu_padding()
        self._swap_buffers()

    def _run_iteration_pipelined_full_sync(self, iteration: int) -> None:
        """Pipelined iteration with sync at ALL pipeline points (Experiment B)."""
        self.default_stream.wait_stream(self.memcpy_stream)
        if self.datadist_stream is not None:
            self.default_stream.wait_stream(self.datadist_stream)
        torch.cuda.synchronize()

        if self.config.cca_cross_stream_alloc:
            self._cca_record_streams()

        if not self._datadist_in_compiled:
            self._inject_datadist(self.batch_gpu_current)

        autocast_ctx = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if self.config.use_bfloat16 else nullcontext()
        )
        shard = self.embed_shard_current if self._datadist_in_compiled else None
        with torch.no_grad(), autocast_ctx:
            output = self._forward(
                self.batch_gpu_current, self.sparse_gpu_current,
                self.seq_emb_gpu_current, shard,
            )
            labels = self.labels_gpu_current

        self._model_fwd_alltoall()
        torch.cuda.synchronize()

        if self.config.cca_cross_stream_alloc:
            self._cca_pressure(iteration)

        self._prefetch_h2d(iteration + 1)
        self._prefetch_datadist(iteration)
        torch.cuda.synchronize()

        self._update_metrics(output, labels)
        torch.cuda.synchronize()

        self._gpu_padding()
        self._swap_buffers()

    def _run_iteration_unpipelined(self, iteration: int) -> None:
        """Fully independent iteration -- no cross-iteration buffer sharing."""
        cfg = self.config

        if cfg.fresh_buffers_each_iter:
            batch_gpu = torch.empty(
                cfg.batch_size, cfg.feature_dim,
                dtype=torch.float32, device="cuda",
            )
            labels_gpu = torch.empty(
                cfg.batch_size, 1,
                dtype=torch.float32, device="cuda",
            )
            sparse_gpu = None
            seq_emb_gpu = None
            if self._uses_sparse:
                sparse_gpu = torch.zeros(
                    cfg.num_embedding_tables, cfg.batch_size * cfg.sparse_pooling_factor,
                    dtype=torch.long, device="cuda",
                )
            if self._uses_seq_embeddings:
                seq_emb_gpu = torch.empty(
                    cfg.batch_size, cfg.seq_len, cfg.feature_dim,
                    dtype=torch.float32, device="cuda",
                )
        else:
            batch_gpu = self.batch_gpu_current
            labels_gpu = self.labels_gpu_current
            sparse_gpu = self.sparse_gpu_current
            seq_emb_gpu = self.seq_emb_gpu_current

        batch_cpu, labels_cpu, sparse_cpu, seq_emb_cpu = self._get_cpu_batch(iteration)
        batch_gpu.copy_(batch_cpu, non_blocking=True)
        labels_gpu.copy_(labels_cpu, non_blocking=True)
        if sparse_cpu is not None and sparse_gpu is not None:
            sparse_gpu.copy_(sparse_cpu, non_blocking=True)
        if seq_emb_cpu is not None and seq_emb_gpu is not None:
            seq_emb_gpu.copy_(seq_emb_cpu, non_blocking=True)
        torch.cuda.synchronize()

        shard = None
        if self.world_size > 1:
            dist.reduce_scatter_tensor(
                self.embed_shard_current, self.embed_full_current
            )
            torch.cuda.synchronize()
            if self._datadist_in_compiled:
                shard = self.embed_shard_current
            else:
                self._inject_datadist(batch_gpu)

        autocast_ctx = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if cfg.use_bfloat16 else nullcontext()
        )
        with torch.no_grad(), autocast_ctx:
            output = self._forward(batch_gpu, sparse_gpu, seq_emb_gpu, shard)

        self._model_fwd_alltoall()
        self._update_metrics(output, labels_gpu)
        torch.cuda.synchronize()

    # =========================================================================
    # Stubs required by BaseReproducer (not used -- we override run())
    # =========================================================================

    def setup_buffers(self) -> None:
        pass

    def run_iteration(self, iteration: int) -> bool:
        return True

    # =========================================================================
    # Run loop (overrides BaseReproducer.run)
    # =========================================================================

    def run(self) -> ReproducerResult:
        cfg = self.config
        self.setup()

        start_time = time.time()
        nan_detected = False
        first_nan_iter: Optional[int] = None
        total_iterations = cfg.warmup_iterations + cfg.verify_iterations

        # Choose iteration function
        if not cfg.enable_pipelining:
            iter_fn = self._run_iteration_unpipelined
        elif cfg.sync_policy == "all_pipeline_points":
            iter_fn = self._run_iteration_pipelined_full_sync
        else:
            iter_fn = self._run_iteration_pipelined

        # Bootstrap pipeline: first H2D + datadist into "current" buffers
        if cfg.enable_pipelining:
            self._bootstrap_pipeline()

        log.info(
            f"Starting eval loop: {total_iterations} iterations, "
            f"sync_policy={cfg.sync_policy}, iter_fn={iter_fn.__name__}"
        )

        # Setup profiler if requested
        profiler = None
        trace_path: Optional[str] = None
        if cfg.profile:
            trace_dir = Path(cfg.profile_output_dir)
            trace_dir.mkdir(parents=True, exist_ok=True)
            trace_path = str(trace_dir / f"trace_rank{self.rank}_{cfg.model_type}_bs{cfg.batch_size}.json")
            prof_wait = max(1, cfg.warmup_iterations)
            prof_warmup = 2
            prof_active = cfg.profile_iterations
            profiler = torch_profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=schedule(
                    wait=prof_wait,
                    warmup=prof_warmup,
                    active=prof_active,
                    repeat=1,
                ),
                record_shapes=True,
                with_stack=True,
                with_flops=True,
            )
            profiler.__enter__()
            log.info(
                f"Profiler enabled: wait={prof_wait}, warmup={prof_warmup}, "
                f"active={prof_active}, output={trace_path}"
            )

        cpu_iter_times: List[float] = []
        loop_cpu_start = time.perf_counter()

        # CUDA events for non-blocking GPU lag tracking.  event.query()
        # returns True if the GPU has finished all work up to that point
        # on the default stream -- no synchronization, no overhead.
        gpu_events: List[torch.cuda.Event] = []
        last_gpu_done: int = 0

        for i in range(total_iterations):
            iter_cpu_t0 = time.perf_counter()
            iter_fn(i)
            cpu_iter_times.append(time.perf_counter() - iter_cpu_t0)

            ev = torch.cuda.Event()
            ev.record()
            gpu_events.append(ev)

            if profiler is not None:
                profiler.step()

            # Sync policy handling (Layer 2: Experiment A)
            should_check = False
            if cfg.sync_policy == "every_iter":
                torch.cuda.synchronize()
                should_check = True
            elif cfg.sync_policy == "periodic":
                if i > 0 and i % cfg.nan_check_interval == 0:
                    torch.cuda.synchronize()
                    should_check = True
            elif cfg.sync_policy == "all_pipeline_points":
                should_check = True

            if should_check:
                local_nan = self._check_nan()
                # All ranks must break at the same iteration to avoid
                # collective mismatches (one rank calling collectives while
                # the other has already exited the loop).
                if self.world_size > 1:
                    nan_flag = torch.tensor(
                        [1.0 if local_nan else 0.0], device="cuda"
                    )
                    dist.all_reduce(nan_flag)
                    any_nan = nan_flag.item() > 0
                else:
                    any_nan = local_nan

                if any_nan:
                    nan_detected = True
                    first_nan_iter = i
                    if local_nan:
                        log.error(
                            f"NaN DETECTED at iteration {i} (rank {self.rank})"
                        )
                    else:
                        log.warning(
                            f"Peer rank detected NaN at iteration {i} "
                            f"(rank {self.rank} is clean)"
                        )
                    self.corruption_details.append({
                        "type": "nan_in_metrics",
                        "iteration": i,
                        "rank": self.rank,
                        "local_nan": local_nan,
                    })
                    if cfg.stop_on_first_corruption:
                        break

            if (i + 1) % cfg.log_interval == 0:
                while last_gpu_done < len(gpu_events) and gpu_events[last_gpu_done].query():
                    last_gpu_done += 1
                gpu_lag = (i + 1) - last_gpu_done
                elapsed = time.perf_counter() - loop_cpu_start
                integrity_msg = ""
                if cfg.cca_integrity_check:
                    ic = self._integrity_read_corruption_count()
                    integrity_msg = f", integrity_corruptions={ic}"
                log.info(
                    f"[rank{self.rank}] Progress: {i + 1}/{total_iterations} | "
                    f"gpu_done={last_gpu_done}/{i + 1}, "
                    f"lag={gpu_lag} iters, "
                    f"elapsed={elapsed:.3f}s"
                    f"{integrity_msg}"
                )

        loop_cpu_end = time.perf_counter()

        # Export profiler trace
        if profiler is not None:
            profiler.__exit__(None, None, None)
            if trace_path:
                profiler.export_chrome_trace(trace_path)
                log.info(f"Profiler trace saved to: {trace_path}")

        # Report GPU lag from events before syncing (all ranks)
        if gpu_events:
            while last_gpu_done < len(gpu_events) and gpu_events[last_gpu_done].query():
                last_gpu_done += 1
            log.info(
                f"[rank{self.rank}] Loop done (CPU). GPU completed "
                f"{last_gpu_done}/{len(gpu_events)} iterations before final sync."
            )

        # Final sync -- measure how long GPU takes to drain
        pre_sync = time.perf_counter()
        torch.cuda.synchronize()
        sync_duration = time.perf_counter() - pre_sync

        log.info(f"[rank{self.rank}] Final sync (GPU drain): {sync_duration * 1000:.1f}ms")

        # CPU-GPU lag diagnostics (all ranks participate in calibration
        # iterations since they contain collectives).
        # Skip when requested -- the 13 extra iterations with collectives
        # can trigger NCCL watchdog timeouts if the GPU is still draining
        # hundreds of queued ops from the main loop.
        # Also skip if the CCA race may have corrupted NCCL state --
        # any post-loop collective (including a barrier) will hang.
        if len(cpu_iter_times) > 1 and not cfg.skip_lag_diagnostics:
            # Barrier: ensure all ranks finish GPU drain before calibration
            # iterations that enqueue collectives.
            if self.world_size > 1:
                dist.barrier()
            compile_ms = cpu_iter_times[0] * 1000
            steady = cpu_iter_times[1:]
            avg_cpu_us = (sum(steady) / len(steady)) * 1e6
            cpu_loop_total_ms = (loop_cpu_end - loop_cpu_start) * 1000
            sync_ms = sync_duration * 1000

            cal_times = []
            for ci in range(10):
                ct0 = time.perf_counter()
                iter_fn(total_iterations + ci)
                torch.cuda.synchronize()
                cal_times.append(time.perf_counter() - ct0)
            avg_gpu_ms = (sum(cal_times) / len(cal_times)) * 1000

            # Measure actual GPU dispatches per stream using profiler
            dispatch_info = self._measure_dispatches_per_stream(
                iter_fn, total_iterations + 10, num_iters=3,
            )

            if self.rank == 0:
                aql_size = int(os.environ.get("ROC_AQL_QUEUE_SIZE", "16384"))

                if avg_gpu_ms > 0:
                    cpu_lead_iters = sync_ms / avg_gpu_ms
                else:
                    cpu_lead_iters = 0

                log.info("")
                log.info("=" * 60)
                log.info("CPU-GPU LAG DIAGNOSTICS")
                log.info("=" * 60)
                log.info(f"  First iter CPU time (torch.compile JIT): {compile_ms:.1f}ms")
                log.info(f"  Steady-state CPU submit time/iter:  {avg_cpu_us:.1f}us")
                log.info(f"  GPU execution time/iter (synced):   {avg_gpu_ms:.3f}ms")
                log.info(f"  CPU/GPU ratio:  1:{avg_gpu_ms*1000/avg_cpu_us:.1f}  "
                          f"(CPU submits {avg_gpu_ms*1000/avg_cpu_us:.0f}x faster than GPU executes)")
                log.info(f"  CPU loop total (no sync):           {cpu_loop_total_ms:.1f}ms")
                log.info(f"  Final sync wait (GPU drain):        {sync_ms:.1f}ms")
                log.info(f"  Estimated CPU lead at loop end:     {cpu_lead_iters:.0f} iterations")

                # Per-stream dispatch breakdown
                log.info("")
                log.info("  GPU dispatches per iteration (profiler-measured):")
                total_dispatches = 0
                hot_stream_dispatches = 0
                hot_stream_name = "unknown"
                for sname, count in sorted(
                    dispatch_info.items(), key=lambda x: -x[1]
                ):
                    log.info(f"    stream {sname:>10s}: {count:4d} dispatches/iter")
                    total_dispatches += count
                    if count > hot_stream_dispatches:
                        hot_stream_dispatches = count
                        hot_stream_name = sname
                log.info(f"    {'total':>10s}: {total_dispatches:4d} dispatches/iter")
                log.info(f"  Hot stream: {hot_stream_name} "
                         f"({hot_stream_dispatches} dispatches/iter)")

                log.info("")
                log.info(f"  AQL queue size:                     {aql_size}")
                max_iters_in_queue = aql_size / max(hot_stream_dispatches, 1)
                log.info(f"  Hot-stream dispatches/iter:          {hot_stream_dispatches}")
                log.info(f"  Max iterations in hot queue:         ~{max_iters_in_queue:.0f}")
                est_queue_fill = (cpu_lead_iters * hot_stream_dispatches / aql_size) * 100
                log.info(f"  Estimated hot-queue AQL fill:        ~{est_queue_fill:.1f}%")
                if cpu_lead_iters < 10:
                    log.info("")
                    log.info("  ** GPU is keeping up -- CPU is NOT racing ahead **")
                    log.info("  ** AQL queue is NOT filling up significantly **")
                    log.info("  ** This explains why NaN does not reproduce on this hardware **")
                elif est_queue_fill > 80:
                    log.info("")
                    log.info("  ** Hot-queue AQL is near/over capacity -- race window is OPEN **")
                log.info("=" * 60)
        elif cfg.skip_lag_diagnostics and self.rank == 0:
            compile_ms = cpu_iter_times[0] * 1000 if cpu_iter_times else 0
            steady = cpu_iter_times[1:] if len(cpu_iter_times) > 1 else []
            avg_cpu_us = (sum(steady) / len(steady)) * 1e6 if steady else 0
            log.info("")
            log.info("Lag diagnostics skipped (--skip-lag-diagnostics)")
            log.info(f"  Final sync wait (GPU drain): {sync_duration * 1000:.1f}ms")
            if avg_cpu_us > 0:
                log.info(f"  Steady-state CPU submit time/iter: {avg_cpu_us:.1f}us")
        if not nan_detected and self._check_nan():
            nan_detected = True
            first_nan_iter = -1
            log.error(
                f"NaN DETECTED in final metrics (rank {self.rank}). "
                f"Exact iteration unknown (sync_policy={cfg.sync_policy})."
            )
            self.corruption_details.append({
                "type": "nan_in_metrics_final",
                "rank": self.rank,
            })

        elapsed = time.time() - start_time
        avg_ms = (elapsed * 1000) / total_iterations if total_iterations > 0 else 0

        # Log metric values
        if self.rank == 0:
            ne_val = self.ne_sum.item() if not torch.isnan(self.ne_sum) else "NaN"
            mae_val = self.mae_sum.item() if not torch.isnan(self.mae_sum) else "NaN"
            log.info(
                f"Metrics: NE_sum={ne_val}, MAE_sum={mae_val}, "
                f"cal_pred_sum={self.cal_pred_sum.item()}, "
                f"cal_label_sum={self.cal_label_sum.item()}, "
                f"count={self.metric_count}"
            )

        if nan_detected:
            log.error(
                f"NaN DETECTED: first_iter={first_nan_iter}, "
                f"total_iters={total_iterations}, elapsed={elapsed:.2f}s"
            )
        else:
            log.info(
                f"PASSED: No NaN in {total_iterations} iterations, "
                f"elapsed={elapsed:.2f}s, avg_step={avg_ms:.2f}ms"
            )

        if cfg.cca_cross_stream_alloc and self._cca_reuse_count > 0:
            self._cca_log_summary()

        integrity_corruptions = 0
        if cfg.cca_integrity_check:
            integrity_corruptions = self._integrity_report()
            if integrity_corruptions > 0:
                nan_detected = True

        return ReproducerResult(
            passed=not nan_detected,
            total_iterations=total_iterations,
            corruption_count=max(1 if nan_detected else 0, integrity_corruptions),
            first_corruption_iter=first_nan_iter,
            corruption_details=self.corruption_details,
            elapsed_time_sec=elapsed,
            avg_step_time_ms=avg_ms,
        )

    def _bootstrap_pipeline(self) -> None:
        """Fill the pipeline's 'current' buffers for the first iteration."""
        batch_cpu, labels_cpu, sparse_cpu, seq_emb_cpu = self._get_cpu_batch(0)
        with torch.cuda.stream(self.memcpy_stream):
            self.batch_gpu_current.copy_(batch_cpu, non_blocking=True)
            self.labels_gpu_current.copy_(labels_cpu, non_blocking=True)
            if sparse_cpu is not None and self.sparse_gpu_current is not None:
                self.sparse_gpu_current.copy_(sparse_cpu, non_blocking=True)
            if seq_emb_cpu is not None and self.seq_emb_gpu_current is not None:
                self.seq_emb_gpu_current.copy_(seq_emb_cpu, non_blocking=True)

        if self.world_size > 1:
            stream_ctx = (
                torch.cuda.stream(self.datadist_stream)
                if self.datadist_stream is not None
                else nullcontext()
            )
            with stream_ctx:
                dist.reduce_scatter_tensor(
                    self.embed_shard_current, self.embed_full_current
                )
        torch.cuda.synchronize()
        log.info("Pipeline bootstrapped (first batch loaded)")


__all__ = ["EvalPipelinedReproducer", "EvalModel"]
