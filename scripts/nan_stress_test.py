"""
End-to-end NaN stress test with real model, Distributed Shampoo, and
simulated TorchRec embedding traffic.

Runs a real training loop with maximum stream contention to reproduce
the non-deterministic NaN / HSA_STATUS_ERROR_EXCEPTION.

Usage:
    GPU_MAX_HW_QUEUES=4 PYTHONPATH=src torchrun --nproc_per_node=8 \
        scripts/nan_stress_test.py --optimizer shampoo --max-steps 2000

    GPU_MAX_HW_QUEUES=4 PYTHONPATH=src torchrun --nproc_per_node=8 \
        scripts/nan_stress_test.py --config config/nan_stress_test.yaml

    GPU_MAX_HW_QUEUES=2 PYTHONPATH=src torchrun --nproc_per_node=8 \
        scripts/nan_stress_test.py --optimizer shampoo --max-steps 2000
"""

import argparse
import json
import logging
import math
import os
import signal
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass, fields
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | R%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# =========================================================================
# Configuration
# =========================================================================


@dataclass
class NaNStressConfig:
    # Training
    max_steps: int = 50_000
    duration_hours: float = 0.0
    batch_size: int = 64
    mixed_precision: str = "bf16"
    grad_clip_norm: float = 1.0
    gradient_accumulation: int = 1
    compile: bool = False

    # Model
    vocab_size: int = 350_000
    embedding_dim: int = 256
    num_dense_features: int = 32
    dense_dim: int = 256
    model_dim: int = 1024
    num_heads: int = 16
    num_layers: int = 18
    dropout: float = 0.1
    mlp_hidden_dim: int = 4096

    # Dataset
    sequence_length: int = 64
    sparse_features: int = 64
    num_samples: int = 200_000

    # Optimizer
    optimizer: str = "shampoo"
    lr: float = 2e-4
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.985)
    eps: float = 1e-8

    # Shampoo-specific
    precondition_frequency: int = 50
    max_preconditioner_dim: int = 8192
    start_preconditioning_step: int = 50

    # Embedding stress (TorchRec simulation)
    num_embedding_tables: int = 8
    emb_table_rows: int = 2_000_000
    emb_table_dim: int = 128
    emb_lookups_per_iter: int = 8192
    num_extra_streams: int = 4

    # Extra NCCL stress
    extra_nccl_streams: int = 0
    extra_nccl_allgather_size: int = 2_000_000

    # Hardware
    gpu_max_hw_queues: int = 4

    # Monitoring
    log_interval: int = 20
    nan_check_interval: int = 1
    shampoo_state_check_interval: int = 100
    stop_on_nan: bool = True


def load_config(args) -> NaNStressConfig:
    """Load config from YAML file (if provided), then apply CLI overrides."""
    cfg = NaNStressConfig()

    if args.config:
        try:
            import yaml
        except ImportError:
            log.warning("PyYAML not installed; ignoring --config. pip install pyyaml")
            return _apply_cli_overrides(cfg, args)

        with open(args.config) as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Config file must be a YAML dict, got {type(data)}")

        valid_fields = {f.name for f in fields(cfg)}
        for key, val in data.items():
            if key in valid_fields:
                if key == "betas" and isinstance(val, list):
                    val = tuple(val)
                setattr(cfg, key, val)
            else:
                log.warning(f"Unknown config key '{key}' in {args.config}, ignoring")

    return _apply_cli_overrides(cfg, args)


def _apply_cli_overrides(cfg: NaNStressConfig, args) -> NaNStressConfig:
    """CLI arguments override config file values (when explicitly provided)."""
    if args.optimizer is not None:
        cfg.optimizer = args.optimizer
    if args.max_steps is not None:
        cfg.max_steps = args.max_steps
    if args.duration_hours is not None:
        cfg.duration_hours = args.duration_hours
    if args.hw_queues is not None:
        cfg.gpu_max_hw_queues = args.hw_queues
    if args.embedding_tables is not None:
        cfg.num_embedding_tables = args.embedding_tables
    if args.precondition_frequency is not None:
        cfg.precondition_frequency = args.precondition_frequency
    if args.extra_streams is not None:
        cfg.num_extra_streams = args.extra_streams
    if args.gradient_accumulation is not None:
        cfg.gradient_accumulation = args.gradient_accumulation
    if args.extra_nccl_streams is not None:
        cfg.extra_nccl_streams = args.extra_nccl_streams
    if args.compile_model:
        cfg.compile = True
    return cfg


# =========================================================================
# Signal handler for HSA crashes
# =========================================================================

_CRASH_STATE: Dict = {"step": -1, "rank": -1, "nan_checker": None}


def _crash_handler(signum, frame):
    step = _CRASH_STATE["step"]
    rank = _CRASH_STATE["rank"]
    nan_checker = _CRASH_STATE["nan_checker"]
    try:
        sig_name = signal.Signals(signum).name
    except (ValueError, AttributeError):
        sig_name = str(signum)

    msg = f"SIGNAL {sig_name} at step {step} on rank {rank}"
    log.error(msg)

    crash_info = {
        "signal": sig_name,
        "step": step,
        "rank": rank,
    }
    if nan_checker:
        crash_info["nan_summary"] = nan_checker.summary()

    try:
        path = f"nan_stress_crash_rank{rank}.json"
        with open(path, "w") as f:
            json.dump(crash_info, f, indent=2)
        log.error(f"Crash state saved to {path}")
    except Exception:
        pass

    sys.exit(128 + signum)


# =========================================================================
# Embedding stress simulator
# =========================================================================


class EmbeddingStressSimulator:
    """
    Simulates TorchRec-style sharded embedding lookups with all_to_all.

    Creates multiple large embedding tables on each GPU and runs lookups
    on dedicated streams, then redistributes via all_to_all on datadist_stream.
    """

    def __init__(
        self,
        cfg: NaNStressConfig,
        rank: int,
        world_size: int,
        dtype: torch.dtype,
    ):
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size

        self.datadist_stream = torch.cuda.Stream()
        self.embedding_streams = [
            torch.cuda.Stream() for _ in range(cfg.num_extra_streams)
        ]
        self.default_stream = torch.cuda.current_stream()

        self.tables: List[nn.Embedding] = []
        for _ in range(cfg.num_embedding_tables):
            table = nn.Embedding(cfg.emb_table_rows, cfg.emb_table_dim)
            table = table.to(device="cuda", dtype=dtype)
            self.tables.append(table)

        total_emb_elems = cfg.emb_lookups_per_iter * cfg.emb_table_dim
        self.send_buf = torch.empty(
            world_size, total_emb_elems, dtype=dtype, device="cuda",
        )
        self.recv_buf = torch.empty_like(self.send_buf)

        self.aggregated = torch.zeros(
            cfg.emb_lookups_per_iter, cfg.emb_table_dim,
            dtype=dtype, device="cuda",
        )

        self.indices = torch.randint(
            0, cfg.emb_table_rows, (cfg.emb_lookups_per_iter,), device="cuda",
        )

        log.info(
            f"EmbeddingStressSimulator: {cfg.num_embedding_tables} tables "
            f"({cfg.emb_table_rows}x{cfg.emb_table_dim}), "
            f"{cfg.num_extra_streams} streams, "
            f"{cfg.emb_lookups_per_iter} lookups/iter"
        )

    def run_lookups_and_alltoall(self) -> dist.Work:
        """
        Run embedding lookups on embedding streams, then all_to_all on
        datadist_stream.  Returns the async work handle.
        """
        num_streams = len(self.embedding_streams)

        self.indices.random_(0, self.cfg.emb_table_rows)

        # Each embedding stream must wait for index generation on default stream
        idx_event = torch.cuda.current_stream().record_event()
        for stream in self.embedding_streams:
            stream.wait_event(idx_event)

        chunk_size = self.send_buf.shape[1] // self.cfg.num_embedding_tables
        for t, table in enumerate(self.tables):
            stream = self.embedding_streams[t % num_streams]
            with torch.cuda.stream(stream):
                emb_out = table(self.indices)
                flat = emb_out.reshape(-1)
                col_start = t * chunk_size
                col_end = min(col_start + chunk_size, self.send_buf.shape[1])
                fill_size = min(flat.numel(), col_end - col_start)
                self.send_buf[:, col_start:col_start + fill_size] = flat[:fill_size]

        for stream in self.embedding_streams:
            self.datadist_stream.wait_stream(stream)

        with torch.cuda.stream(self.datadist_stream):
            work = dist.all_to_all_single(
                self.recv_buf, self.send_buf, async_op=True,
            )

        return work

    def wait_and_aggregate(self, work: dist.Work) -> torch.Tensor:
        """Wait for all_to_all and aggregate results."""
        self.default_stream.wait_stream(self.datadist_stream)
        work.wait()

        per_rank = self.recv_buf.view(
            self.world_size, self.cfg.emb_lookups_per_iter, self.cfg.emb_table_dim,
        )
        self.aggregated = per_rank.mean(dim=0)
        return self.aggregated


# =========================================================================
# Extra NCCL stress (concurrent AllGather during optimizer.step)
# =========================================================================


class NCCLStressor:
    """
    Runs extra concurrent NCCL AllGather operations on dedicated streams.

    This amplifies the contention pattern that triggers the race:
    DDP all_reduce + Shampoo AllGather + these extra AllGathers all
    running simultaneously across different HW queues.
    """

    def __init__(self, num_streams: int, allgather_size: int,
                 world_size: int, dtype: torch.dtype):
        self.streams = [torch.cuda.Stream() for _ in range(num_streams)]
        self.local_bufs = []
        self.gathered_bufs = []
        for _ in range(num_streams):
            local = torch.randn(allgather_size, dtype=dtype, device="cuda")
            gathered = torch.empty(
                allgather_size * world_size, dtype=dtype, device="cuda",
            )
            self.local_bufs.append(local)
            self.gathered_bufs.append(gathered)

        log.info(
            f"NCCLStressor: {num_streams} extra streams, "
            f"allgather_size={allgather_size}"
        )

    def fire(self):
        """Launch concurrent AllGather ops on all extra streams."""
        for i, stream in enumerate(self.streams):
            self.local_bufs[i].normal_()
            with torch.cuda.stream(stream):
                dist.all_gather_into_tensor(
                    self.gathered_bufs[i], self.local_bufs[i],
                )

    def wait(self, target_stream: torch.cuda.Stream):
        """Wait for all extra NCCL ops to complete on target_stream."""
        for stream in self.streams:
            target_stream.wait_stream(stream)


# =========================================================================
# NaN checker
# =========================================================================


class NaNChecker:
    """Tracks NaN/Inf occurrences across training."""

    def __init__(self):
        self.nan_steps: List[int] = []
        self.total_nans = 0

    def check_loss(self, loss: torch.Tensor, step: int) -> bool:
        val = loss.item()
        if math.isnan(val) or math.isinf(val):
            self.nan_steps.append(step)
            self.total_nans += 1
            log.error(f"NaN/Inf in LOSS at step {step}: {val}")
            return False
        return True

    def check_gradients(self, model: nn.Module, step: int) -> bool:
        ok = True
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                nan_ct = torch.isnan(param.grad).sum().item()
                inf_ct = torch.isinf(param.grad).sum().item()
                self.nan_steps.append(step)
                self.total_nans += 1
                log.error(
                    f"NaN/Inf in GRADIENT at step {step}: {name} "
                    f"(NaN={nan_ct}, Inf={inf_ct})"
                )
                ok = False
        return ok

    def check_parameters(self, model: nn.Module, step: int) -> bool:
        ok = True
        for name, param in model.named_parameters():
            if torch.isnan(param.data).any() or torch.isinf(param.data).any():
                nan_ct = torch.isnan(param.data).sum().item()
                inf_ct = torch.isinf(param.data).sum().item()
                self.nan_steps.append(step)
                self.total_nans += 1
                log.error(
                    f"NaN/Inf in PARAMETER at step {step}: {name} "
                    f"(NaN={nan_ct}, Inf={inf_ct})"
                )
                ok = False
        return ok

    def check_shampoo_state(self, optimizer, step: int) -> bool:
        """Inspect Shampoo's distributed preconditioner state for NaN."""
        ok = True
        try:
            state_dict = optimizer.distributed_state_dict()
        except (AttributeError, TypeError):
            return True

        for key, val in state_dict.items():
            if not isinstance(val, torch.Tensor):
                continue
            if torch.isnan(val).any() or torch.isinf(val).any():
                nan_ct = int(torch.isnan(val).sum().item())
                inf_ct = int(torch.isinf(val).sum().item())
                self.nan_steps.append(step)
                self.total_nans += 1
                log.error(
                    f"NaN/Inf in SHAMPOO STATE at step {step}: {key} "
                    f"(NaN={nan_ct}, Inf={inf_ct})"
                )
                ok = False
        return ok

    def summary(self) -> str:
        if self.total_nans == 0:
            return "No NaN/Inf detected."
        return (
            f"TOTAL NaN/Inf events: {self.total_nans}, "
            f"first at step {self.nan_steps[0]}, "
            f"affected steps: {sorted(set(self.nan_steps))[:20]}"
        )


# =========================================================================
# Double-buffered batch generator
# =========================================================================


class DoubleBufferedBatchGenerator:
    """
    Pre-allocates two sets of pinned CPU + GPU buffers for H2D pipelining.

    Generates fresh random data each step (unlike static batch reuse),
    exercising different numerical paths through Shampoo's eigendecomposition.
    """

    def __init__(self, cfg: NaNStressConfig, device: torch.device,
                 dtype: torch.dtype, memcpy_stream: torch.cuda.Stream):
        self.cfg = cfg
        self.device = device
        self.dtype = dtype
        self.memcpy_stream = memcpy_stream
        self.default_stream = torch.cuda.current_stream()

        self._cpu_bufs = [self._alloc_cpu_batch(), self._alloc_cpu_batch()]
        self._gpu_bufs: List[Optional[Dict[str, torch.Tensor]]] = [None, None]
        self._current = 0
        self._prefetched = False

    def _alloc_cpu_batch(self) -> Dict[str, torch.Tensor]:
        B = self.cfg.batch_size
        T = self.cfg.sequence_length
        F = self.cfg.num_dense_features
        D = self.cfg.dense_dim
        S = self.cfg.sparse_features

        dense = torch.empty(B, T, F, D, dtype=torch.float32).pin_memory()
        categorical = torch.empty(B, T, S, dtype=torch.long).pin_memory()
        target = torch.empty(B, T, dtype=torch.float32).pin_memory()
        return {"dense": dense, "categorical": categorical, "target": target}

    def _fill_batch(self, batch: Dict[str, torch.Tensor]):
        """Fill a CPU batch with fresh random data."""
        batch["dense"].normal_()
        batch["categorical"].random_(0, self.cfg.vocab_size)
        batch["target"].uniform_()

    def _transfer_to_gpu(self, cpu_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        gpu_batch = {}
        with torch.cuda.stream(self.memcpy_stream):
            for k, v in cpu_batch.items():
                target_dtype = self.dtype if v.is_floating_point() else v.dtype
                gpu_batch[k] = v.to(self.device, non_blocking=True, dtype=target_dtype)
        return gpu_batch

    def get_batch(self) -> Dict[str, torch.Tensor]:
        """Get the current batch (transfers to GPU if not prefetched)."""
        idx = self._current
        if self._gpu_bufs[idx] is None:
            self._fill_batch(self._cpu_bufs[idx])
            self._gpu_bufs[idx] = self._transfer_to_gpu(self._cpu_bufs[idx])
        return self._gpu_bufs[idx]

    def prefetch_next(self):
        """Start H2D for the next batch on memcpy_stream (call during backward)."""
        next_idx = 1 - self._current
        self._fill_batch(self._cpu_bufs[next_idx])
        self._gpu_bufs[next_idx] = self._transfer_to_gpu(self._cpu_bufs[next_idx])
        self._prefetched = True

    def swap(self):
        """Swap to the prefetched buffer for the next iteration."""
        old_idx = self._current
        if self._prefetched:
            self._current = 1 - self._current
            self._prefetched = False
        self._gpu_bufs[old_idx] = None

    def wait_h2d(self):
        """Wait for H2D to complete on the default stream."""
        self.default_stream.wait_stream(self.memcpy_stream)


# =========================================================================
# Embedding projection (strong coupling)
# =========================================================================


class EmbeddingProjection(nn.Module):
    """Projects aggregated embedding output into model's dense feature space."""

    def __init__(self, emb_dim: int, dense_dim: int):
        super().__init__()
        self.proj = nn.Linear(emb_dim, dense_dim)

    def forward(self, emb_agg: torch.Tensor) -> torch.Tensor:
        projected = self.proj(emb_agg.mean(dim=0))
        return projected


# =========================================================================
# Main
# =========================================================================


def init_distributed():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def create_model(cfg: NaNStressConfig, device: torch.device) -> nn.Module:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from aorta.models import ModelConfig, RankingTransformerModel

    model_cfg = ModelConfig(
        vocab_size=cfg.vocab_size,
        embedding_dim=cfg.embedding_dim,
        num_dense_features=cfg.num_dense_features,
        dense_dim=cfg.dense_dim,
        model_dim=cfg.model_dim,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        mlp_hidden_dim=cfg.mlp_hidden_dim,
    )
    model = RankingTransformerModel(model_cfg).to(device)
    return model


def create_optimizer(model: nn.Module, cfg: NaNStressConfig,
                     extra_params=None):
    params = list(model.parameters())
    if extra_params:
        params = params + list(extra_params)

    if cfg.optimizer.lower() == "shampoo":
        from distributed_shampoo import (
            DDPDistributedConfig,
            DistributedShampoo,
        )

        distributed_config = DDPDistributedConfig(
            communication_dtype=torch.float32,
            num_trainers_per_group=-1,
            communicate_params=False,
        )
        log.info(
            f"Using DistributedShampoo (lr={cfg.lr}, "
            f"precondition_freq={cfg.precondition_frequency}, "
            f"max_precond_dim={cfg.max_preconditioner_dim})"
        )
        optimizer = DistributedShampoo(
            params,
            lr=cfg.lr,
            betas=cfg.betas,
            epsilon=cfg.eps,
            weight_decay=cfg.weight_decay,
            max_preconditioner_dim=cfg.max_preconditioner_dim,
            precondition_frequency=cfg.precondition_frequency,
            start_preconditioning_step=cfg.start_preconditioning_step,
            distributed_config=distributed_config,
        )
    elif cfg.optimizer.lower() in ("adam", "adamw"):
        log.info(f"Using AdamW (lr={cfg.lr})")
        optimizer = torch.optim.AdamW(
            params,
            lr=cfg.lr,
            betas=cfg.betas,
            eps=cfg.eps,
            weight_decay=cfg.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer}")

    return optimizer


def should_stop(step: int, cfg: NaNStressConfig, start_time: float) -> bool:
    """Check both step-based and time-based stopping conditions."""
    if step >= cfg.max_steps:
        return True
    if cfg.duration_hours > 0:
        elapsed_hours = (time.time() - start_time) / 3600.0
        if elapsed_hours >= cfg.duration_hours:
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description="NaN stress test")
    parser.add_argument("--config", type=str, default=None,
                        help="YAML config file (e.g. config/nan_stress_test.yaml)")
    parser.add_argument("--optimizer", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--duration-hours", type=float, default=None,
                        help="Max runtime in hours (0=unlimited)")
    parser.add_argument("--hw-queues", type=int, default=None)
    parser.add_argument("--embedding-tables", type=int, default=None)
    parser.add_argument("--precondition-frequency", type=int, default=None)
    parser.add_argument("--no-embeddings", action="store_true")
    parser.add_argument("--extra-streams", type=int, default=None)
    parser.add_argument("--gradient-accumulation", type=int, default=None,
                        help="Gradient accumulation steps (default: 1)")
    parser.add_argument("--compile", dest="compile_model", action="store_true",
                        help="Enable torch.compile with inductor backend")
    parser.add_argument("--extra-nccl-streams", type=int, default=None,
                        help="Extra NCCL AllGather streams for contention stress")
    args = parser.parse_args()

    cfg = load_config(args)

    # GPU_MAX_HW_QUEUES must be set before any CUDA call
    os.environ["GPU_MAX_HW_QUEUES"] = str(cfg.gpu_max_hw_queues)

    rank, world_size, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}")
    dtype = torch.bfloat16 if cfg.mixed_precision == "bf16" else torch.float32

    nan_checker = NaNChecker()

    # Install signal handler for HSA_STATUS_ERROR_EXCEPTION
    _CRASH_STATE.update(rank=rank, nan_checker=nan_checker)
    signal.signal(signal.SIGABRT, _crash_handler)

    if rank == 0:
        log.info("=" * 70)
        log.info("NaN STRESS TEST")
        log.info("=" * 70)
        log.info(f"World size: {world_size}")
        log.info(f"Optimizer: {cfg.optimizer}")
        log.info(f"Max steps: {cfg.max_steps}")
        if cfg.duration_hours > 0:
            log.info(f"Duration limit: {cfg.duration_hours}h")
        log.info(f"Batch size: {cfg.batch_size}")
        log.info(f"Gradient accumulation: {cfg.gradient_accumulation}")
        log.info(f"Precision: {cfg.mixed_precision}")
        log.info(f"GPU_MAX_HW_QUEUES: {cfg.gpu_max_hw_queues}")
        log.info(f"Embedding tables: {cfg.num_embedding_tables}")
        log.info(f"Extra streams: {cfg.num_extra_streams}")
        log.info(f"Extra NCCL streams: {cfg.extra_nccl_streams}")
        log.info(f"torch.compile: {cfg.compile}")
        if cfg.optimizer.lower() == "shampoo":
            log.info(f"Shampoo precondition_frequency: {cfg.precondition_frequency}")
            log.info(f"Shampoo max_preconditioner_dim: {cfg.max_preconditioner_dim}")

        env_vars = [
            "GPU_MAX_HW_QUEUES", "ROC_SIGNAL_POOL_SIZE", "HSA_ENABLE_SDMA",
            "GPU_FORCE_BLIT_COPY_SIZE", "NCCL_LAUNCH_ORDER_IMPLICIT",
            "RCCL_GFX9_CHEAP_FENCE_OFF", "RCCL_GFX942_CHEAP_FENCE_OFF",
            "DEBUG_CLR_BATCH_CPU_SYNC_SIZE",
        ]
        log.info("Environment:")
        for var in env_vars:
            val = os.environ.get(var, "(not set)")
            log.info(f"  {var}={val}")
        log.info("=" * 70)

    # --- Model setup ---
    model = create_model(cfg, device)
    model = DDP(model, device_ids=[local_rank])

    # --- Embedding setup ---
    emb_sim = None
    emb_proj = None
    if not args.no_embeddings and cfg.num_embedding_tables > 0:
        emb_sim = EmbeddingStressSimulator(cfg, rank, world_size, dtype)
        emb_proj = EmbeddingProjection(cfg.emb_table_dim, cfg.dense_dim).to(
            device=device, dtype=dtype,
        )

    # Optimizer includes emb_proj parameters for gradient flow through embeddings
    extra_params = list(emb_proj.parameters()) if emb_proj else None
    optimizer = create_optimizer(model, cfg, extra_params=extra_params)

    # --- torch.compile ---
    if cfg.compile:
        if rank == 0:
            log.info("Applying torch.compile(backend='inductor')...")
        model = torch.compile(model, backend="inductor")

    # --- NCCL stressor ---
    nccl_stressor = None
    if cfg.extra_nccl_streams > 0:
        nccl_stressor = NCCLStressor(
            cfg.extra_nccl_streams, cfg.extra_nccl_allgather_size,
            world_size, dtype,
        )

    memcpy_stream = torch.cuda.Stream()
    default_stream = torch.cuda.current_stream()

    # --- Double-buffered batch generator ---
    batch_gen = DoubleBufferedBatchGenerator(cfg, device, dtype, memcpy_stream)

    start_time = time.time()
    loss_fn = nn.MSELoss()

    if rank == 0:
        log.info(f"Starting training (max_steps={cfg.max_steps}, "
                 f"duration={cfg.duration_hours}h)...")

    step = 0
    while not should_stop(step, cfg, start_time):
        _CRASH_STATE["step"] = step

        optimizer.zero_grad()

        # --- Gradient accumulation loop ---
        for micro_step in range(cfg.gradient_accumulation):
            use_ddp_sync = (micro_step == cfg.gradient_accumulation - 1)
            sync_ctx = nullcontext() if use_ddp_sync else model.no_sync()

            # --- H2D on memcpy_stream (fresh data each micro-step) ---
            gpu_batch = batch_gen.get_batch()

            # --- Embedding lookups + all_to_all (overlapping with H2D) ---
            emb_work = None
            if emb_sim is not None:
                emb_work = emb_sim.run_lookups_and_alltoall()

            # --- Wait for H2D ---
            batch_gen.wait_h2d()

            # --- Wait for embeddings and project into dense features ---
            if emb_sim is not None and emb_work is not None:
                emb_agg = emb_sim.wait_and_aggregate(emb_work)
                emb_features = emb_proj(emb_agg)
                gpu_batch["dense"] = gpu_batch["dense"] + emb_features

            with sync_ctx:
                # --- Forward ---
                with torch.amp.autocast("cuda", dtype=dtype):
                    scores = model(gpu_batch)
                    target = gpu_batch["target"]
                    loss = loss_fn(scores, target)
                    if cfg.gradient_accumulation > 1:
                        loss = loss / cfg.gradient_accumulation

                # --- Check loss for NaN ---
                if step % cfg.nan_check_interval == 0 and micro_step == 0:
                    loss_ok = nan_checker.check_loss(loss, step)
                    if not loss_ok and cfg.stop_on_nan:
                        log.error(f"Stopping at step {step} due to NaN in loss")
                        break

                # --- Backward (start prefetch during backward) ---
                batch_gen.prefetch_next()
                loss.backward()

            # Swap to prefetched buffer for next micro-step/iteration
            batch_gen.swap()

        else:
            # Only runs if inner loop didn't break
            # --- Check gradients for NaN ---
            if step % cfg.nan_check_interval == 0:
                grad_ok = nan_checker.check_gradients(model, step)
                if not grad_ok and cfg.stop_on_nan:
                    log.error(f"Stopping at step {step} due to NaN in gradients")
                    break

            # --- Gradient clipping ---
            if cfg.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)

            # --- Fire extra NCCL stress (concurrent with optimizer.step) ---
            if nccl_stressor is not None:
                nccl_stressor.fire()

            # --- Optimizer step ---
            optimizer.step()

            # --- Wait for extra NCCL stress to complete ---
            if nccl_stressor is not None:
                nccl_stressor.wait(default_stream)

            # --- Check parameters for NaN ---
            if step % cfg.nan_check_interval == 0:
                param_ok = nan_checker.check_parameters(model, step)
                if not param_ok and cfg.stop_on_nan:
                    log.error(f"Stopping at step {step} due to NaN in parameters")
                    break

            # --- Check Shampoo preconditioner state ---
            if (cfg.optimizer.lower() == "shampoo"
                    and step % cfg.shampoo_state_check_interval == 0
                    and step > 0):
                shampoo_ok = nan_checker.check_shampoo_state(optimizer, step)
                if not shampoo_ok and cfg.stop_on_nan:
                    log.error(
                        f"Stopping at step {step} due to NaN in Shampoo state"
                    )
                    break

            # --- Logging ---
            if rank == 0 and (step + 1) % cfg.log_interval == 0:
                elapsed = time.time() - start_time
                steps_per_sec = (step + 1) / elapsed
                ms_per_step = 1000.0 / steps_per_sec
                remaining = ""
                if cfg.duration_hours > 0:
                    hours_left = cfg.duration_hours - elapsed / 3600.0
                    remaining = f" | {hours_left:.2f}h remaining"
                log.info(
                    f"Step {step + 1}/{cfg.max_steps} | "
                    f"loss={loss.item():.4f} | "
                    f"{ms_per_step:.1f} ms/step | "
                    f"NaN={nan_checker.total_nans}"
                    f"{remaining}"
                )

            step += 1
            continue

        # Inner loop broke due to NaN
        break

    # --- Final report ---
    elapsed = time.time() - start_time

    try:
        dist.barrier()
    except Exception:
        pass

    if rank == 0:
        log.info("")
        log.info("=" * 70)
        log.info("RESULTS")
        log.info("=" * 70)
        log.info(f"Total steps: {step}")
        log.info(f"Elapsed: {elapsed:.1f}s ({elapsed / 3600:.2f}h)")
        log.info(f"Optimizer: {cfg.optimizer}")
        log.info(f"Gradient accumulation: {cfg.gradient_accumulation}")
        log.info(f"torch.compile: {cfg.compile}")
        log.info(f"Extra NCCL streams: {cfg.extra_nccl_streams}")
        log.info(f"NaN summary: {nan_checker.summary()}")
        log.info("=" * 70)

    sys.exit(1 if nan_checker.total_nans > 0 else 0)


if __name__ == "__main__":
    main()
