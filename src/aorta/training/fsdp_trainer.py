"""FSDP2 multi-stream training benchmark with profiling."""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import random
import signal
import subprocess
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, Optional
from functools import partial

import numpy as np

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.profiler import ProfilerActivity, schedule, tensorboard_trace_handler, profile
from torch.distributed.fsdp import BackwardPrefetch, FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP

from aorta.data import SyntheticDatasetConfig, create_dataloader
from aorta.losses import NormalizedEntropyLoss
from aorta.models import ModelConfig, RankingTransformerModel
from aorta.profiling.stream_profiler import StreamProfiler
from aorta.utils import (
    detect_accelerator,
    get_device,
    get_distributed_backend,
    load_config,
    manual_sync_params,
    merge_cli_overrides,
    setup_logging,
    warmup_rccl_communicators,
    warmup_training_collectives,
)

log = logging.getLogger(__name__)


@dataclass
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 2e-4
    weight_decay: float = 1e-2
    betas: tuple[float, float] = (0.9, 0.98)
    eps: float = 1e-8


@dataclass
class SchedulerConfig:
    warmup_steps: int = 200
    total_steps: int = 2000


@dataclass
class PrecisionConfig:
    param_dtype: str = "bf16"     # dtype for parameters during forward/backward: fp32, fp16, bf16
    reduce_dtype: str = "fp32"    # dtype for gradient all-reduce communication: fp32, fp16, bf16
    buffer_dtype: str = "fp32"    # dtype for module buffers (e.g. BatchNorm stats): fp32, fp16, bf16
    tf32_mode: str = "disabled"   # TF32 matmul mode: disabled, x1, x3


@dataclass
class LossConfig:
    name: str = "bce"  # "bce" or "normalized_entropy"
    ne_window_size: int = 100
    ne_initial_ctr: float = 0.1


@dataclass
class TrainingConfig:
    epochs: int = 1
    batch_size: int = 8
    gradient_accumulation: int = 1
    max_steps: Optional[int] = None
    grad_clip_norm: float = 1.0
    log_interval: int = 10
    output_dir: Path = Path("artifacts")
    inject_allreduce_copies: bool = False  # Inject all_reduce + host-device copies to trigger hang
    allreduce_stress_level: int = 1  # Number of all_reduce ops per iteration (1-10)


@dataclass
class FSDPConfig:
    sharding_strategy: str = "full_shard"
    backward_prefetch: str = "BACKWARD_PRE"
    use_orig_params: bool = True
    limit_all_gathers: bool = True
    forward_prefetch: bool = True
    sync_module_states: bool = True
    param_init_device: str = "cpu"
    # For HYBRID_SHARD: GPUs per node (None = auto-detect from LOCAL_WORLD_SIZE env var)
    # Only set this if auto-detection fails or you want to override
    hybrid_shard_gpus_per_node: Optional[int] = None
    # Number of warmup operations to perform on RCCL communicators before FSDP init
    # This helps avoid race conditions in inter-node RDMA setup
    # Higher values provide more stability but increase startup time
    rccl_warmup_iterations: int = 10
    # Skip RCCL warmup entirely (for testing race conditions)
    skip_rccl_warmup: bool = False
    # Number of training warmup steps (forward/backward/optimizer) before main loop
    # This exercises all collectives to ensure RCCL is fully established
    training_warmup_steps: int = 1
    # Skip training warmup entirely
    skip_training_warmup: bool = False


@dataclass
class CompileConfig:
    enabled: bool = False
    backend: Optional[str] = "inductor"
    mode: Optional[str] = "max-autotune"
    fullgraph: bool = False
    dynamic: bool = False
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DDPConfig:
    gradient_as_bucket_view: bool = True
    static_graph: bool = False
    bucket_cap_mb: int = 25
    find_unused_parameters: bool = False


@dataclass
class ProfilerConfig:
    enabled: bool = True
    wait: int = 1
    warmup: int = 1
    active: int = 2
    repeat: int = 1
    record_shapes: bool = True
    profile_memory: bool = True
    with_stack: bool = False
    with_flops: bool = False
    tensorboard: bool = False
    chrome_trace: bool = True
    trace_filename: str = "trace.json"


def _parse_config(args: argparse.Namespace) -> Dict[str, Any]:
    config_path = Path(args.config)
    config = load_config(config_path)
    config = merge_cli_overrides(config, args.override or [])
    return config


def _build_training_config(raw: Dict[str, Any]) -> TrainingConfig:
    training = raw.get("training", {})
    cfg = TrainingConfig()
    for field in dataclass_fields(TrainingConfig):
        if field.name in training:
            setattr(cfg, field.name, training[field.name])
    cfg.output_dir = Path(cfg.output_dir)
    return cfg


def _build_precision_config(raw: Dict[str, Any]) -> PrecisionConfig:
    section = raw.get("precision", {})
    cfg = PrecisionConfig()
    for field in dataclass_fields(PrecisionConfig):
        if field.name in section:
            setattr(cfg, field.name, section[field.name])

    for dtype_field in ("param_dtype", "reduce_dtype", "buffer_dtype"):
        _resolve_dtype(getattr(cfg, dtype_field))

    tf32 = cfg.tf32_mode.lower()
    if tf32 not in _VALID_TF32_MODES:
        raise ValueError(
            f"Unknown tf32_mode '{cfg.tf32_mode}'. "
            f"Expected one of: {', '.join(sorted(_VALID_TF32_MODES))}"
        )

    return cfg


def _build_loss_config(raw: Dict[str, Any]) -> LossConfig:
    section = raw.get("loss", {})
    cfg = LossConfig()
    for field in dataclass_fields(LossConfig):
        if field.name in section:
            setattr(cfg, field.name, section[field.name])
    return cfg


def _build_optimizer_config(raw: Dict[str, Any]) -> OptimizerConfig:
    section = raw.get("optimizer", {})
    cfg = OptimizerConfig()
    for field in dataclass_fields(OptimizerConfig):
        if field.name in section:
            setattr(cfg, field.name, section[field.name])
    return cfg


def _build_scheduler_config(raw: Dict[str, Any]) -> SchedulerConfig:
    section = raw.get("scheduler", {})
    cfg = SchedulerConfig()
    for field in dataclass_fields(SchedulerConfig):
        if field.name in section:
            setattr(cfg, field.name, section[field.name])
    return cfg


def _build_model_config(raw: Dict[str, Any]) -> ModelConfig:
    section = raw.get("model", {})
    cfg = ModelConfig()
    for field in dataclass_fields(ModelConfig):
        if field.name in section:
            setattr(cfg, field.name, section[field.name])
    return cfg


def _build_dataset_config(raw: Dict[str, Any]) -> SyntheticDatasetConfig:
    section = raw.get("dataset", {})
    cfg = SyntheticDatasetConfig()
    for field in dataclass_fields(SyntheticDatasetConfig):
        if field.name in section:
            setattr(cfg, field.name, section[field.name])
    return cfg


def _build_fsdp_config(raw: Dict[str, Any]) -> FSDPConfig:
    section = raw.get("fsdp", {})
    cfg = FSDPConfig()
    for field in dataclass_fields(FSDPConfig):
        if field.name in section:
            setattr(cfg, field.name, section[field.name])
    return cfg


def _build_ddp_config(raw: Dict[str, Any]) -> DDPConfig:
    section = raw.get("distributed", {})
    cfg = DDPConfig()
    for field in dataclass_fields(DDPConfig):
        if field.name in section:
            setattr(cfg, field.name, section[field.name])
    return cfg


def _build_compile_config(raw: Dict[str, Any]) -> CompileConfig:
    section = raw.get("compile", {})
    cfg = CompileConfig()
    for field in dataclass_fields(CompileConfig):
        if field.name in section:
            setattr(cfg, field.name, section[field.name])
    if cfg.options is None:
        cfg.options = {}
    return cfg


def _build_profiler_config(raw: Dict[str, Any]) -> ProfilerConfig:
    section = raw.get("profiling", {})
    cfg = ProfilerConfig()
    for field in dataclass_fields(ProfilerConfig):
        if field.name in section:
            setattr(cfg, field.name, section[field.name])
    return cfg


def dataclass_fields(cls) -> Iterable[Any]:
    return getattr(cls, "__dataclass_fields__").values()


_DTYPE_MAP: Dict[str, torch.dtype] = {
    "fp32": torch.float32,
    "float32": torch.float32,
    "none": torch.float32,
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
}

# Canonical names shown in error messages and validation output.
_DTYPE_CANONICAL_NAMES = ("fp32", "fp16", "bf16")

_VALID_TF32_MODES = frozenset({"disabled", "x1", "x3"})


def _resolve_dtype(name: str) -> torch.dtype:
    """Resolve a user-facing dtype string to a ``torch.dtype``.

    Accepts canonical short names (fp32, fp16, bf16), long names (float32,
    float16, bfloat16), and the legacy value ``"none"`` (mapped to fp32).
    """
    key = name.strip().lower()
    if key not in _DTYPE_MAP:
        raise ValueError(
            f"Unknown dtype '{name}'. Expected one of: {', '.join(_DTYPE_CANONICAL_NAMES)}"
        )
    return _DTYPE_MAP[key]


def build_fsdp_mixed_precision(precision_cfg: PrecisionConfig) -> Optional[MixedPrecision]:
    """Build an ``FSDP MixedPrecision`` policy from *precision_cfg*.

    Returns ``None`` when all dtypes resolve to fp32, since no mixed
    precision is needed in that case.
    """
    param_dtype = _resolve_dtype(precision_cfg.param_dtype)
    reduce_dtype = _resolve_dtype(precision_cfg.reduce_dtype)
    buffer_dtype = _resolve_dtype(precision_cfg.buffer_dtype)

    all_fp32 = (
        param_dtype == torch.float32
        and reduce_dtype == torch.float32
        and buffer_dtype == torch.float32
    )
    if all_fp32:
        log.info("FSDP MixedPrecision disabled (all dtypes are fp32)")
        return None

    policy = MixedPrecision(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        buffer_dtype=buffer_dtype,
    )
    log.info(
        "FSDP MixedPrecision | param_dtype=%s reduce_dtype=%s buffer_dtype=%s",
        param_dtype, reduce_dtype, buffer_dtype,
    )
    return policy


def init_distributed(training_cfg: TrainingConfig, log_level: str) -> Dict[str, Any]:
    backend = get_distributed_backend()
    timeout_seconds = int(os.environ.get("TORCH_DIST_INIT_TIMEOUT", "600"))
    dist.init_process_group(backend=backend, timeout=timedelta(seconds=timeout_seconds))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
    device = get_device(local_rank)
    torch.cuda.set_device(device)
    training_cfg.output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(level=log_level, log_file=training_cfg.output_dir / f"rank{rank}.log", rank=rank)

    log.info(
        "Initialised distributed training | backend=%s rank=%s world=%s local_rank=%s device=%s",
        backend,
        rank,
        world_size,
        local_rank,
        device,
    )

    return {
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "device": device,
    }


def build_fsdp_model(
    model_cfg: ModelConfig,
    fsdp_cfg: FSDPConfig,
    compile_cfg: CompileConfig,
    device: torch.device,
    mixed_precision_policy: Optional[MixedPrecision] = None,
) -> FSDP:
    model = RankingTransformerModel(model_cfg)

    sharding = getattr(ShardingStrategy, fsdp_cfg.sharding_strategy.upper())
    backward_prefetch = getattr(BackwardPrefetch, fsdp_cfg.backward_prefetch.upper())

    auto_wrap_policy = partial(
        transformer_auto_wrap_policy, transformer_layer_cls={nn.TransformerEncoderLayer}
    )

    # Create process groups for hybrid_shard strategy
    process_group = None
    shard_group = None
    replicate_group = None

    if sharding == ShardingStrategy.HYBRID_SHARD:
        result = _create_hybrid_shard_process_groups(fsdp_cfg.hybrid_shard_gpus_per_node)
        if result is not None:
            shard_group, replicate_group = result
            process_group = (shard_group, replicate_group)

            # Warmup RCCL communicators BEFORE FSDP initialization
            # This ensures inter-node communicators are fully established before
            # the _sync_params_and_buffers broadcasts that can cause hangs
            if fsdp_cfg.skip_rccl_warmup:
                log.warning("SKIPPING RCCL warmup (skip_rccl_warmup=True) - may cause hangs or race conditions")
            else:
                warmup_rccl_communicators(
                    shard_group,
                    replicate_group,
                    device,
                    num_warmup_ops=fsdp_cfg.rccl_warmup_iterations,
                )
            log.info("Created custom process groups for HYBRID_SHARD strategy")

    # Ensure GPU operations are complete before FSDP wrapping
    torch.cuda.synchronize()
    dist.barrier()

    # For HYBRID_SHARD with sync_module_states, we disable automatic sync and do it
    # manually with explicit barriers to avoid RCCL race conditions
    use_sync_module_states = fsdp_cfg.sync_module_states
    needs_manual_sync = False

    if sharding == ShardingStrategy.HYBRID_SHARD and fsdp_cfg.sync_module_states:
        use_sync_module_states = False
        needs_manual_sync = True
        log.info(
            "Disabling sync_module_states for HYBRID_SHARD - will sync manually after wrapping"
        )

    log.info("Starting FSDP model wrapping with sync_module_states=%s", use_sync_module_states)

    fsdp_kwargs: Dict[str, Any] = dict(
        sharding_strategy=sharding,
        process_group=process_group,
        auto_wrap_policy=auto_wrap_policy,
        use_orig_params=fsdp_cfg.use_orig_params,
        backward_prefetch=backward_prefetch,
        limit_all_gathers=fsdp_cfg.limit_all_gathers,
        forward_prefetch=fsdp_cfg.forward_prefetch,
        device_id=torch.cuda.current_device(),
        sync_module_states=use_sync_module_states,
    )
    if mixed_precision_policy is not None:
        fsdp_kwargs["mixed_precision"] = mixed_precision_policy

    fsdp_model = FSDP(model.to(device), **fsdp_kwargs)

    # Manual parameter synchronization for HYBRID_SHARD
    if needs_manual_sync and replicate_group is not None:
        manual_sync_params(fsdp_model, replicate_group)

    if compile_cfg.enabled:
        fsdp_model = _maybe_compile(fsdp_model, compile_cfg)
    return fsdp_model


def _create_hybrid_shard_process_groups(gpus_per_node: Optional[int] = None):
    """
    Create process groups for HYBRID_SHARD strategy.

    Args:
        gpus_per_node: Number of GPUs per node. If None, auto-detects from LOCAL_WORLD_SIZE.
                       This should match the --nproc value from torchrun.

    Returns:
        Tuple of (shard_group, replicate_group) or None
    """
    if not dist.is_initialized():
        return None

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    timeout_seconds = int(os.environ.get("TORCH_DIST_INIT_TIMEOUT", "600"))
    group_timeout = timedelta(seconds=timeout_seconds)

    # Auto-detect GPUs per node from environment if not provided
    if gpus_per_node is None:
        # torchrun sets LOCAL_WORLD_SIZE to the number of processes per node
        local_world_size_str = os.environ.get("LOCAL_WORLD_SIZE")
        if local_world_size_str:
            gpus_per_node = int(local_world_size_str)
            log.info("Auto-detected gpus_per_node=%d from LOCAL_WORLD_SIZE", gpus_per_node)
        else:
            log.error(
                "Cannot determine gpus_per_node: LOCAL_WORLD_SIZE not set and "
                "hybrid_shard_gpus_per_node not configured. Set hybrid_shard_gpus_per_node in config."
            )
            return None

    # Validate configuration
    if world_size % gpus_per_node != 0:
        log.error(
            "Invalid HYBRID_SHARD config: world_size=%d not divisible by gpus_per_node=%d",
            world_size, gpus_per_node
        )
        return None

    num_nodes = world_size // gpus_per_node
    node_id = rank // gpus_per_node

    if num_nodes <= 1:
        log.warning("HYBRID_SHARD with single node - consider using FULL_SHARD instead")
        return None

    log.info(
        "Creating HYBRID_SHARD process groups | rank=%d world_size=%d num_nodes=%d gpus_per_node=%d node_id=%d timeout=%ds",
        rank, world_size, num_nodes, gpus_per_node, node_id, timeout_seconds
    )

    # Intra-node groups: shard within each node
    for i in range(num_nodes):
        ranks_in_node = list(range(i * gpus_per_node, (i + 1) * gpus_per_node))
        group = dist.new_group(ranks=ranks_in_node, timeout=group_timeout)
        if i == node_id:
            my_shard_group = group

    # Inter-node groups: replicate across nodes (same local_rank)
    for local_r in range(gpus_per_node):
        ranks_across_nodes = [node * gpus_per_node + local_r for node in range(num_nodes)]
        group = dist.new_group(ranks=ranks_across_nodes, timeout=group_timeout)
        if local_r == local_rank:
            my_replicate_group = group

    log.info(
        "Created process groups | shard_group_size=%d replicate_group_size=%d",
        dist.get_world_size(my_shard_group),
        dist.get_world_size(my_replicate_group),
    )

    return (my_shard_group, my_replicate_group)


def build_ddp_model(
    model_cfg: ModelConfig,
    ddp_cfg: DDPConfig,
    compile_cfg: CompileConfig,
    device: torch.device,
) -> DDP:
    model = RankingTransformerModel(model_cfg).to(device)
    if compile_cfg.enabled:
        model = _maybe_compile(model, compile_cfg)

    device_ids = None
    if device.type == "cuda":
        device_ids = [device.index if device.index is not None else torch.cuda.current_device()]

    ddp_model = DDP(
        model,
        device_ids=device_ids,
        gradient_as_bucket_view=ddp_cfg.gradient_as_bucket_view,
        static_graph=ddp_cfg.static_graph,
        bucket_cap_mb=ddp_cfg.bucket_cap_mb,
        find_unused_parameters=ddp_cfg.find_unused_parameters,
    )
    return ddp_model


class MetricsLogger:
    def __init__(self, output_dir: Path, rank: int) -> None:
        self.path = output_dir / f"rank_{rank:02d}_metrics.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a", encoding="utf-8")

    def log(self, payload: Dict[str, Any]) -> None:
        self.handle.write(json.dumps(payload, default=self._serialize) + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()

    @staticmethod
    def _serialize(obj: Any) -> Any:
        if isinstance(obj, Path):
            return str(obj)
        return obj


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: tensor.to(device, non_blocking=True) for key, tensor in batch.items()}


_VALID_LOSS_NAMES = frozenset({"bce", "normalized_entropy"})


def build_loss_criterion(loss_cfg: LossConfig) -> Optional[nn.Module]:
    """Build the loss function from config.

    Returns ``None`` for BCE (handled inline by :func:`compute_loss`),
    or a :class:`NormalizedEntropyLoss` module for NE.

    Raises:
        ValueError: If ``loss_cfg.name`` is not a recognised loss function.
    """
    name = loss_cfg.name.lower().strip()
    if name not in _VALID_LOSS_NAMES:
        raise ValueError(
            f"Unknown loss function '{loss_cfg.name}'. "
            f"Expected one of: {', '.join(sorted(_VALID_LOSS_NAMES))}"
        )
    if name == "normalized_entropy":
        log.info(
            "Using NormalizedEntropyLoss | window_size=%d initial_ctr=%.3f",
            loss_cfg.ne_window_size,
            loss_cfg.ne_initial_ctr,
        )
        return NormalizedEntropyLoss(
            window_size=loss_cfg.ne_window_size,
            initial_ctr=loss_cfg.ne_initial_ctr,
        )
    log.info("Using BCE loss (default)")
    return None


def compute_loss(
    scores: torch.Tensor,
    batch: Dict[str, torch.Tensor],
    criterion: Optional[nn.Module] = None,
) -> torch.Tensor:
    target = batch["target"].to(scores.dtype)
    importance = batch["importance"].to(scores.dtype)
    if criterion is not None:
        return criterion(scores, target, weight=importance)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(scores, target, weight=importance)
    return loss.mean()


def collect_rocm_metrics(enabled: bool) -> Dict[str, Any]:
    if not enabled or detect_accelerator() != "amd":
        return {}
    try:
        result = subprocess.run(
            ["rocm-smi", "--showtemp", "--showuse", "--showpower", "--showmemuse"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return {"rocm_smi_error": "rocm-smi not available"}

    metrics: Dict[str, Any] = {"rocm_smi_exit_code": result.returncode}
    if result.stdout:
        metrics["rocm_smi_output"] = result.stdout.strip()
    if result.stderr:
        metrics["rocm_smi_stderr"] = result.stderr.strip()
    return metrics


def setup_signal_handlers(stop_flag: Dict[str, bool]) -> None:
    def _handle(signum, frame):  # pragma: no cover - signal handling
        stop_flag["stop"] = True
        log.warning("Received signal %s; will stop after current iteration", signum)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _handle)


def training_loop(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader,
    training_cfg: TrainingConfig,
    precision_cfg: PrecisionConfig,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    environment: Dict[str, Any],
    profiler: StreamProfiler,
    enable_rocm_metrics: bool,
    profiler_cfg: ProfilerConfig,
    *,
    use_autocast: bool = False,
    criterion: Optional[nn.Module] = None,
) -> None:
    rank = environment["rank"]
    world_size = environment["world_size"]
    device = environment["device"]

    param_dtype = _resolve_dtype(precision_cfg.param_dtype)
    scaler: Optional[torch.cuda.amp.GradScaler] = (
        torch.cuda.amp.GradScaler() if param_dtype == torch.float16 else None
    )
    autocast_dtype: Optional[torch.dtype] = (
        param_dtype if use_autocast and param_dtype != torch.float32 else None
    )

    metrics_logger = MetricsLogger(training_cfg.output_dir, rank)

    total_steps = training_cfg.max_steps or len(dataloader) * training_cfg.epochs
    global_step = 0
    stop_flag = {"stop": False}
    setup_signal_handlers(stop_flag)

    model.train()

    profiler_dir = training_cfg.output_dir / "torch_profiler"
    with profiler.intercept_distributed_ops():
        with _torch_profiler_context(profiler_cfg, profiler_dir, rank, device) as torch_profiler:
            for epoch in range(training_cfg.epochs):
                if hasattr(dataloader.sampler, "set_epoch"):
                    dataloader.sampler.set_epoch(epoch)

                for step, cpu_batch in enumerate(dataloader):
                    profiler.start_iteration(global_step)

                    with profiler.range("aux", f"epoch{epoch}_step{step}_prefetch"):
                        batch = move_batch_to_device(cpu_batch, device)

                    profiler.stream("compute").wait_stream(profiler.stream("aux"))

                    optimizer.zero_grad(set_to_none=True)

                    with profiler.range("compute", f"epoch{epoch}_step{step}_forward"):
                        if autocast_dtype:
                            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                                scores = model(batch)
                                loss = compute_loss(scores, batch, criterion)
                        else:
                            scores = model(batch)
                            loss = compute_loss(scores, batch, criterion)

                    with profiler.range("compute", f"epoch{epoch}_step{step}_backward"):
                        if scaler is not None:
                            scaler.scale(loss).backward()
                        else:
                            loss.backward()

                    # Synchronize before gradient clipping to ensure backward is complete
                    # This prevents race conditions between FSDP gradient reduction and clipping
                    torch.cuda.synchronize()

                    grad_norm = None
                    if training_cfg.grad_clip_norm is not None and training_cfg.grad_clip_norm > 0:
                        with profiler.range("aux", f"epoch{epoch}_step{step}_grad_clip"):
                            grad_norm = clip_grad_norm_(model.parameters(), training_cfg.grad_clip_norm)

                    with profiler.range("aux", f"epoch{epoch}_step{step}_optimizer"):
                        if scaler is not None:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()

                        if scheduler is not None:
                            scheduler.step()

                    # Inject all_reduce operations to trigger hang pattern
                    # Pattern: all_reduce → device-to-device copy → host-device copy → compute blocked
                    if training_cfg.inject_allreduce_copies:
                        import torch.distributed as dist
                        if dist.is_initialized():
                            with profiler.range("aux", f"epoch{epoch}_step{step}_allreduce_sync"):
                                # Perform multiple all_reduce + memory copy cycles
                                # This stresses the pattern: all_reduce → device copies → hipMemcpyWithStream → rocprim deadlock
                                stress_level = min(max(training_cfg.allreduce_stress_level, 1), 10)

                                for i in range(stress_level):
                                    # Create moderately-sized tensors to stress RCCL and memory copy
                                    # Size: ~4MB per tensor
                                    tensor_size = 1024 * 1024  # 1M elements = 4MB in FP32
                                    stress_tensor = torch.randn(tensor_size, device=device, dtype=torch.float32)

                                    # All-reduce operation (collective that triggers RCCL multi-stream)
                                    dist.all_reduce(stress_tensor, op=dist.ReduceOp.AVG)

                                    # Device-to-device copy (triggers hipMemcpyAsync device-to-device)
                                    # This happens when moving data between GPU memory regions or during P2P transfers
                                    device_copy_1 = stress_tensor.clone()  # Explicit device copy
                                    device_copy_2 = device_copy_1.contiguous()  # May trigger another copy if not contiguous

                                    # Force device-to-device copy via different tensor
                                    temp_storage = torch.empty_like(device_copy_2)
                                    temp_storage.copy_(device_copy_2, non_blocking=False)  # Blocking device-to-device copy

                                    # All-reduce on device-copied tensor
                                    dist.all_reduce(temp_storage, op=dist.ReduceOp.SUM)

                                    # Force blocking host-device copy (triggers hipMemcpyWithStream)
                                    # This is the pattern that causes the hang according to customer
                                    tensor_cpu = temp_storage.cpu()  # Device → Host copy
                                    tensor_back = tensor_cpu.to(device, non_blocking=False)  # Host → Device blocking copy

                                    # Additional all_reduce on the copied-back tensor
                                    dist.all_reduce(tensor_back, op=dist.ReduceOp.AVG)

                                    # More device-to-device copies after all-reduce
                                    final_copy = tensor_back.clone()
                                    _ = final_copy.contiguous()

                                    # Clean up
                                    del stress_tensor, device_copy_1, device_copy_2, temp_storage
                                    del tensor_cpu, tensor_back, final_copy

                                # Also all-reduce actual metrics (common pattern) with device copies
                                loss_tensor = loss.detach().clone()
                                dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)

                                # Device-to-device copy
                                loss_device_copy = loss_tensor.clone()

                                # Host-device copy
                                loss_cpu = loss_device_copy.cpu()
                                _ = loss_cpu.to(device, non_blocking=False)

                    profiler.record_marker("compute", f"epoch{epoch}_step{step}_end")

                    iteration_profile = profiler.end_iteration()

                    iteration_payload = {
                        "rank": rank,
                        "world_size": world_size,
                        "epoch": epoch,
                        "step": step,
                        "global_step": global_step,
                        "loss": float(loss.detach().cpu()),
                        "grad_norm": float(grad_norm.cpu()) if grad_norm is not None else None,
                        "lr": optimizer.param_groups[0]["lr"],
                        "profile": iteration_profile,
                    }
                    if isinstance(criterion, NormalizedEntropyLoss):
                        iteration_payload["ne"] = criterion.normalized_entropy
                        iteration_payload["background_ctr"] = criterion.background_ctr

                    iteration_payload.update(collect_rocm_metrics(enable_rocm_metrics))
                    metrics_logger.log(iteration_payload)

                    if global_step % training_cfg.log_interval == 0 and rank == 0:
                        ne_str = ""
                        if "ne" in iteration_payload:
                            ne_str = f" NE={iteration_payload['ne']:.4f} bg_ctr={iteration_payload['background_ctr']:.4f}"
                        log.info(
                            "epoch=%s step=%s loss=%.5f lr=%.6f%s overlap=%.3fms compute=%.3fms",
                            epoch,
                            step,
                            iteration_payload["loss"],
                            iteration_payload["lr"],
                            ne_str,
                            iteration_profile["overlap"]["overlap_ms"].get("compute_comm", 0.0),
                            iteration_profile["overlap"]["per_stream_ms"].get("compute", 0.0),
                        )

                    if torch_profiler is not None:
                        torch_profiler.step()

                    global_step += 1
                    if training_cfg.max_steps and global_step >= training_cfg.max_steps:
                        stop_flag["stop"] = True

                    if stop_flag["stop"]:
                        break

                if stop_flag["stop"]:
                    break

    metrics_logger.close()


def configure_optimizer(model: nn.Module, cfg: OptimizerConfig, dist_mode: str = "ddp") -> torch.optim.Optimizer:
    if cfg.name.lower() == "shampoo":
        from distributed_shampoo import DistributedShampoo, DDPDistributedConfig

        # Configure distributed Shampoo for multi-GPU training
        distributed_config = DDPDistributedConfig(
            communication_dtype=torch.float32,
            num_trainers_per_group=-1,  # Use all ranks in the group
            communicate_params=False,
        )
        log.info("Using DistributedShampoo optimizer with DDPDistributedConfig")
        optimizer = DistributedShampoo(
            model.parameters(),
            lr=cfg.lr,
            betas=cfg.betas,
            epsilon=cfg.eps,
            weight_decay=cfg.weight_decay,
            distributed_config=distributed_config,
        )
    else:
        log.info("Using AdamW optimizer")
        optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=cfg.betas, eps=cfg.eps)
    return optimizer


def configure_scheduler(optimizer: torch.optim.Optimizer, cfg: SchedulerConfig, total_steps: int):
    if cfg.total_steps <= 0:
        return None

    def lr_lambda(step: int) -> float:
        if step < cfg.warmup_steps:
            return float(step + 1) / float(max(1, cfg.warmup_steps))
        progress = min(1.0, float(step - cfg.warmup_steps) / max(1, cfg.total_steps - cfg.warmup_steps))
        return max(0.0, 1.0 - progress)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _maybe_compile(module: FSDP, cfg: CompileConfig) -> FSDP:
    compile_fn = getattr(torch, "compile", None)
    if not cfg.enabled:
        return module
    if compile_fn is None:  # pragma: no cover - defensive when torch.compile missing
        log.warning("torch.compile not available in this PyTorch build; skipping compilation")
        return module

    kwargs: Dict[str, Any] = {}
    backend = cfg.backend or "inductor"
    accelerator = detect_accelerator()
#    if accelerator == "amd" and backend in {"inductor", "inductor_dynamic"}:
#        log.warning("torch.compile backend '%s' is experimental on ROCm; disabling compilation", backend)
#        return module
    kwargs["backend"] = backend
    if cfg.mode:
        kwargs["mode"] = cfg.mode
    kwargs["fullgraph"] = cfg.fullgraph
    kwargs["dynamic"] = cfg.dynamic
    if cfg.options:
        kwargs["options"] = cfg.options

    try:
        compiled = compile_fn(module, **kwargs)
        log.info(
            "Enabled torch.compile | backend=%s mode=%s fullgraph=%s dynamic=%s",
            cfg.backend,
            cfg.mode,
            cfg.fullgraph,
            cfg.dynamic,
        )
        return compiled
    except Exception as exc:  # pragma: no cover - best effort
        log.warning("torch.compile failed (%s); continuing with eager module", exc, exc_info=True)
        return module


def _restore_rocm_profiler_env() -> None:
    keys = [
        "ROCPROFILER_LOG_LEVEL",
        "ROCPROFILER_ENABLE_TRACING",
        "ROCPROFILER_KERNEL_TIMESTAMPS",
        "ROCPROFILER_DEVICE_CLOCK_SYNC",
    ]
    for key in keys:
        if key in os.environ:
            os.environ.pop(key, None)


@contextlib.contextmanager
def _torch_profiler_context(
    cfg: ProfilerConfig, output_dir: Path, rank: int, device: torch.device
) -> Generator[Optional[profile], None, None]:
    if not cfg.enabled:
        yield None
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    rank_dir = output_dir / f"rank{rank}"
    rank_dir.mkdir(parents=True, exist_ok=True)

    if detect_accelerator() == "amd":
        _restore_rocm_profiler_env()

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    schedule_obj = schedule(wait=cfg.wait, warmup=cfg.warmup, active=cfg.active, repeat=cfg.repeat)

    log.info(
        "Enabling torch profiler | rank=%s wait=%s warmup=%s active=%s repeat=%s",
        rank,
        cfg.wait,
        cfg.warmup,
        cfg.active,
        cfg.repeat,
    )

    prof = profile(
        activities=activities,
        schedule=schedule_obj,
        record_shapes=cfg.record_shapes,
        profile_memory=cfg.profile_memory,
        with_stack=cfg.with_stack,
        with_flops=cfg.with_flops,
    )

    try:
        prof.__enter__()
        yield prof
    finally:
        prof.__exit__(None, None, None)
        produce_tb = cfg.tensorboard
        produce_chrome = cfg.chrome_trace
        try:
            stats_available = prof._stats() is not None  # type: ignore[attr-defined]
        except Exception:
            stats_available = False

        if produce_tb and stats_available:
            try:
                handler = tensorboard_trace_handler(str(rank_dir))
                handler(prof)
            except Exception as exc:  # pragma: no cover - best effort
                log.warning("TensorBoard trace export failed: %s", exc, exc_info=True)

        if produce_chrome and stats_available:
            stem, ext = os.path.splitext(cfg.trace_filename)
            if not ext:
                ext = ".json"
            trace_name = f"{stem}{ext}"
            if cfg.repeat != 1 or cfg.active > 1:
                trace_name = f"{stem}_step{prof.step_num}{ext}"
            try:
                prof.export_chrome_trace(str(rank_dir / trace_name))
            except Exception as exc:  # pragma: no cover - best effort
                log.warning("Chrome trace export failed: %s", exc, exc_info=True)


def seed_everything(seed: int) -> None:
    """Set all random seeds and enable deterministic mode for reproducibility.

    Covers: Python stdlib, NumPy global RNG, PyTorch CPU & all CUDA/ROCm
    devices, cuDNN/MIOpen algorithm selection, and PyTorch op determinism.

    Must be called before model construction, dataloader creation, and any
    operation that consumes the global RNG (weight init, dropout, etc.).
    """
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed also seeds all CUDA/ROCm devices (since PyTorch 1.x).
    torch.manual_seed(seed)

    # Disable autotuning so the same algorithms are chosen across runs.
    # On NVIDIA this controls cuDNN; on ROCm PyTorch keeps these attributes
    # and routes them to MIOpen where applicable.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # CUBLAS_WORKSPACE_CONFIG is NVIDIA-only; on ROCm deterministic dispatch
    # is handled via torch.use_deterministic_algorithms.
    is_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None
    if not is_rocm:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    # warn_only=True lets training continue when an op has no deterministic
    # implementation (common on ROCm) while still logging a warning.
    # NOTE: Disabled on ROCm — Flash Attention backward has no deterministic
    # implementation and can produce NaN when this flag interacts with hipBLAS.
    if not is_rocm:
        torch.use_deterministic_algorithms(True, warn_only=True)

    hash_seed = os.environ.get("PYTHONHASHSEED")
    if hash_seed is None or hash_seed == "random":
        log.warning(
            "PYTHONHASHSEED is not fixed (%s). Set PYTHONHASHSEED=%d before "
            "launching Python for fully reproducible dict/set iteration order.",
            hash_seed, seed,
        )

    log.info(
        "Seeded all RNGs with seed=%d | deterministic_algorithms=True "
        "(warn_only) | cudnn.deterministic=True | cudnn.benchmark=False | "
        "platform=%s",
        seed, "ROCm" if is_rocm else "CUDA",
    )


def _worker_init_fn(worker_id: int) -> None:
    """Seed each DataLoader worker for reproducibility when num_workers > 0."""
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def configure_tf32_precision(precision_cfg: PrecisionConfig) -> str:
    """Apply TF32 precision settings from *precision_cfg* to the process.

    Configures three levels of TF32 control:

    1. ``torch.set_float32_matmul_precision`` — the documented PyTorch API.
    2. ``torch.backends.cuda.matmul.allow_tf32`` — low-level flag (set
       implicitly by the API above, but also set explicitly for clarity).
    3. ``HIPBLASLT_OVERRIDE_COMPUTE_TYPE_XF32`` env var — on AMD/ROCm,
       PyTorch always passes ``HIPBLAS_COMPUTE_32F_FAST_TF32`` (xf32) to
       hipBLASLt when ``allow_tf32=True``.  This env var overrides the
       compute type *inside* hipBLASLt to select the actual accumulation
       strategy.

    Hardware notes (AMD):

    * **gfx942 (MI300X/MI308)** — native TF32 (truncates fp32 mantissa to
      10 bits).  ``x1`` uses a single TF32 accumulation pass.
    * **gfx950 (MI355)** — no native TF32; ``x1`` uses a single BF16 matmul,
      ``x3`` uses three BF16 matmuls (BF16x3) for higher accuracy.

    ========  ======================================  ==========================  ================================
    tf32_mode HIPBLASLT_OVERRIDE_COMPUTE_TYPE_XF32    float32_matmul_precision    hipBLAS compute type
    ========  ======================================  ==========================  ================================
    disabled  (unset)                                 highest                     FP32 GEMM
    x1        2                                       high                        HIPBLAS_COMPUTE_32F_FAST_16BF
    x3        1 (default / unset)                     high                        HIPBLAS_COMPUTE_32F_FAST_TF32
    ========  ======================================  ==========================  ================================

    Returns:
        The active tf32_mode string (lower-cased, validated).
    """
    tf32_mode = precision_cfg.tf32_mode.lower()
    if tf32_mode not in _VALID_TF32_MODES:
        log.warning(
            "Unknown tf32_mode '%s', falling back to 'disabled'. Valid: %s",
            tf32_mode, sorted(_VALID_TF32_MODES),
        )
        tf32_mode = "disabled"

    is_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None

    if tf32_mode == "disabled":
        torch.set_float32_matmul_precision("highest")
        os.environ.pop("HIPBLASLT_OVERRIDE_COMPUTE_TYPE_XF32", None)
    else:
        torch.set_float32_matmul_precision("high")
        if is_rocm:
            # PyTorch sends COMPUTE_32XF (FAST_TF32) to hipBLASLt when allow_tf32=True.
            # Override=2 downgrades to FAST_16BF (single BF16 acc, x1).
            # Override=1 keeps FAST_TF32 (triple BF16 acc, x3).
            override_values = {"x1": "2", "x3": "1"}
            os.environ["HIPBLASLT_OVERRIDE_COMPUTE_TYPE_XF32"] = override_values[tf32_mode]
        else:
            os.environ.pop("HIPBLASLT_OVERRIDE_COMPUTE_TYPE_XF32", None)

    log.info(
        "TF32 precision | tf32_mode=%s allow_tf32=%s "
        "HIPBLASLT_OVERRIDE_COMPUTE_TYPE_XF32=%s "
        "float32_matmul_precision=%s platform=%s",
        tf32_mode,
        torch.backends.cuda.matmul.allow_tf32,
        os.environ.get("HIPBLASLT_OVERRIDE_COMPUTE_TYPE_XF32", "unset"),
        torch.get_float32_matmul_precision(),
        "ROCm" if is_rocm else "CUDA",
    )

    param_dtype = _resolve_dtype(precision_cfg.param_dtype)
    if tf32_mode != "disabled" and param_dtype != torch.float32:
        log.warning(
            "param_dtype=%s with tf32_mode=%s: TF32 only affects fp32 matmuls outside "
            "the mixed-precision forward/backward (e.g. Shampoo optimizer matmuls).",
            precision_cfg.param_dtype, tf32_mode,
        )

    return tf32_mode


def _run_matmul_with_tf32(
    A: torch.Tensor,
    B: torch.Tensor,
    override_xf32: Optional[str],
    device: torch.device,
) -> torch.Tensor:
    """Run a single matmul with specific TF32 settings.

    *override_xf32* maps to ``HIPBLASLT_OVERRIDE_COMPUTE_TYPE_XF32``:
    ``None`` → TF32 disabled (pure fp32), ``"1"`` → keep FAST_TF32 (x3),
    ``"2"`` → downgrade to FAST_16BF (x1).
    """
    allow = override_xf32 is not None
    torch.backends.cuda.matmul.allow_tf32 = allow
    if allow:
        torch.set_float32_matmul_precision("high")
        os.environ["HIPBLASLT_OVERRIDE_COMPUTE_TYPE_XF32"] = override_xf32
    else:
        torch.set_float32_matmul_precision("highest")
        os.environ.pop("HIPBLASLT_OVERRIDE_COMPUTE_TYPE_XF32", None)
    torch.cuda.synchronize(device)
    result = torch.mm(A, B)
    torch.cuda.synchronize(device)
    return result


def verify_tf32_active(device: torch.device, tf32_mode: str) -> None:
    """Verify TF32 precision with a matmul probe.

    Runs fp32 matmuls under three settings — fp32 (disabled), x1
    (``HIPBLASLT_OVERRIDE_COMPUTE_TYPE_XF32=2``, FAST_16BF), and x3
    (``HIPBLASLT_OVERRIDE_COMPUTE_TYPE_XF32=1``, FAST_TF32) — then
    reports:

    1. Whether the *configured* mode is active (differs from fp32).
    2. Accuracy of each mode vs fp64 ground truth.
    3. **x1 vs x3 delta** — confirms different hipBLASLt compute passes.

    All ranks execute the probe independently (local matmul, no collectives).
    Only rank 0 emits detailed log messages.
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    is_primary = rank == 0

    if is_primary:
        log.info(
            "[PRECISION PROBE] tf32_mode=%s allow_tf32=%s "
            "HIPBLASLT_OVERRIDE_COMPUTE_TYPE_XF32=%s "
            "float32_matmul_precision=%s — running matmul verification...",
            tf32_mode,
            torch.backends.cuda.matmul.allow_tf32,
            os.environ.get("HIPBLASLT_OVERRIDE_COMPUTE_TYPE_XF32", "unset"),
            torch.get_float32_matmul_precision(),
        )

    saved_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    saved_override = os.environ.get("HIPBLASLT_OVERRIDE_COMPUTE_TYPE_XF32")
    saved_precision = torch.get_float32_matmul_precision()
    rng_state = torch.cuda.get_rng_state(device)

    try:
        torch.cuda.manual_seed(98765)

        M, K, N = 4096, 4096, 4096
        A = torch.randn(M, K, device=device, dtype=torch.float32)
        B = torch.randn(K, N, device=device, dtype=torch.float32)

        # fp64 ground truth
        ref_f64 = torch.mm(A.double(), B.double())

        # fp32 baseline (TF32 disabled)
        out_fp32 = _run_matmul_with_tf32(A, B, None, device)

        # x1: Override=2 → HIPBLAS_COMPUTE_32F_FAST_16BF (single BF16 acc)
        out_x1 = _run_matmul_with_tf32(A, B, "2", device)

        # x3: Override=1 → HIPBLAS_COMPUTE_32F_FAST_TF32 (triple BF16 acc)
        out_x3 = _run_matmul_with_tf32(A, B, "1", device)

        # --- Errors vs fp64 ground truth ---
        fp32_err = (out_fp32.double() - ref_f64).abs()
        x1_err = (out_x1.double() - ref_f64).abs()
        x3_err = (out_x3.double() - ref_f64).abs()

        # --- x1 vs x3 delta (the client-relevant comparison) ---
        x1_x3_diff = (out_x1 - out_x3).abs()

        # --- Configured mode vs fp32 (activation check) ---
        configured_map = {"disabled": out_fp32, "x1": out_x1, "x3": out_x3}
        out_configured = configured_map[tf32_mode]
        vs_fp32_diff = (out_fp32 - out_configured).abs().max().item()
        results_identical = vs_fp32_diff == 0.0

        if tf32_mode == "disabled":
            if results_identical:
                if is_primary:
                    log.info(
                        "[PRECISION PROBE] PASS  tf32_mode=disabled — matmul output is "
                        "bit-identical to fp32 reference. TF32 is correctly OFF."
                    )
            else:
                log.error(
                    "[PRECISION PROBE] FAIL  tf32_mode=disabled but matmul output DIFFERS "
                    "from fp32 reference (max_diff=%.4e). TF32 may be unexpectedly active!",
                    vs_fp32_diff,
                )
        else:
            if not results_identical:
                if is_primary:
                    log.info(
                        "[PRECISION PROBE] PASS  tf32_mode=%s — matmul output differs from "
                        "fp32 reference (max_diff=%.4e). TF32 is confirmed ACTIVE.",
                        tf32_mode, vs_fp32_diff,
                    )
            else:
                log.warning(
                    "[PRECISION PROBE] WARN  tf32_mode=%s but matmul output is bit-identical "
                    "to fp32 reference. The runtime may not support TF32.",
                    tf32_mode,
                )

        if is_primary:
            log.info(
                "[PRECISION PROBE] Accuracy vs fp64 (max_err) — "
                "fp32=%.4e, x1=%.4e (%.2fx), x3=%.4e (%.2fx)",
                fp32_err.max().item(),
                x1_err.max().item(),
                x1_err.max().item() / fp32_err.max().item() if fp32_err.max().item() > 0 else float("inf"),
                x3_err.max().item(),
                x3_err.max().item() / fp32_err.max().item() if fp32_err.max().item() > 0 else float("inf"),
            )
            log.info(
                "[PRECISION PROBE] x1 vs x3 delta — max=%.4e, mean=%.4e, "
                "identical=%s",
                x1_x3_diff.max().item(),
                x1_x3_diff.mean().item(),
                x1_x3_diff.max().item() == 0.0,
            )

        del A, B, ref_f64, out_fp32, out_x1, out_x3
        del fp32_err, x1_err, x3_err, x1_x3_diff
        torch.cuda.synchronize(device)

    except Exception as exc:
        log.warning(
            "[PRECISION PROBE] Probe failed (%s); precision flags were set but "
            "could not be verified at runtime.",
            exc, exc_info=True,
        )
    finally:
        torch.cuda.set_rng_state(rng_state, device)
        torch.backends.cuda.matmul.allow_tf32 = saved_allow_tf32
        torch.set_float32_matmul_precision(saved_precision)
        if saved_override is not None:
            os.environ["HIPBLASLT_OVERRIDE_COMPUTE_TYPE_XF32"] = saved_override
        else:
            os.environ.pop("HIPBLASLT_OVERRIDE_COMPUTE_TYPE_XF32", None)


def main_cli() -> None:  # pragma: no cover - CLI entry
    parser = argparse.ArgumentParser(description="FSDP2 training benchmark with multi-stream profiling")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML/JSON config file")
    parser.add_argument(
        "--override",
        type=str,
        nargs="*",
        default=None,
        help="Configuration overrides as dotted key=value entries",
    )
    parser.add_argument("--enable-rocm-metrics", action="store_true", help="Collect rocm-smi metrics")
    args = parser.parse_args()
    main(args, enable_rocm_metrics=args.enable_rocm_metrics)


def main(args: Optional[argparse.Namespace] = None, *, enable_rocm_metrics: bool = False) -> None:
    if args is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", type=str, required=True)
        parser.add_argument("--override", type=str, nargs="*", default=None)
        parsed = parser.parse_args()
    else:
        parsed = args

    config = _parse_config(parsed)
    training_cfg = _build_training_config(config)
    precision_cfg = _build_precision_config(config)
    optimizer_cfg = _build_optimizer_config(config)
    scheduler_cfg = _build_scheduler_config(config)
    model_cfg = _build_model_config(config)
    dataset_cfg = _build_dataset_config(config)
    loss_cfg = _build_loss_config(config)
    fsdp_cfg = _build_fsdp_config(config)
    ddp_cfg = _build_ddp_config(config)
    compile_cfg = _build_compile_config(config)
    profiler_cfg = _build_profiler_config(config)

    seed_everything(dataset_cfg.seed)

    tf32_mode = configure_tf32_precision(precision_cfg)

    log_level = config.get("logging", {}).get("level", "INFO")
    env = init_distributed(training_cfg, log_level)
    rank = env["rank"]

    verify_tf32_active(env["device"], tf32_mode)

    dataloader = create_dataloader(
        dataset_cfg,
        batch_size=training_cfg.batch_size,
        world_size=env["world_size"],
        rank=rank,
        num_workers=config.get("dataloader", {}).get("num_workers", 4),
        pin_memory=config.get("dataloader", {}).get("pin_memory", True),
        worker_init_fn=_worker_init_fn,
    )

    dist_mode = config.get("distributed", {}).get("mode")
    if dist_mode is None:
        dist_mode = "fsdp"
    dist_mode = dist_mode.lower()

    # For FSDP, FSDP MixedPrecision handles dtype casting (no autocast needed).
    # For DDP, we fall back to torch.autocast in the training loop.
    use_fsdp = dist_mode != "ddp"
    mp_policy = build_fsdp_mixed_precision(precision_cfg) if use_fsdp else None

    if dist_mode == "ddp":
        model = build_ddp_model(model_cfg, ddp_cfg, compile_cfg, env["device"])
    else:
        model = build_fsdp_model(model_cfg, fsdp_cfg, compile_cfg, env["device"], mp_policy)
    optimizer = configure_optimizer(model, optimizer_cfg, dist_mode)
    scheduler = configure_scheduler(
        optimizer,
        scheduler_cfg,
        training_cfg.max_steps or training_cfg.epochs * len(dataloader),
    )
    criterion = build_loss_criterion(loss_cfg)

    profiler = StreamProfiler(env["device"])

    # Training warmup: run a few forward/backward/optimizer steps to warm up collectives
    if not fsdp_cfg.skip_training_warmup and fsdp_cfg.training_warmup_steps > 0:
        log.info("Starting training warmup with %d steps...", fsdp_cfg.training_warmup_steps)
        param_dtype = _resolve_dtype(precision_cfg.param_dtype)
        warmup_scaler = torch.cuda.amp.GradScaler() if param_dtype == torch.float16 else None
        # Only use autocast for DDP; FSDP MixedPrecision handles casting
        warmup_autocast_dtype = (
            param_dtype if not use_fsdp and param_dtype != torch.float32 else None
        )

        warmup_training_collectives(
            model=model,
            optimizer=optimizer,
            dataloader=dataloader,
            device=env["device"],
            autocast_dtype=warmup_autocast_dtype,
            scaler=warmup_scaler,
            loss_fn=compute_loss,
            num_warmup_steps=fsdp_cfg.training_warmup_steps,
        )
        log.info("Training warmup complete")

    try:
        training_loop(
            model,
            optimizer,
            dataloader,
            training_cfg,
            precision_cfg,
            scheduler,
            env,
            profiler,
            enable_rocm_metrics,
            profiler_cfg,
            use_autocast=not use_fsdp,
            criterion=criterion,
        )
    finally:
        if dist.is_initialized():
            try:
                dist.barrier()
            except Exception as e:
                log.warning("Barrier failed during cleanup: %s", e)
            try:
                dist.destroy_process_group()
            except Exception as e:
                log.warning("destroy_process_group failed: %s", e)


__all__ = ["main", "main_cli"]
