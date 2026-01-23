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
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, Optional, Tuple
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.profiler import ProfilerActivity, schedule, tensorboard_trace_handler, profile
from torch.distributed.fsdp import BackwardPrefetch, FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP
from datetime import timedelta
from aorta.data import SyntheticDatasetConfig, create_dataloader
from aorta.models import ModelConfig, RankingTransformerModel
from aorta.profiling.stream_profiler import StreamProfiler
from aorta.race import RaceConfig
from aorta.race.injectors import (
    setup_gpu_max_hw_queues,
    check_hw_queues_warning,
    log_race_config_status,
    inject_h2d_racing,
    inject_datadist_racing,
    inject_timing_skew,
    should_skip_h2d_sync,
    should_skip_datadist_sync,
    wait_pending_datadist_work,
    is_datadist_work_pending,
    check_nccl_async_behavior,
    is_h2d_still_in_flight,
    is_h2d_tensor_in_flight,
    check_loss_for_nan,
    check_gradients_for_nan,
    schedule_inflight_check,
    flush_inflight_checks,
)
from aorta.race.h2d_racing import move_batch_to_device, get_memcpy_stream
from aorta.race.datadist_racing import get_datadist_stream
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
class TrainingConfig:
    epochs: int = 1
    batch_size: int = 8
    gradient_accumulation: int = 1
    max_steps: Optional[int] = None
    grad_clip_norm: float = 1.0
    mixed_precision: str = "bf16"  # options: none, fp16, bf16
    log_interval: int = 10
    output_dir: Path = Path("artifacts")
    inject_allreduce_copies: bool = False  # Inject all_reduce + host-device copies to trigger hang
    allreduce_stress_level: int = 1  # Number of all_reduce ops per iteration (1-10)
    # Race experiment settings are in RaceConfig (loaded from race_experiment: section)


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


def _build_race_config(raw: Dict[str, Any]) -> RaceConfig:
    """Build RaceConfig from the race_experiment section."""
    section = raw.get("race_experiment", {})
    cfg = RaceConfig()
    for field in dataclass_fields(RaceConfig):
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


def set_seed(seed: int, rank: int) -> None:
    """Set all random seeds for reproducibility across runs."""
    seed_value = seed + rank
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    log.info("Set random seed=%d for rank=%d (base_seed=%d)", seed_value, rank, seed)


def init_distributed(training_cfg: TrainingConfig, race_cfg: RaceConfig, log_level: str) -> Dict[str, Any]:
    # Set GPU_MAX_HW_QUEUES if configured (must be set BEFORE GPU init)
    # This controls hardware queue parallelism - 4+ is needed to expose race conditions
    setup_gpu_max_hw_queues(race_cfg)

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
    race_cfg: Optional[RaceConfig] = None,
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
            # Can be skipped with skip_rccl_warmup=True to test race conditions
            if race_cfg is not None and race_cfg.skip_rccl_warmup:
                log.warning("SKIPPING RCCL warmup (skip_rccl_warmup=True) - may cause hangs or race conditions")
            else:
                # Use race_cfg iterations if provided, otherwise fall back to fsdp_cfg
                rccl_iterations = race_cfg.rccl_warmup_iterations if race_cfg is not None else fsdp_cfg.rccl_warmup_iterations
                warmup_rccl_communicators(
                    shard_group,
                    replicate_group,
                    device,
                    num_warmup_ops=rccl_iterations,
                )
            log.info("Created custom process groups for HYBRID_SHARD strategy")

    # Ensure GPU operations are complete before FSDP wrapping
    # This helps prevent race conditions with inter-node communicators
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

    fsdp_model = FSDP(
        model.to(device),
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

    log.info("FSDP model wrapping complete")

    # Manual parameter sync for HYBRID_SHARD after FSDP wrapping
    if needs_manual_sync and replicate_group is not None:
        manual_sync_params(fsdp_model, replicate_group)
        # Extra synchronization after manual sync before proceeding
        torch.cuda.synchronize()
        dist.barrier()
        log.info("Post-sync barrier complete")

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
        "Creating HYBRID_SHARD process groups | rank=%d world_size=%d num_nodes=%d gpus_per_node=%d node_id=%d",
        rank, world_size, num_nodes, gpus_per_node, node_id
    )

    # Intra-node groups: shard within each node
    for i in range(num_nodes):
        ranks_in_node = list(range(i * gpus_per_node, (i + 1) * gpus_per_node))
        group = dist.new_group(ranks=ranks_in_node)
        if i == node_id:
            my_shard_group = group

    # Inter-node groups: replicate across nodes (same local_rank)
    for local_r in range(gpus_per_node):
        ranks_across_nodes = [node * gpus_per_node + local_r for node in range(num_nodes)]
        group = dist.new_group(ranks=ranks_across_nodes)
        if local_r == local_rank:
            my_replicate_group = group

    log.info(
        "Created process groups | shard_group_size=%d replicate_group_size=%d",
        dist.get_world_size(my_shard_group),
        dist.get_world_size(my_replicate_group),
    )

    # Note: We don't barrier here because the warmup function will handle synchronization.
    # Calling barrier here can trigger NCCL init race conditions before warmup runs.

    return (my_shard_group, my_replicate_group)


# NOTE: warmup functions moved to aorta.utils.warmup


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


# NOTE: move_batch_to_device and H2D racing functions moved to aorta.race.h2d_racing


def compute_loss(scores: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    target = batch["target"].to(scores.dtype)
    importance = batch["importance"].to(scores.dtype)
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


# ---------------------------------------------------------------------------
# Training loop helper functions (extracted for clarity and maintainability)
# ---------------------------------------------------------------------------


def _setup_mixed_precision(
    mode: str,
) -> Tuple[Optional[torch.dtype], Optional[torch.cuda.amp.GradScaler]]:
    """Configure mixed precision training (fp16/bf16/none).

    Args:
        mode: Mixed precision mode string ("fp16", "bf16", or "none").

    Returns:
        Tuple of (autocast_dtype, scaler). scaler is None for bf16/none.
    """
    mp_mode = mode.lower()
    if mp_mode == "fp16":
        return torch.float16, torch.cuda.amp.GradScaler()
    elif mp_mode == "bf16":
        return torch.bfloat16, None
    return None, None


def _inject_allreduce_stress(
    device: torch.device,
    loss: torch.Tensor,
    stress_level: int,
    profiler: "StreamProfiler",
    step: int,
    epoch: int,
) -> None:
    """Inject all_reduce + memcpy stress pattern for hang reproduction.

    Pattern: all_reduce -> device-to-device copy -> host-device copy -> compute blocked.
    This stresses the pattern that can cause rocprim deadlocks.

    Args:
        device: Target device.
        loss: Current loss tensor.
        stress_level: Number of stress cycles (1-10).
        profiler: StreamProfiler for range annotations.
        step: Current step in the epoch.
        epoch: Current epoch.
    """
    with profiler.range("aux", f"epoch{epoch}_step{step}_allreduce_sync"):
        # Perform multiple all_reduce + memory copy cycles
        stress_level = min(max(stress_level, 1), 10)

        for i in range(stress_level):
            # Create moderately-sized tensors to stress RCCL and memory copy
            # Size: ~4MB per tensor
            tensor_size = 1024 * 1024  # 1M elements = 4MB in FP32
            stress_tensor = torch.randn(tensor_size, device=device, dtype=torch.float32)

            # All-reduce operation (collective that triggers RCCL multi-stream)
            dist.all_reduce(stress_tensor, op=dist.ReduceOp.AVG)

            # Device-to-device copy (triggers hipMemcpyAsync device-to-device)
            device_copy_1 = stress_tensor.clone()
            device_copy_2 = device_copy_1.contiguous()

            # Force device-to-device copy via different tensor
            temp_storage = torch.empty_like(device_copy_2)
            temp_storage.copy_(device_copy_2, non_blocking=False)

            # All-reduce on device-copied tensor
            dist.all_reduce(temp_storage, op=dist.ReduceOp.SUM)

            # Force blocking host-device copy (triggers hipMemcpyWithStream)
            tensor_cpu = temp_storage.cpu()
            tensor_back = tensor_cpu.to(device, non_blocking=False)

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


def _log_iteration_metrics(
    profiler: "StreamProfiler",
    metrics_logger: MetricsLogger,
    loss: torch.Tensor,
    grad_norm: Optional[torch.Tensor],
    optimizer: torch.optim.Optimizer,
    training_cfg: TrainingConfig,
    iteration_profile: Dict[str, Any],
    step: int,
    epoch: int,
    global_step: int,
    rank: int,
    world_size: int,
    enable_rocm_metrics: bool,
) -> None:
    """Record iteration metrics and log to console/files.

    Args:
        profiler: StreamProfiler instance.
        metrics_logger: MetricsLogger instance.
        loss: Loss tensor from forward pass.
        grad_norm: Gradient norm (or None if clipping disabled).
        optimizer: The optimizer (for learning rate).
        training_cfg: Training configuration.
        iteration_profile: Profile data from profiler.end_iteration().
        step: Current step in the epoch.
        epoch: Current epoch.
        global_step: Global step counter.
        rank: Current rank.
        world_size: Total number of ranks.
        enable_rocm_metrics: Whether to collect ROCm metrics.
    """
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

    iteration_payload.update(collect_rocm_metrics(enable_rocm_metrics))
    metrics_logger.log(iteration_payload)

    # Write to loss log file
    loss_log = training_cfg.output_dir / f"loss_rank{rank}.log"
    with open(loss_log, "a") as f:
        f.write(
            f"step={global_step} epoch={epoch} loss={iteration_payload['loss']:.6f} "
            f"lr={iteration_payload['lr']:.6f}\n"
        )

    # Periodic console logging
    if global_step % training_cfg.log_interval == 0 and rank == 0:
        log.info(
            "epoch=%s step=%s loss=%.5f lr=%.6f overlap=%.3fms compute=%.3fms",
            epoch,
            step,
            iteration_payload["loss"],
            iteration_payload["lr"],
            iteration_profile["overlap"]["overlap_ms"].get("compute_comm", 0.0),
            iteration_profile["overlap"]["per_stream_ms"].get("compute", 0.0),
        )


def _finalize_training(
    metrics_logger: MetricsLogger,
    profiler: "StreamProfiler",
    training_cfg: TrainingConfig,
    race_cfg: RaceConfig,
    rank: int,
) -> None:
    """Close metrics logger and report NaN summary.

    Args:
        metrics_logger: MetricsLogger instance to close.
        profiler: StreamProfiler instance for NaN results.
        training_cfg: Training configuration.
        race_cfg: Race experiment configuration.
        rank: Current rank.
    """
    metrics_logger.close()

    # Report NaN check results if enabled
    if race_cfg.nan_check_collectives:
        nan_results = profiler.get_nan_check_results()
        if nan_results:
            log.warning(
                "[NaN SUMMARY] rank=%d detected %d NaN/Inf events during training",
                rank,
                len(nan_results),
            )
            # Write NaN results to file for analysis
            nan_log_path = training_cfg.output_dir / f"nan_check_rank{rank}.json"
            with open(nan_log_path, "w") as f:
                json.dump(nan_results, f, indent=2)
            log.info("NaN check results written to %s", nan_log_path)
        else:
            log.info("[NaN SUMMARY] rank=%d no NaN/Inf detected in RCCL collectives", rank)

    if rank == 0:
        log.info("Training loop finished. Profiler will export traces in cleanup phase.")
        log.info("Output directory: %s", training_cfg.output_dir)
        log.info("Torch profiler traces: %s", training_cfg.output_dir / "torch_profiler")
        log.info("Loss logs: %s/loss_rank*.log", training_cfg.output_dir)
        log.info("Metrics: %s/rank_*_metrics.jsonl", training_cfg.output_dir)


def training_loop(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader,
    training_cfg: TrainingConfig,
    race_cfg: RaceConfig,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    environment: Dict[str, Any],
    profiler: StreamProfiler,
    enable_rocm_metrics: bool,
    profiler_cfg: ProfilerConfig,
    warmup_dataloader=None,
) -> None:
    """Main training loop with profiling, race injection, and metrics logging.

    The loop structure keeps race injection timing visible:
    - H2D racing injection happens BEFORE forward pass
    - Datadist racing injection happens BEFORE forward pass
    - Timing skew injection happens BEFORE forward pass
    - Forward and backward passes are clearly separated
    - NaN checks happen after forward and after backward

    Args:
        model: The model to train.
        optimizer: The optimizer.
        dataloader: Training dataloader.
        training_cfg: Training configuration.
        race_cfg: Race experiment configuration.
        scheduler: Learning rate scheduler (optional).
        environment: Distributed environment info (rank, world_size, device).
        profiler: StreamProfiler for timing and tracing.
        enable_rocm_metrics: Whether to collect ROCm GPU metrics.
        profiler_cfg: Torch profiler configuration.
        warmup_dataloader: Optional separate dataloader for warmup (with smaller batch_size).
    """
    # =========================================================================
    # SETUP
    # =========================================================================
    rank = environment["rank"]
    world_size = environment["world_size"]
    device = environment["device"]

    autocast_dtype, scaler = _setup_mixed_precision(training_cfg.mixed_precision)
    metrics_logger = MetricsLogger(training_cfg.output_dir, rank)

    global_step = 0
    stop_flag = {"stop": False}
    setup_signal_handlers(stop_flag)

    model.train()

    # =========================================================================
    # WARMUP (timing visible - warmup reduces race likelihood)
    # =========================================================================
    if race_cfg.skip_training_warmup:
        log.warning("SKIPPING training warmup (skip_training_warmup=True) - may increase race likelihood")
    else:
        # Use warmup_dataloader if provided (allows smaller batch size for faster warmup)
        warmup_dl = warmup_dataloader if warmup_dataloader is not None else dataloader
        warmup_batch_info = ""
        if warmup_dataloader is not None:
            warmup_batch_info = f", warmup_batch_size={race_cfg.warmup_batch_size}"
        log.info(
            "Starting training warmup pass (rank=%d, num_steps=%d%s)...",
            rank, race_cfg.training_warmup_steps, warmup_batch_info
        )
        warmup_training_collectives(
            model, optimizer, warmup_dl, device, autocast_dtype, scaler,
            loss_fn=compute_loss, num_warmup_steps=race_cfg.training_warmup_steps,
        )
        log.info("Training warmup complete (rank=%d)", rank)

    # Enable NaN checking around RCCL collectives if configured
    if race_cfg.nan_check_collectives:
        profiler.enable_nan_checking(True)
        log.info("NaN checking enabled around RCCL collectives (rank=%d)", rank)

    check_hw_queues_warning(race_cfg)
    log_race_config_status(race_cfg, rank)

    # =========================================================================
    # MAIN TRAINING LOOP
    # =========================================================================
    profiler_dir = training_cfg.output_dir / "torch_profiler"

    # Client stream layout: bypass DistributedOpsInterceptor so FSDP collectives
    # run on the default stream (matches client's actual TorchRec architecture)
    if race_cfg.client_stream_layout:
        log.info("CLIENT STREAM LAYOUT: Bypassing DistributedOpsInterceptor - collectives on default stream")
        dist_ops_context = contextlib.nullcontext()
    else:
        dist_ops_context = profiler.intercept_distributed_ops()

    with dist_ops_context:
        with _torch_profiler_context(profiler_cfg, profiler_dir, rank, device) as torch_profiler:
            for epoch in range(training_cfg.epochs):
                if hasattr(dataloader.sampler, "set_epoch"):
                    dataloader.sampler.set_epoch(epoch)

                for step, cpu_batch in enumerate(dataloader):
                    profiler.start_iteration(global_step)
                    profiler.set_current_step(global_step)

                    # ---------------------------------------------------------
                    # H2D RACING INJECTION (before forward)
                    # ---------------------------------------------------------
                    # Initialize timing variables for debug logs
                    t_h2d_start = t_h2d_end = 0.0
                    t_datadist_start = t_datadist_end = 0.0
                    t_skew_start = t_skew_end = 0.0
                    skew_delay = 0

                    # GPU event-based timing (non-blocking, doesn't interfere with race)
                    gpu_events_enabled = race_cfg.gpu_event_timing
                    if gpu_events_enabled:
                        h2d_start_evt = torch.cuda.Event(enable_timing=True)
                        h2d_end_evt = torch.cuda.Event(enable_timing=True)
                        dd_start_evt = torch.cuda.Event(enable_timing=True)
                        dd_end_evt = torch.cuda.Event(enable_timing=True)
                        fwd_start_evt = torch.cuda.Event(enable_timing=True)
                        fwd_end_evt = torch.cuda.Event(enable_timing=True)

                    h2d_racing_active = race_cfg.h2d_memcpy_racing and step >= race_cfg.h2d_racing_start_step

                    if h2d_racing_active:
                        # Log first activation
                        if step == race_cfg.h2d_racing_start_step:
                            log.info(
                                "H2D RACING: ACTIVATED at step=%d rank=%d (skip_sync=%s)",
                                step, rank, race_cfg.h2d_skip_sync_before_forward,
                            )

                        if race_cfg.timing_debug_logs:
                            t_h2d_start = time.perf_counter()

                        # Get memcpy stream upfront for accurate GPU event recording
                        # Both start and end events must be on the SAME stream for accurate timing
                        memcpy_stream_for_events = get_memcpy_stream(device)

                        # Record GPU event before H2D (on memcpy_stream where H2D actually runs)
                        if gpu_events_enabled:
                            with torch.cuda.stream(memcpy_stream_for_events):
                                h2d_start_evt.record()

                        with profiler.range("aux", f"epoch{epoch}_step{step}_prefetch"):
                            batch, memcpy_stream = inject_h2d_racing(cpu_batch, device, step, race_cfg, rank)

                        # Record end event on memcpy_stream (same stream as start for accurate timing)
                        if gpu_events_enabled:
                            with torch.cuda.stream(memcpy_stream_for_events):
                                h2d_end_evt.record()

                        if race_cfg.timing_debug_logs:
                            t_h2d_end = time.perf_counter()

                        if should_skip_h2d_sync(step, race_cfg):
                            # RACE: skip memcpy sync - forward may read incomplete H2D data
                            log.info("H2D RACING: step=%d rank=%d - race window open", step, rank)
                            profiler.stream("compute").wait_stream(profiler.stream("aux"))
                        else:
                            if memcpy_stream is not None:
                                profiler.stream("compute").wait_stream(memcpy_stream)
                            profiler.stream("compute").wait_stream(profiler.stream("aux"))
                    else:
                        # Normal path (no H2D racing)
                        if race_cfg.timing_debug_logs:
                            t_h2d_start = time.perf_counter()

                        if gpu_events_enabled:
                            h2d_start_evt.record()

                        with profiler.range("aux", f"epoch{epoch}_step{step}_prefetch"):
                            batch = move_batch_to_device(cpu_batch, device)

                        if gpu_events_enabled:
                            h2d_end_evt.record()

                        if race_cfg.timing_debug_logs:
                            t_h2d_end = time.perf_counter()

                        profiler.stream("compute").wait_stream(profiler.stream("aux"))

                    if race_cfg.timing_debug_logs:
                        log.info(
                            "TIMING: step=%d rank=%d H2D_START=%.6f H2D_END=%.6f H2D_DUR=%.3fms",
                            step, rank, t_h2d_start, t_h2d_end, (t_h2d_end - t_h2d_start) * 1000
                        )

                    # ---------------------------------------------------------
                    # DATADIST RACING INJECTION (before forward)
                    # ---------------------------------------------------------
                    datadist_racing_active = race_cfg.datadist_racing and step >= race_cfg.datadist_racing_start_step

                    if datadist_racing_active:
                        # Log first activation
                        if step == race_cfg.datadist_racing_start_step:
                            log.info(
                                "DATADIST RACING: ACTIVATED at step=%d rank=%d (skip_sync=%s)",
                                step, rank, race_cfg.datadist_skip_sync_before_collective,
                            )

                        if race_cfg.timing_debug_logs:
                            t_datadist_start = time.perf_counter()

                        # Get datadist stream upfront for accurate GPU event recording
                        # Both start and end events must be on the SAME stream for accurate timing
                        datadist_stream_for_events = get_datadist_stream(device)

                        # Record GPU event before datadist (on datadist_stream where all_to_all runs)
                        if gpu_events_enabled:
                            with torch.cuda.stream(datadist_stream_for_events):
                                dd_start_evt.record()

                        batch, datadist_stream = inject_datadist_racing(batch, device, step, race_cfg, rank)

                        # Record end event on datadist_stream (same stream as start for accurate timing)
                        if gpu_events_enabled:
                            with torch.cuda.stream(datadist_stream_for_events):
                                dd_end_evt.record()

                        if race_cfg.timing_debug_logs:
                            t_datadist_end = time.perf_counter()

                        if should_skip_datadist_sync(step, race_cfg):
                            # RACE: skip datadist sync - FSDP collective may read incomplete data
                            log.info("DATADIST RACING: step=%d rank=%d - race window open", step, rank)
                        else:
                            if datadist_stream is not None:
                                torch.cuda.current_stream().wait_stream(datadist_stream)

                        if race_cfg.timing_debug_logs:
                            log.info(
                                "TIMING: step=%d rank=%d DATADIST_START=%.6f DATADIST_END=%.6f DATADIST_DUR=%.3fms",
                                step, rank, t_datadist_start, t_datadist_end, (t_datadist_end - t_datadist_start) * 1000
                            )

                        # ---------------------------------------------------------
                        # NCCL ASYNC DIAGNOSTIC (non-blocking check)
                        # ---------------------------------------------------------
                        # Check if the all_to_all work is still in-flight BEFORE forward starts.
                        # This verifies that the race window is actually open.
                        # If work is already completed, NCCL may have internal sync that masks the race.
                        if race_cfg.nccl_async_diagnostic:
                            is_async, nccl_diag_msg = check_nccl_async_behavior(step, rank)
                            if is_async:
                                log.info(
                                    "NCCL_DIAG: step=%d rank=%d all_to_all status=%s (race window OPEN)",
                                    step, rank, nccl_diag_msg
                                )
                            else:
                                log.warning(
                                    "NCCL_DIAG: step=%d rank=%d all_to_all status=%s (race window may be CLOSED!)",
                                    step, rank, nccl_diag_msg
                                )

                    # ---------------------------------------------------------
                    # TIMING SKEW INJECTION (before forward)
                    # ---------------------------------------------------------
                    if race_cfg.is_timing_skew_active(step):
                        if race_cfg.timing_debug_logs:
                            t_skew_start = time.perf_counter()

                        skew_delay = inject_timing_skew(step, rank, race_cfg)

                        if race_cfg.timing_debug_logs:
                            t_skew_end = time.perf_counter()
                            log.info(
                                "TIMING: step=%d rank=%d SKEW_START=%.6f SKEW_END=%.6f SKEW_DUR=%.3fms SKEW_US=%d",
                                step, rank, t_skew_start, t_skew_end, (t_skew_end - t_skew_start) * 1000, skew_delay
                            )

                        if skew_delay > 0:
                            log.debug("TIMING SKEW: rank=%d step=%d delay=%.0f us", rank, step, skew_delay)

                    # ---------------------------------------------------------
                    # FORWARD PASS
                    # ---------------------------------------------------------
                    optimizer.zero_grad(set_to_none=True)

                    # ---------------------------------------------------------
                    # H2D RACE WINDOW CHECK (right before forward)
                    # ---------------------------------------------------------
                    # Check if H2D copy is still in-flight when forward starts.
                    # If yes, forward will read incomplete/torn data from batch tensors.
                    #
                    # IMPORTANT: We use is_h2d_still_in_flight() which calls event.query()
                    # This is a NON-BLOCKING polling query that doesn't sync CUDA.
                    # We do NOT check for NaN/Inf here as .item() would sync and mask the race!
                    if race_cfg.nccl_async_diagnostic and h2d_racing_active:
                        if race_cfg.h2d_split_dense_copy:
                            h2d_in_flight = is_h2d_tensor_in_flight("dense")
                            h2d_label = "H2D dense copy"
                        else:
                            h2d_in_flight = is_h2d_still_in_flight()
                            h2d_label = "H2D copy"
                        if h2d_in_flight:
                            log.info(
                                "H2D_RACE: step=%d rank=%d %s STILL IN-FLIGHT when forward starts - RACE CONFIRMED!",
                                step, rank, h2d_label
                            )
                            # NOTE: We do NOT call check_batch_for_corruption() here!
                            # That would use .item() which syncs CUDA and masks the race.
                            # NaN will be detected in the loss check after forward.

                            # Schedule repeated in-flight reads to detect instability
                            if (race_cfg.inflight_read_check_enabled and
                                race_cfg.inflight_read_repeats > 0 and
                                "dense" in batch and isinstance(batch["dense"], torch.Tensor)):
                                dense = batch["dense"]
                                # Compute tail offset using same math as h2d_racing.py
                                total = dense.numel()
                                tail_frac = max(0.0, min(1.0, race_cfg.h2d_dense_tail_fraction))
                                split_idx = int(total * (1.0 - tail_frac))
                                split_idx = max(1, min(total - 1, split_idx))
                                # Get tail region
                                tail_tensor = dense.reshape(-1)[split_idx:]
                                schedule_inflight_check(
                                    name="h2d_dense",
                                    tensor=tail_tensor,
                                    sample_size=race_cfg.inflight_read_sample_size,
                                    repeats=race_cfg.inflight_read_repeats,
                                    step=step,
                                    rank=rank,
                                )
                        else:
                            log.warning(
                                "H2D_RACE: step=%d rank=%d %s ALREADY COMPLETED before forward - NO RACE!",
                                step, rank, h2d_label
                            )

                    # ---------------------------------------------------------
                    # NCCL RACE WINDOW CHECK (right before forward)
                    # ---------------------------------------------------------
                    # This is the critical check: Is all_to_all still running when forward starts?
                    # If yes, we have a true race condition. If no, NCCL completed before forward.
                    if race_cfg.nccl_async_diagnostic and datadist_racing_active:
                        work_pending = is_datadist_work_pending()
                        if work_pending:
                            log.info(
                                "NCCL_RACE: step=%d rank=%d all_to_all STILL PENDING when forward starts - RACE CONFIRMED!",
                                step, rank
                            )
                        else:
                            log.warning(
                                "NCCL_RACE: step=%d rank=%d all_to_all ALREADY COMPLETED before forward - NO RACE!",
                                step, rank
                            )

                    if race_cfg.timing_debug_logs:
                        t_forward_start = time.perf_counter()

                    # Record GPU event before forward (on default stream)
                    if gpu_events_enabled:
                        fwd_start_evt.record()

                    if race_cfg.client_stream_layout:
                        # Client mode: forward on default stream (no profiler range)
                        if autocast_dtype:
                            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                                scores = model(batch)
                                loss = compute_loss(scores, batch)
                        else:
                            scores = model(batch)
                            loss = compute_loss(scores, batch)
                    else:
                        # Normal mode: forward on compute stream via profiler
                        with profiler.range("compute", f"epoch{epoch}_step{step}_forward"):
                            if autocast_dtype:
                                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                                    scores = model(batch)
                                    loss = compute_loss(scores, batch)
                            else:
                                scores = model(batch)
                                loss = compute_loss(scores, batch)

                    # Record GPU event after forward (on default/compute stream)
                    if gpu_events_enabled:
                        fwd_end_evt.record()

                    if race_cfg.timing_debug_logs:
                        t_forward_end = time.perf_counter()
                        log.info(
                            "TIMING: step=%d rank=%d FORWARD_START=%.6f FORWARD_END=%.6f FORWARD_DUR=%.3fms",
                            step, rank, t_forward_start, t_forward_end, (t_forward_end - t_forward_start) * 1000
                        )

                        # Calculate gaps (negative = overlap = race window)
                        gap_h2d_to_forward = (t_forward_start - t_h2d_end) * 1000
                        gap_datadist_to_forward = (t_forward_start - t_datadist_end) * 1000 if t_datadist_end > 0 else 0.0
                        gap_skew_to_forward = (t_forward_start - t_skew_end) * 1000 if t_skew_end > 0 else 0.0

                        log.info(
                            "TIMING: step=%d rank=%d GAP_H2D_TO_FORWARD=%.3fms GAP_DATADIST_TO_FORWARD=%.3fms GAP_SKEW_TO_FORWARD=%.3fms",
                            step, rank, gap_h2d_to_forward, gap_datadist_to_forward, gap_skew_to_forward
                        )

                    # NaN check after forward
                    if race_cfg.nan_check_collectives:
                        if check_loss_for_nan(loss, step, rank):
                            log.warning("NaN detected in loss at step=%d rank=%d", step, rank)

                    # ---------------------------------------------------------
                    # BACKWARD PASS
                    # ---------------------------------------------------------
                    if race_cfg.client_stream_layout:
                        # Client mode: backward on default stream (FSDP collectives also here)
                        if scaler is not None:
                            scaler.scale(loss).backward()
                        else:
                            loss.backward()
                    else:
                        # Normal mode: backward on compute stream via profiler
                        with profiler.range("compute", f"epoch{epoch}_step{step}_backward"):
                            if scaler is not None:
                                scaler.scale(loss).backward()
                            else:
                                loss.backward()

                    # NaN check after backward
                    if race_cfg.nan_check_collectives:
                        has_nan_grads, nan_count = check_gradients_for_nan(model, step, rank)
                        if has_nan_grads:
                            log.warning("NaN detected in gradients at step=%d rank=%d count=%d", step, rank, nan_count)

                    # Stream sync: only needed in normal mode (client mode uses default stream)
                    if not race_cfg.client_stream_layout:
                        profiler.stream("aux").wait_stream(profiler.stream("compute"))

                    # ---------------------------------------------------------
                    # GRADIENT CLIPPING
                    # ---------------------------------------------------------
                    grad_norm = None
                    if training_cfg.grad_clip_norm is not None and training_cfg.grad_clip_norm > 0:
                        if race_cfg.client_stream_layout:
                            # Client mode: clip on default stream
                            grad_norm = clip_grad_norm_(model.parameters(), training_cfg.grad_clip_norm)
                        else:
                            # Normal mode: clip on aux stream via profiler
                            with profiler.range("aux", f"epoch{epoch}_step{step}_grad_clip"):
                                grad_norm = clip_grad_norm_(model.parameters(), training_cfg.grad_clip_norm)

                    # ---------------------------------------------------------
                    # OPTIMIZER STEP
                    # ---------------------------------------------------------
                    if race_cfg.client_stream_layout:
                        # Client mode: optimizer on default stream
                        try:
                            if scaler is not None:
                                scaler.step(optimizer)
                                scaler.update()
                            else:
                                optimizer.step()
                        except AssertionError as e:
                            if "NaN" in str(e) or "Inf" in str(e):
                                log.error("NaN/Inf detected in rank %d at step %d: %s", rank, global_step, e)
                                log.error("Stopping training to save traces")
                                stop_flag["stop"] = True
                                break
                            else:
                                raise

                        if scheduler is not None:
                            scheduler.step()
                    else:
                        # Normal mode: optimizer on aux stream via profiler
                        with profiler.range("aux", f"epoch{epoch}_step{step}_optimizer"):
                            try:
                                if scaler is not None:
                                    scaler.step(optimizer)
                                    scaler.update()
                                else:
                                    optimizer.step()
                            except AssertionError as e:
                                if "NaN" in str(e) or "Inf" in str(e):
                                    log.error("NaN/Inf detected in rank %d at step %d: %s", rank, global_step, e)
                                    log.error("Stopping training to save traces")
                                    stop_flag["stop"] = True
                                    break
                                else:
                                    raise

                            if scheduler is not None:
                                scheduler.step()

                    # ---------------------------------------------------------
                    # OPTIONAL ALLREDUCE STRESS TEST
                    # ---------------------------------------------------------
                    if training_cfg.inject_allreduce_copies and dist.is_initialized():
                        _inject_allreduce_stress(device, loss, training_cfg.allreduce_stress_level, profiler, step, epoch)

                    # ---------------------------------------------------------
                    # END ITERATION AND LOG METRICS
                    # ---------------------------------------------------------
                    # Wait for any pending async datadist work to complete before next step
                    # This prevents NCCL collective desync while preserving the race window
                    # during forward/backward (the race already happened by now)
                    if race_cfg.datadist_racing:
                        wait_pending_datadist_work()

                    # ---------------------------------------------------------
                    # FLUSH IN-FLIGHT INSTABILITY CHECKS
                    # ---------------------------------------------------------
                    # After race window closes, flush and log any detected mismatches
                    if race_cfg.inflight_read_check_enabled and race_cfg.inflight_read_repeats > 0:
                        flush_inflight_checks(step, rank)

                    # ---------------------------------------------------------
                    # GPU EVENT TIMING (calculated after iteration completes)
                    # ---------------------------------------------------------
                    # WARNING: Cross-stream timing (e.g., DD->FWD, H2D->FWD) is INACCURATE!
                    # Events on different streams measure enqueue time, not execution time.
                    # For accurate race detection, use nccl_async_diagnostic instead.
                    if gpu_events_enabled:
                        # Synchronize to ensure all GPU work is complete before reading events
                        torch.cuda.synchronize()

                        # Calculate GPU-side durations
                        # NOTE: Same-stream timing (H2D_DUR, FWD_DUR) is accurate.
                        # Cross-stream timing (H2D_TO_FWD, DD_TO_FWD) is approximate/unreliable.
                        try:
                            gpu_h2d_dur = h2d_start_evt.elapsed_time(h2d_end_evt)
                            gpu_fwd_dur = fwd_start_evt.elapsed_time(fwd_end_evt)

                            # CAUTION: This is cross-stream timing and may be inaccurate!
                            # The elapsed_time measures wall-clock between event recordings,
                            # but events on different streams can have arbitrary GPU execution order.
                            gpu_h2d_to_fwd = h2d_end_evt.elapsed_time(fwd_start_evt)

                            log.info(
                                "GPU_TIMING: step=%d rank=%d GPU_H2D_DUR=%.3fms GPU_FWD_DUR=%.3fms GPU_H2D_TO_FWD=%.3fms (CROSS-STREAM: may be inaccurate)",
                                step, rank, gpu_h2d_dur, gpu_fwd_dur, gpu_h2d_to_fwd
                            )

                            # Datadist timing (only if active)
                            if datadist_racing_active:
                                gpu_dd_dur = dd_start_evt.elapsed_time(dd_end_evt)
                                # CAUTION: dd_end_evt is recorded immediately after launching
                                # async all_to_all, so GPU_DD_DUR measures launch time, NOT
                                # actual collective duration. Use nccl_async_diagnostic for
                                # accurate race window detection.
                                gpu_dd_to_fwd = dd_end_evt.elapsed_time(fwd_start_evt)
                                log.info(
                                    "GPU_TIMING: step=%d rank=%d GPU_DD_DUR=%.3fms (launch only!) GPU_DD_TO_FWD=%.3fms (CROSS-STREAM: may be inaccurate)",
                                    step, rank, gpu_dd_dur, gpu_dd_to_fwd
                                )

                                # Summary line showing overlap status
                                # NOTE: These overlap/gap indicators are UNRELIABLE for cross-stream.
                                # Use NCCL_RACE logs from nccl_async_diagnostic for accurate detection.
                                h2d_overlap = "OVERLAP" if gpu_h2d_to_fwd < 0 else "GAP"
                                dd_overlap = "OVERLAP" if gpu_dd_to_fwd < 0 else "GAP"
                                log.info(
                                    "GPU_OVERLAP: step=%d rank=%d H2D->FWD=%s(%.3fms) DD->FWD=%s(%.3fms) [CAUTION: cross-stream timing unreliable, use nccl_async_diagnostic]",
                                    step, rank, h2d_overlap, gpu_h2d_to_fwd, dd_overlap, gpu_dd_to_fwd
                                )
                        except RuntimeError as e:
                            log.warning("GPU_TIMING: Failed to calculate event times: %s", e)

                    profiler.record_marker("compute", f"epoch{epoch}_step{step}_end")
                    iteration_profile = profiler.end_iteration()

                    _log_iteration_metrics(
                        profiler=profiler,
                        metrics_logger=metrics_logger,
                        loss=loss,
                        grad_norm=grad_norm,
                        optimizer=optimizer,
                        training_cfg=training_cfg,
                        iteration_profile=iteration_profile,
                        step=step,
                        epoch=epoch,
                        global_step=global_step,
                        rank=rank,
                        world_size=world_size,
                        enable_rocm_metrics=enable_rocm_metrics,
                    )

                    if torch_profiler is not None:
                        torch_profiler.step()

                    global_step += 1
                    if training_cfg.max_steps and global_step >= training_cfg.max_steps:
                        stop_flag["stop"] = True

                    if stop_flag["stop"]:
                        break

                if stop_flag["stop"]:
                    if rank == 0:
                        log.info("Training stopped at epoch=%d step=%d", epoch, step)
                    break

    # =========================================================================
    # FINALIZATION
    # =========================================================================
    _finalize_training(
        metrics_logger=metrics_logger,
        profiler=profiler,
        training_cfg=training_cfg,
        race_cfg=race_cfg,
        rank=rank,
    )


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
        
        if produce_tb:
            try:
                handler = tensorboard_trace_handler(str(rank_dir))
                handler(prof)
                log.info("Exported TensorBoard trace to %s", rank_dir)
            except Exception as exc:
                log.warning("TensorBoard trace export failed: %s", exc)

        if produce_chrome:
            stem, ext = os.path.splitext(cfg.trace_filename)
            if not ext:
                ext = ".json"
            trace_name = f"{stem}{ext}"
            if cfg.repeat != 1 or cfg.active > 1:
                trace_name = f"{stem}_step{prof.step_num}{ext}"
            try:
                prof.export_chrome_trace(str(rank_dir / trace_name))
                log.info("Exported chrome trace to %s/%s", rank_dir, trace_name)
            except Exception as exc:
                log.warning("Chrome trace export failed: %s", exc)


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
    race_cfg = _build_race_config(config)
    optimizer_cfg = _build_optimizer_config(config)
    scheduler_cfg = _build_scheduler_config(config)
    model_cfg = _build_model_config(config)
    dataset_cfg = _build_dataset_config(config)
    fsdp_cfg = _build_fsdp_config(config)
    ddp_cfg = _build_ddp_config(config)
    compile_cfg = _build_compile_config(config)
    profiler_cfg = _build_profiler_config(config)

    log_level = config.get("logging", {}).get("level", "INFO")
    env = init_distributed(training_cfg, race_cfg, log_level)
    rank = env["rank"]

    set_seed(dataset_cfg.seed, rank)

    dataloader = create_dataloader(
        dataset_cfg,
        batch_size=training_cfg.batch_size,
        world_size=env["world_size"],
        rank=rank,
        num_workers=config.get("dataloader", {}).get("num_workers", 4),
        pin_memory=config.get("dataloader", {}).get("pin_memory", True),
    )

    # Create separate warmup dataloader if warmup_batch_size is specified
    warmup_dataloader = None
    if race_cfg.warmup_batch_size is not None and not race_cfg.skip_training_warmup:
        warmup_dataloader = create_dataloader(
            dataset_cfg,
            batch_size=race_cfg.warmup_batch_size,
            world_size=env["world_size"],
            rank=rank,
            num_workers=0,  # Use fewer workers for warmup
            pin_memory=config.get("dataloader", {}).get("pin_memory", True),
        )
        log.info(
            "Created warmup dataloader with batch_size=%d (training batch_size=%d)",
            race_cfg.warmup_batch_size, training_cfg.batch_size
        )

    dist_mode = config.get("distributed", {}).get("mode")
    if dist_mode is None:
        dist_mode = "fsdp"
    dist_mode = dist_mode.lower()

    if dist_mode == "ddp":
        model = build_ddp_model(model_cfg, ddp_cfg, compile_cfg, env["device"])
    else:
        model = build_fsdp_model(model_cfg, fsdp_cfg, compile_cfg, env["device"], race_cfg)
    optimizer = configure_optimizer(model, optimizer_cfg, dist_mode)
    scheduler = configure_scheduler(
        optimizer,
        scheduler_cfg,
        training_cfg.max_steps or training_cfg.epochs * len(dataloader),
    )

    profiler = StreamProfiler(env["device"])

    try:
        training_loop(
            model,
            optimizer,
            dataloader,
            training_cfg,
            race_cfg,
            scheduler,
            env,
            profiler,
            enable_rocm_metrics,
            profiler_cfg,
            warmup_dataloader=warmup_dataloader,
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
