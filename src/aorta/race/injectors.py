"""
High-level injection interface for race condition experiments.

This module provides clean entry points for injecting race conditions
into the training loop. It wraps the lower-level modules and provides
a simple interface for the trainer to use.
"""

import logging
import os
from typing import Dict, Optional, Tuple

import torch

from aorta.race.config import RaceConfig
from aorta.race.h2d_racing import (
    get_memcpy_stream,
    move_batch_to_device_racing as _move_batch_racing,
    should_skip_h2d_sync,
    is_h2d_still_in_flight,
    is_h2d_tensor_in_flight,
    check_h2d_race_status,
)
from aorta.race.datadist_racing import (
    get_datadist_stream,
    inject_datadist_racing as _inject_datadist_racing,
    should_skip_datadist_sync,
    wait_pending_datadist_work,
    is_datadist_work_pending,
    check_nccl_async_behavior,
    get_pending_datadist_work,
)
from aorta.race.timing_skew_experiment import (
    inject_timing_skew_from_race_config,
    check_loss_for_nan,
    check_gradients_for_nan,
)


log = logging.getLogger(__name__)


# Re-export for convenience
__all__ = [
    "inject_h2d_racing",
    "inject_datadist_racing",
    "inject_timing_skew",
    "should_skip_h2d_sync",
    "should_skip_datadist_sync",
    "wait_pending_datadist_work",
    "is_datadist_work_pending",
    "check_nccl_async_behavior",
    "get_pending_datadist_work",
    "is_h2d_still_in_flight",
    "is_h2d_tensor_in_flight",
    "check_h2d_race_status",
    "get_memcpy_stream",
    "get_datadist_stream",
    "setup_gpu_max_hw_queues",
    "log_race_config_status",
    "check_loss_for_nan",
    "check_gradients_for_nan",
    "schedule_inflight_check",
    "flush_inflight_checks",
    "clear_inflight_checks",
]


def setup_gpu_max_hw_queues(race_cfg: RaceConfig) -> None:
    """
    Set GPU_MAX_HW_QUEUES environment variable if configured.

    CRITICAL: This must be called BEFORE any GPU initialization!

    GPU_MAX_HW_QUEUES controls hardware queue parallelism:
    - 1-2: Streams share HW queues → implicit serialization → RACE MASKED
    - 4+: Each stream gets own HW queue → true parallelism → RACE EXPOSED

    Args:
        race_cfg: Race configuration
    """
    if race_cfg.gpu_max_hw_queues is not None:
        os.environ["GPU_MAX_HW_QUEUES"] = str(race_cfg.gpu_max_hw_queues)
        log.info("Set GPU_MAX_HW_QUEUES=%d for race testing", race_cfg.gpu_max_hw_queues)

    # Log current setting
    hw_queues = os.environ.get("GPU_MAX_HW_QUEUES", "not set (using default)")
    log.info("GPU_MAX_HW_QUEUES=%s", hw_queues)


def check_hw_queues_warning(race_cfg: RaceConfig) -> None:
    """
    Warn if H2D or datadist racing is enabled but HW queues may mask the race.

    Args:
        race_cfg: Race configuration
    """
    racing_enabled = race_cfg.h2d_memcpy_racing or race_cfg.datadist_racing
    if not racing_enabled:
        return

    hw_queues_str = os.environ.get("GPU_MAX_HW_QUEUES")
    try:
        hw_queues_val = int(hw_queues_str) if hw_queues_str else 0
    except ValueError:
        hw_queues_val = 0

    racing_types = []
    if race_cfg.h2d_memcpy_racing:
        racing_types.append("H2D")
    if race_cfg.datadist_racing:
        racing_types.append("datadist")
    racing_str = " + ".join(racing_types)

    if hw_queues_val < 4:
        log.warning(
            "%s racing enabled but GPU_MAX_HW_QUEUES=%s (< 4). "
            "Race condition may be MASKED by implicit stream serialization. "
            "Set gpu_max_hw_queues: 4 in config or export GPU_MAX_HW_QUEUES=4 for true parallelism.",
            racing_str,
            hw_queues_str or "not set",
        )
    else:
        log.info(
            "%s racing enabled with GPU_MAX_HW_QUEUES=%d - sufficient for true stream parallelism",
            racing_str,
            hw_queues_val,
        )


def log_race_config_status(race_cfg: RaceConfig, rank: int) -> None:
    """
    Log the current race configuration status.

    Args:
        race_cfg: Race configuration
        rank: Current rank
    """
    if not race_cfg.is_any_race_enabled():
        log.debug("No race injection enabled (rank=%d)", rank)
        return

    log.info("Race injection ENABLED on rank=%d:", rank)

    if race_cfg.h2d_memcpy_racing:
        log.info("  - H2D memcpy racing: ON (start_step=%d, skip_sync=%s)",
                 race_cfg.h2d_racing_start_step, race_cfg.h2d_skip_sync_before_forward)

    if race_cfg.datadist_racing:
        log.info("  - Datadist racing: ON (start_step=%d, skip_sync=%s)",
                 race_cfg.datadist_racing_start_step, race_cfg.datadist_skip_sync_before_collective)

    if race_cfg.skip_training_warmup:
        log.info("  - Training warmup: SKIPPED (for timing variability)")
    else:
        log.info("  - Training warmup steps: %d", race_cfg.training_warmup_steps)

    if race_cfg.skip_rccl_warmup:
        log.info("  - RCCL warmup: SKIPPED (may cause hangs or race conditions)")
    else:
        log.info("  - RCCL warmup iterations: %d", race_cfg.rccl_warmup_iterations)

    if race_cfg.nan_check_collectives:
        log.info("  - NaN checking around collectives: ON")

    if race_cfg.timing_skew_enabled:
        log.info("  - Timing skew experiment: ON (mode=%s, us=%d, start_step=%d, ranks=%s)",
                 race_cfg.timing_skew_mode, race_cfg.timing_skew_us,
                 race_cfg.timing_skew_start_step,
                 race_cfg.timing_skew_ranks if race_cfg.timing_skew_ranks else "all")


def inject_timing_skew(
    step: int,
    rank: int,
    race_cfg: RaceConfig,
    stream: Optional[torch.cuda.Stream] = None,
) -> float:
    """
    Inject controlled timing skew for the timing skew experiment.

    This introduces controlled delays to demonstrate the relationship
    between timing skew and NaN occurrence.

    Args:
        step: Current training step
        rank: Current rank
        race_cfg: Race configuration
        stream: GPU stream to introduce delay on (None = current stream)

    Returns:
        Actual delay introduced in microseconds
    """
    return inject_timing_skew_from_race_config(step, rank, race_cfg, stream)


def inject_h2d_racing(
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    step: int,
    race_cfg: RaceConfig,
    rank: int,
) -> Tuple[Dict[str, torch.Tensor], Optional[torch.cuda.Stream]]:
    """
    Move batch to device with optional H2D racing.

    This is the main entry point for H2D racing. Returns the batch and
    the memcpy_stream so the caller can decide whether to sync.

    Args:
        batch: CPU batch tensors
        device: Target GPU device
        step: Current training step
        race_cfg: Race configuration
        rank: Current rank

    Returns:
        Tuple of (batch on GPU, memcpy_stream or None)
    """
    return _move_batch_racing(batch, device, step, race_cfg, rank)


def inject_datadist_racing(
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    step: int,
    race_cfg: RaceConfig,
    rank: int,
) -> Tuple[Dict[str, torch.Tensor], Optional[torch.cuda.Stream]]:
    """
    Inject datadist (all_to_all) racing pattern.

    This simulates TorchRec's SparseDataDistributedAllToAll where all_to_all
    runs on a separate stream from FSDP collectives, potentially causing race.

    Args:
        batch: Batch tensors (already on GPU)
        device: GPU device
        step: Current training step
        race_cfg: Race configuration
        rank: Current rank

    Returns:
        Tuple of (batch, datadist_stream or None)
    """
    return _inject_datadist_racing(batch, device, step, race_cfg, rank)


# =============================================================================
# In-flight read instability checks (re-exported from inflight_checks module)
# =============================================================================
from aorta.race.inflight_checks import (
    schedule_inflight_check,
    flush_inflight_checks,
    clear_inflight_checks,
)
