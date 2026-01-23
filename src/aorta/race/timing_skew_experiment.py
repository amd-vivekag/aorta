"""
Timing Skew to NaN Experiment.

This module provides a controlled experiment to demonstrate the relationship
between timing skew and NaN occurrence. It bridges the gap between:
- What we observed: Timing differences → Collective timeout/hang
- What happens in production: Stream races → NaN

The experiment introduces controlled timing skew to show the progression:
- No skew + sync = healthy
- Small skew + no sync = intermittent NaN
- Medium skew + no sync = consistent NaN
- Large skew = hang/timeout

Usage:
    Configure via race_experiment section:
    
    timing_skew_experiment:
      enabled: true
      skew_mode: "progressive"  # none, fixed, progressive, random
      skew_us: 100              # Microseconds of delay (for fixed mode)
      skew_ranks: [0, 1]        # Which ranks get delayed
      check_nan_every_step: true
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

log = logging.getLogger(__name__)


@dataclass
class TimingSkewConfig:
    """Configuration for timing skew experiment."""
    
    enabled: bool = False
    """Enable the timing skew experiment."""
    
    skew_mode: str = "none"
    """
    Skew mode:
    - none: No artificial skew
    - fixed: Fixed delay in microseconds
    - progressive: Increase delay each step
    - random: Random delay within range
    """
    
    skew_us: int = 0
    """Base delay in microseconds (for fixed/progressive modes)."""
    
    skew_ranks: List[int] = None
    """Which ranks get delayed. None = all ranks."""
    
    skew_start_step: int = 0
    """Step to start introducing skew."""
    
    check_nan_every_step: bool = True
    """Check for NaN after every step."""
    
    log_timing: bool = True
    """Log timing information for analysis."""
    
    def __post_init__(self):
        if self.skew_ranks is None:
            self.skew_ranks = []


def introduce_timing_skew(
    step: int,
    rank: int,
    config: TimingSkewConfig,
    stream: Optional[torch.cuda.Stream] = None,
) -> float:
    """
    Introduce controlled timing skew on specific ranks.
    
    This simulates the timing variability that can cause NaN in production workloads.
    By controlling the amount of skew, we can show the progression from
    healthy → NaN → hang.
    
    Args:
        step: Current training step
        rank: Current rank
        config: Timing skew configuration
        stream: GPU stream to introduce delay on (None = current stream)
    
    Returns:
        Actual delay introduced in microseconds
    """
    if not config.enabled:
        return 0.0
    
    if step < config.skew_start_step:
        return 0.0
    
    # Check if this rank should be skewed
    if config.skew_ranks and rank not in config.skew_ranks:
        return 0.0
    
    # Calculate delay based on mode
    if config.skew_mode == "none":
        delay_us = 0
    elif config.skew_mode == "fixed":
        delay_us = config.skew_us
    elif config.skew_mode == "progressive":
        # Increase delay each step: base * (step - start_step + 1)
        delay_us = config.skew_us * (step - config.skew_start_step + 1)
    elif config.skew_mode == "random":
        import random
        delay_us = random.randint(0, config.skew_us)
    else:
        delay_us = 0
    
    if delay_us <= 0:
        return 0.0
    
    # Introduce delay via GPU kernel (more realistic than CPU sleep)
    if stream is not None:
        with torch.cuda.stream(stream):
            _gpu_delay_kernel(delay_us)
    else:
        _gpu_delay_kernel(delay_us)
    
    if config.log_timing:
        log.debug(
            "TIMING SKEW: rank=%d step=%d delay=%d us mode=%s",
            rank, step, delay_us, config.skew_mode
        )
    
    return delay_us


def _gpu_delay_kernel(delay_us: int) -> None:
    """
    Introduce a delay on the GPU via a compute kernel.
    
    This is more realistic than CPU sleep because it actually
    occupies GPU resources and affects GPU stream scheduling.
    
    IMPORTANT: We do NOT synchronize here - the delay work is enqueued
    to the GPU but the CPU continues immediately. This preserves the
    race window between H2D/datadist and forward pass.
    """
    if delay_us <= 0:
        return
    
    # Create a tensor and do busy work to introduce delay
    # Approximate: 1000 iterations ≈ 10 microseconds on MI300
    iterations = max(1, delay_us * 100)
    
    device = torch.cuda.current_device()
    x = torch.ones(1024, device=device)
    
    for _ in range(iterations // 1000 + 1):
        x = x * 1.0001  # Small multiply to prevent optimization
    
    # DO NOT synchronize - let GPU work run async to preserve race window
    # Previously: torch.cuda.current_stream().synchronize() - this was
    # blocking and serializing streams, hiding the NaN race condition


# ============================================================================
# Integration helpers for RaceConfig
# ============================================================================

def timing_skew_config_from_race_config(race_cfg) -> TimingSkewConfig:
    """
    Create a TimingSkewConfig from a RaceConfig.
    
    This bridges the RaceConfig fields to the TimingSkewConfig dataclass.
    
    Args:
        race_cfg: RaceConfig instance with timing_skew_* fields
    
    Returns:
        TimingSkewConfig instance
    """
    return TimingSkewConfig(
        enabled=race_cfg.timing_skew_enabled,
        skew_mode=race_cfg.timing_skew_mode,
        skew_us=race_cfg.timing_skew_us,
        skew_ranks=list(race_cfg.timing_skew_ranks) if race_cfg.timing_skew_ranks else [],
        skew_start_step=race_cfg.timing_skew_start_step,
        check_nan_every_step=race_cfg.nan_check_collectives,
        log_timing=True,
    )


def inject_timing_skew_from_race_config(
    step: int,
    rank: int,
    race_cfg,
    stream: Optional[torch.cuda.Stream] = None,
) -> float:
    """
    Inject timing skew using RaceConfig directly.
    
    This is a convenience wrapper that creates TimingSkewConfig from RaceConfig
    and calls introduce_timing_skew.
    
    Args:
        step: Current training step
        rank: Current rank
        race_cfg: RaceConfig instance
        stream: GPU stream to introduce delay on (None = current stream)
    
    Returns:
        Actual delay introduced in microseconds
    """
    if not race_cfg.is_timing_skew_active(step):
        return 0.0
    
    timing_cfg = timing_skew_config_from_race_config(race_cfg)
    return introduce_timing_skew(step, rank, timing_cfg, stream)


def check_loss_for_nan(
    loss: torch.Tensor,
    step: int,
    rank: int,
) -> bool:
    """
    Check if loss is NaN and log if so.
    
    Args:
        loss: Loss tensor
        step: Current training step
        rank: Current rank
    
    Returns:
        True if NaN detected
    """
    if torch.isnan(loss).any():
        log.warning(
            "NaN LOSS DETECTED: rank=%d step=%d loss=%s",
            rank, step, loss.item() if loss.numel() == 1 else "tensor"
        )
        return True
    return False


def check_gradients_for_nan(
    model: torch.nn.Module,
    step: int,
    rank: int,
) -> Tuple[bool, int]:
    """
    Check model gradients for NaN values.
    
    Args:
        model: The model to check
        step: Current training step
        rank: Current rank
    
    Returns:
        Tuple of (has_nan, nan_count)
    """
    total_nan = 0
    total_params = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            nan_count = torch.isnan(param.grad).sum().item()
            total_nan += nan_count
            total_params += param.grad.numel()
    
    if total_nan > 0:
        log.warning(
            "NaN GRADIENTS DETECTED: rank=%d step=%d nan_count=%d/%d (%.4f%%)",
            rank, step, total_nan, total_params,
            100.0 * total_nan / total_params if total_params > 0 else 0
        )
        return True, total_nan
    
    return False, 0
