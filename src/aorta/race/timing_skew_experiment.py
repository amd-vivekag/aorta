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
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

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
        stream: CUDA stream to introduce delay on (None = current stream)
    
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
    occupies GPU resources and affects CUDA stream scheduling.
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
    
    # Ensure kernel completes
    torch.cuda.current_stream().synchronize()


def check_tensor_for_nan(
    tensor: torch.Tensor,
    name: str,
    step: int,
    rank: int,
) -> Tuple[bool, int]:
    """
    Check a tensor for NaN values.
    
    Returns:
        Tuple of (has_nan, nan_count)
    """
    if not isinstance(tensor, torch.Tensor):
        return False, 0
    
    nan_mask = torch.isnan(tensor)
    nan_count = nan_mask.sum().item()
    has_nan = nan_count > 0
    
    if has_nan:
        log.warning(
            "NaN DETECTED: rank=%d step=%d tensor=%s nan_count=%d/%d (%.2f%%)",
            rank, step, name, nan_count, tensor.numel(),
            100.0 * nan_count / tensor.numel()
        )
    
    return has_nan, nan_count


def check_batch_for_nan(
    batch: Dict[str, torch.Tensor],
    step: int,
    rank: int,
    prefix: str = "",
) -> Dict[str, Tuple[bool, int]]:
    """
    Check all tensors in a batch for NaN values.
    
    Returns:
        Dictionary mapping tensor names to (has_nan, nan_count)
    """
    results = {}
    
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            name = f"{prefix}{key}" if prefix else key
            has_nan, nan_count = check_tensor_for_nan(value, name, step, rank)
            results[key] = (has_nan, nan_count)
    
    return results


def run_timing_skew_test(
    rank: int,
    world_size: int,
    device: torch.device,
    config: TimingSkewConfig,
    num_steps: int = 100,
) -> Dict[str, any]:
    """
    Run a standalone timing skew experiment.
    
    This creates a minimal distributed workload that demonstrates
    the relationship between timing skew and NaN.
    
    Args:
        rank: Current rank
        world_size: Total number of ranks
        device: CUDA device
        config: Timing skew configuration
        num_steps: Number of steps to run
    
    Returns:
        Dictionary with experiment results
    """
    results = {
        "rank": rank,
        "world_size": world_size,
        "num_steps": num_steps,
        "config": config,
        "nan_steps": [],
        "hang_step": None,
        "total_nans": 0,
    }
    
    # Create streams to simulate multi-stream pattern
    memcpy_stream = torch.cuda.Stream(device=device)
    compute_stream = torch.cuda.Stream(device=device)
    
    log.info(
        "Starting timing skew experiment: rank=%d mode=%s skew_us=%d",
        rank, config.skew_mode, config.skew_us
    )
    
    for step in range(num_steps):
        try:
            # Simulate H2D copy on memcpy_stream
            with torch.cuda.stream(memcpy_stream):
                batch_data = torch.randn(1024, 1024, device=device)
                
                # Introduce timing skew on specific ranks
                introduce_timing_skew(step, rank, config, memcpy_stream)
            
            # Simulate forward pass on compute_stream WITHOUT waiting
            # This is where the race condition occurs
            with torch.cuda.stream(compute_stream):
                # Read data that may not be fully written
                result = batch_data.sum()
            
            # Check for NaN
            has_nan, nan_count = check_tensor_for_nan(
                result, "result", step, rank
            )
            
            if has_nan:
                results["nan_steps"].append(step)
                results["total_nans"] += nan_count
            
            # Collective to synchronize ranks
            # This is where hang can occur if skew is too large
            dist.barrier()
            
            if step % 10 == 0:
                log.info(
                    "Timing skew experiment: rank=%d step=%d nans_so_far=%d",
                    rank, step, len(results["nan_steps"])
                )
        
        except Exception as e:
            log.error(
                "Timing skew experiment error: rank=%d step=%d error=%s",
                rank, step, str(e)
            )
            results["hang_step"] = step
            break
    
    log.info(
        "Timing skew experiment complete: rank=%d nan_steps=%s total_nans=%d",
        rank, results["nan_steps"], results["total_nans"]
    )
    
    return results


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
        stream: CUDA stream to introduce delay on (None = current stream)
    
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


def inject_nan_on_race(
    batch: Dict[str, torch.Tensor],
    step: int,
    rank: int,
    probability: float,
    race_type: str = "unknown",
    inject_all_tensors: bool = False,
    nan_fraction: float = 0.02,
) -> bool:
    """
    Inject NaN into batch tensors to simulate race corruption.
    
    This simulates what happens when a race condition actually corrupts data
    to produce NaN values. Used to bridge:
    - What we observe: race → eventual hang (corruption not visible as NaN)
    - What happens in production: race → NaN in loss/gradients
    
    Args:
        batch: Batch tensors to potentially corrupt
        step: Current training step
        rank: Current rank
        probability: Probability of injection (0.0-1.0). Use 1.0 for guaranteed injection.
        race_type: Type of race ("h2d", "datadist", etc.) for logging
        inject_all_tensors: If True, inject into ALL tensors; if False, pick one random tensor
        nan_fraction: Fraction of tensor elements to corrupt (default 2%)
    
    Returns:
        True if NaN was injected
    """
    import random
    
    if probability <= 0.0:
        return False
    
    # Probabilistic check (skip if probability < 1.0 and we "lose" the roll)
    if probability < 1.0 and random.random() > probability:
        log.info(
            "╔══════════════════════════════════════════════════════════════╗\n"
            "║  RACE→NaN: step=%d rank=%d SKIPPED (prob=%.0f%% roll failed)   ║\n"
            "╚══════════════════════════════════════════════════════════════╝",
            step, rank, probability * 100
        )
        return False
    
    # Find injectable tensors
    tensor_keys = [k for k, v in batch.items() if isinstance(v, torch.Tensor) and v.numel() > 0]
    if not tensor_keys:
        log.warning(
            "RACE→NaN: step=%d rank=%d NO TENSORS in batch to corrupt",
            step, rank
        )
        return False
    
    # Select which tensors to corrupt
    if inject_all_tensors:
        keys_to_corrupt = tensor_keys
    else:
        keys_to_corrupt = [random.choice(tensor_keys)]
    
    total_nan_injected = 0
    total_elements = 0
    corrupted_tensors = []
    
    for key in keys_to_corrupt:
        tensor = batch[key]
        num_elements = tensor.numel()
        num_nan = max(1, int(num_elements * nan_fraction))
        
        # Create flat view and inject NaN at random positions
        flat = tensor.view(-1)
        indices = torch.randperm(num_elements, device=tensor.device)[:num_nan]
        flat[indices] = float('nan')
        
        total_nan_injected += num_nan
        total_elements += num_elements
        corrupted_tensors.append(f"{key}({num_nan})")
    
    # Prominent logging with visual markers
    log.warning(
        "\n"
        "╔══════════════════════════════════════════════════════════════════════════╗\n"
        "║  RACE->NaN INJECTION                                                     ║\n"
        "╠══════════════════════════════════════════════════════════════════════════╣\n"
        "║  Step: %-4d   Rank: %-2d   Race Type: %-20s               ║\n"
        "║  Tensors corrupted: %-50s ║\n"
        "║  Total NaN injected: %d / %d elements (%.2f%%)                           ║\n"
        "╚══════════════════════════════════════════════════════════════════════════╝",
        step, rank, race_type,
        ", ".join(corrupted_tensors),
        total_nan_injected, total_elements,
        100.0 * total_nan_injected / total_elements if total_elements > 0 else 0
    )
    
    return True


# Global tracker for NaN injection statistics
_nan_injection_stats = {
    "total_injections": 0,
    "total_nan_values": 0,
    "steps_with_nan": [],
}


def get_nan_injection_stats() -> Dict[str, any]:
    """Get statistics about NaN injections so far."""
    return _nan_injection_stats.copy()


def reset_nan_injection_stats() -> None:
    """Reset NaN injection statistics."""
    _nan_injection_stats["total_injections"] = 0
    _nan_injection_stats["total_nan_values"] = 0
    _nan_injection_stats["steps_with_nan"] = []


# Experiment presets for common scenarios
EXPERIMENT_PRESETS = {
    "baseline": TimingSkewConfig(
        enabled=False,
    ),
    "small_skew": TimingSkewConfig(
        enabled=True,
        skew_mode="fixed",
        skew_us=50,
        skew_ranks=[0],  # Only rank 0 delayed
        skew_start_step=3,
    ),
    "medium_skew": TimingSkewConfig(
        enabled=True,
        skew_mode="fixed",
        skew_us=500,
        skew_ranks=[0],
        skew_start_step=3,
    ),
    "large_skew": TimingSkewConfig(
        enabled=True,
        skew_mode="fixed",
        skew_us=5000,
        skew_ranks=[0],
        skew_start_step=3,
    ),
    "progressive": TimingSkewConfig(
        enabled=True,
        skew_mode="progressive",
        skew_us=10,  # Starts at 10us, increases each step
        skew_ranks=[0],
        skew_start_step=3,
    ),
}


def get_experiment_preset(name: str) -> TimingSkewConfig:
    """Get a predefined experiment configuration."""
    if name not in EXPERIMENT_PRESETS:
        raise ValueError(
            f"Unknown preset: {name}. Available: {list(EXPERIMENT_PRESETS.keys())}"
        )
    return EXPERIMENT_PRESETS[name]
