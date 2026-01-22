"""
Race experiment configuration dataclass.

This module defines the RaceConfig dataclass which contains all settings
for race condition injection experiments.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class RaceConfig:
    """
    Configuration for race condition injection experiments.

    Four categories of race conditions are supported:

    1. H2D Race (realistic pattern):
       - h2d_memcpy_racing: Uses separate stream for H2D batch copy
       - h2d_skip_sync_before_forward: Skips wait_stream() before forward (causes race!)
       - h2d_racing_start_step: Step to start H2D racing

    2. Datadist Race (TorchRec-style all_to_all):
       - datadist_racing: Uses separate stream for all_to_all operations
       - datadist_skip_sync_before_collective: Skips wait_stream() before FSDP collective (causes race!)
       - datadist_racing_start_step: Step to start datadist racing

    3. Timing Skew Experiment (demonstrates NaN progression):
       - timing_skew_enabled: Enable controlled timing skew
       - timing_skew_mode: none, fixed, progressive, random
       - timing_skew_us: Delay in microseconds
       - timing_skew_ranks: Which ranks get delayed
       - timing_skew_start_step: Step to start skew

    Supporting options:
       - skip_training_warmup: Skip training warmup to maximize timing variability
       - skip_rccl_warmup: Skip RCCL communicator warmup before FSDP init
       - nan_check_collectives: Enable NaN checking around RCCL collectives
       - gpu_max_hw_queues: Set GPU_MAX_HW_QUEUES (4+ needed to expose race)
    """

    # =========================================================================
    # H2D memcpy racing (realistic pattern)
    # =========================================================================
    h2d_memcpy_racing: bool = False
    """Use separate memcpy_stream for H2D batch copy."""

    h2d_skip_sync_before_forward: bool = False
    """Skip wait_stream() before forward pass - THIS CAUSES THE RACE!"""

    h2d_racing_start_step: int = 0
    """Step to start H2D racing (0 = aggressive, from first step)."""

    # =========================================================================
    # Datadist racing (TorchRec-style all_to_all on separate stream)
    # =========================================================================
    datadist_racing: bool = False
    """Use separate datadist_stream for all_to_all operations."""

    datadist_skip_sync_before_collective: bool = False
    """Skip wait_stream() before FSDP collective - THIS CAUSES THE RACE!"""

    datadist_racing_start_step: int = 0
    """Step to start datadist racing (0 = aggressive, from first step)."""

    # =========================================================================
    # Supporting options
    # =========================================================================
    skip_training_warmup: bool = False
    """Skip training warmup to maximize timing variability for race testing."""

    training_warmup_steps: int = 1
    """Number of training warmup steps to run (if not skipped)."""

    warmup_batch_size: Optional[int] = None
    """
    Batch size for warmup steps. If None, uses the training batch_size.
    Set this smaller than training batch_size to speed up warmup while
    still exercising the collectives, then use larger batch during racing
    for wider race windows.
    """

    skip_rccl_warmup: bool = False
    """Skip RCCL communicator warmup before FSDP init to test race conditions."""

    rccl_warmup_iterations: int = 10
    """Number of RCCL warmup iterations (if not skipped). Higher = more stable but slower startup."""

    nan_check_collectives: bool = False
    """Enable NaN checking before/after RCCL collectives."""

    gpu_max_hw_queues: Optional[int] = None
    """
    Set GPU_MAX_HW_QUEUES environment variable.
    
    CRITICAL for race exposure:
    - 1-2: Streams share HW queues → implicit serialization → RACE MASKED
    - 4+: Each stream gets own HW queue → true parallelism → RACE EXPOSED
    
    Recommended: 4 for race testing.
    """

    # =========================================================================
    # Timing skew experiment (demonstrates NaN progression)
    # =========================================================================
    timing_skew_enabled: bool = False
    """Enable controlled timing skew experiment to show NaN progression."""

    timing_skew_mode: str = "none"
    """
    Skew mode:
    - none: No artificial skew
    - fixed: Fixed delay in microseconds
    - progressive: Increase delay each step (skew_us * step)
    - random: Random delay within range
    """

    timing_skew_us: int = 0
    """Base delay in microseconds (for fixed/progressive modes)."""

    timing_skew_ranks: List[int] = field(default_factory=list)
    """Which ranks get delayed. Empty = all ranks."""

    timing_skew_start_step: int = 3
    """Step to start introducing timing skew (after warmup)."""

    def is_any_race_enabled(self) -> bool:
        """Check if any race injection is enabled."""
        return (
            self.h2d_memcpy_racing
            or self.datadist_racing
            or self.timing_skew_enabled
        )

    def is_h2d_race_enabled(self) -> bool:
        """Check if H2D racing is enabled."""
        return self.h2d_memcpy_racing and self.h2d_skip_sync_before_forward

    def is_datadist_race_enabled(self) -> bool:
        """Check if datadist racing is enabled."""
        return self.datadist_racing and self.datadist_skip_sync_before_collective

    def is_timing_skew_active(self, step: int) -> bool:
        """Check if timing skew should be applied at this step."""
        return (
            self.timing_skew_enabled
            and self.timing_skew_mode != "none"
            and step >= self.timing_skew_start_step
        )
