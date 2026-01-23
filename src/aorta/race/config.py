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
    
    Recommended: 4 for race testing, or 3 with client_stream_layout.
    """

    client_stream_layout: bool = False
    """
    Use client's stream layout for accurate reproduction of their NaN issue.
    
    When enabled:
    - 3 streams only: default_stream, memcpy_stream, datadist_stream
    - Forward/backward/clip/optimizer/FSDP collectives run on default stream
    - Only H2D and datadist operations on separate racing streams
    - DistributedOpsInterceptor is bypassed (no stream redirection for collectives)
    
    This matches the client's actual TorchRec-style architecture where:
    - memcpy_stream races with default stream (H2D not synced before forward)
    - datadist_stream races with default stream (all_to_all not synced before collective)
    
    Recommended: Use with gpu_max_hw_queues: 3 for 1:1 stream-to-queue mapping.
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

    # =========================================================================
    # Debug instrumentation
    # =========================================================================
    timing_debug_logs: bool = False
    """
    Enable detailed timing logs around H2D, datadist, and forward operations.
    
    When enabled, logs wall-clock timestamps for:
    - H2D copy (memcpy_stream operations)
    - Datadist racing (all_to_all operations)
    - Timing skew delay
    - Forward pass
    
    Also logs gap/overlap between operations to verify if race windows exist.
    Negative gap = overlap = potential race.
    """

    gpu_event_timing: bool = False
    """
    Enable GPU event-based timing to measure actual stream overlap.
    
    Unlike timing_debug_logs which measures CPU-side timestamps, this uses
    CUDA/HIP events to measure actual GPU execution times and overlap.
    
    Event recording is non-blocking and doesn't interfere with race conditions.
    Timing calculations happen AFTER the iteration completes (when we sync anyway).
    
    Logs GPU-side durations and overlap between streams:
    - GPU_H2D_DUR: Actual GPU time for H2D copy
    - GPU_DD_DUR: Actual GPU time for datadist all_to_all
    - GPU_FWD_DUR: Actual GPU time for forward pass
    - GPU_OVERLAP: Time between stream operations (negative = overlap = race!)
    
    WARNING: Cross-stream timing (e.g., DD->FWD) is INACCURATE because events
    are on different streams. Use nccl_async_diagnostic for accurate overlap detection.
    """

    nccl_async_diagnostic: bool = False
    """
    Enable NCCL async behavior diagnostic.
    
    When enabled, checks if the datadist all_to_all work is still in-flight
    when forward pass starts. This uses work.is_completed() which is non-blocking
    and does NOT affect the race condition.
    
    Logs:
    - NCCL_DIAG: Whether all_to_all is PENDING (async, race possible) or 
      COMPLETED (sync, race may be masked by NCCL internal sync)
    
    This is the authoritative way to verify if a race window exists, because
    it directly queries the NCCL work handle status rather than relying on
    GPU event timing which is inaccurate for cross-stream measurements.
    """

    datadist_use_real_dependency: bool = True
    """
    Make datadist output actually used by forward pass.
    
    When enabled, the all_to_all output tensor is added as noise to batch["dense"],
    creating a real data dependency that forward must read. This more accurately
    reproduces the client's TorchRec pattern where distributed embeddings race
    with the forward pass.
    
    Without this, datadist creates synthetic tensors that are discarded,
    so there's no actual data race with forward (just stream contention).
    """

    # =========================================================================
    # Controlled in-flight reads (no NaN poisoning)
    # =========================================================================
    h2d_split_dense_copy: bool = False
    """
    Copy the dense tensor in two chunks on the memcpy stream.

    The tail chunk is copied last to increase the chance it is still in-flight
    when forward starts. This helps ensure the forward reads a tensor that is
    actively being copied without injecting NaNs.
    """

    h2d_dense_tail_fraction: float = 0.5
    """
    Fraction of dense elements copied in the tail chunk (copied last).
    Clamped to (0, 1) and adjusted to keep at least one element in each chunk.
    """

    datadist_split_alltoall: bool = False
    """
    Split the all_to_all into two sequential collectives on the datadist stream.

    The tail chunk collective is launched last, so reading that tail immediately
    on the default stream is guaranteed to be in-flight.
    """

    datadist_tail_fraction: float = 0.5
    """
    Fraction of all_to_all elements assigned to the tail chunk (launched last).
    Clamped to (0, 1) and adjusted to be divisible by world_size.
    """

    datadist_read_tail_only: bool = False
    """
    Use only the tail chunk of the all_to_all output when creating the
    datadist->forward dependency. This guarantees the read targets the
    in-flight chunk without NaN poisoning.
    """

    # =========================================================================
    # In-flight read instability checks
    # =========================================================================
    inflight_read_check_enabled: bool = False
    """
    Enable repeated in-flight reads to detect instability/mismatch.

    When enabled, the tail region of in-flight tensors is read multiple times
    on the default stream while the racing stream is still writing. If the
    values change between reads, it indicates a torn read (race condition).
    """

    inflight_read_repeats: int = 0
    """
    Number of repeated reads to perform on the in-flight tail region.

    Set to 0 to disable repeated reads. Higher values increase the chance
    of detecting instability but add more GPU work on the default stream.
    Recommended: 3-10 for typical detection.
    """

    inflight_read_sample_size: int = 4096
    """
    Number of elements to sample from the tail region for instability checks.

    A smaller sample reduces GPU work; a larger sample increases detection
    probability. The sample is taken from the start of the tail region.
    """

    datadist_tensor_size: Optional[int] = 1_000_000
    """
    Fixed tensor size for datadist all_to_all operation.
    
    If set, uses this fixed size instead of basing it on dense.numel().
    This allows controlling the all_to_all duration independently of dense_dim.
    
    Recommended: 1M-10M elements for ~100-500ms all_to_all duration.
    Set to None to use dense.numel() (original behavior, but can be very slow
    with large dense_dim).
    """

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
