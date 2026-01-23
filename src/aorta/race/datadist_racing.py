"""
Datadist (data distribution) racing module.

This module implements datadist racing patterns where the race occurs when:

1. all_to_all happens on datadist_stream (not default stream)
2. FSDP collective starts before all_to_all completes
3. Collective reads potentially incomplete distributed data

This simulates TorchRec's SparseDataDistributedAllToAll pattern where
data distribution runs on a separate stream from FSDP collectives.

Using the default stream for all_to_all avoids this race. This module
intentionally uses a separate stream to simulate the race condition.

CLIENT PATTERN MATCHING:
The client uses async_op=True for their all_to_all, which allows the CPU
to continue immediately. This creates the race window where:
- CPU launches all_to_all on datadist_stream (async, returns immediately)
- CPU launches forward on default_stream (while all_to_all still running)
- GPU streams race: datadist_stream vs default_stream

We match this pattern by using async_op=True and returning the work handle.
The caller is responsible for waiting on the work handle at an appropriate
time (e.g., end of step) to prevent NCCL desync between steps.
"""

import logging
from typing import Dict, Tuple, Optional, Any, List

import torch
import torch.distributed as dist

from aorta.race.config import RaceConfig


log = logging.getLogger(__name__)


# Global datadist stream for racing (created lazily per device)
_datadist_streams: Dict[torch.device, torch.cuda.Stream] = {}

# Global work handle storage for async collectives (per step, will be waited at end of step)
_pending_datadist_work: List[Any] = []
_pending_datadist_tail_work: Optional[Any] = None

# Diagnostic: track if NCCL appeared synchronous (work completed immediately)
_nccl_sync_warning_logged: bool = False


def get_datadist_stream(device: torch.device) -> torch.cuda.Stream:
    """
    Get or create the datadist stream for a device.

    The datadist stream is used for all_to_all operations in the racing pattern.
    It's created lazily and cached per device.

    Args:
        device: GPU device

    Returns:
        GPU stream for datadist operations
    """
    if device not in _datadist_streams:
        _datadist_streams[device] = torch.cuda.Stream(device=device)
    return _datadist_streams[device]


def inject_datadist_racing(
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    step: int,
    race_cfg: RaceConfig,
    rank: int,
) -> Tuple[Dict[str, torch.Tensor], Optional[torch.cuda.Stream]]:
    """
    Inject datadist (all_to_all) racing pattern.

    This simulates TorchRec's SparseDataDistributedAllToAll where:
    1. all_to_all happens on datadist_stream (not default stream)
    2. FSDP collective may start before all_to_all completes
    3. Collective reads potentially incomplete distributed data

    The race condition occurs when:
    - datadist_racing=True: all_to_all runs on separate stream
    - datadist_skip_sync_before_collective=True: no wait before FSDP collective

    CLIENT PATTERN MATCHING:
    We use async_op=True so the CPU returns immediately after launching the
    collective. This matches the client's TorchRec pattern where:
    - all_to_all is launched async on datadist_stream
    - CPU continues to launch forward on default_stream
    - GPU streams race: datadist vs default

    The work handle is stored globally and should be waited on at the end of
    each step via wait_pending_datadist_work() to prevent NCCL desync.

    Args:
        batch: Batch tensors (already on GPU)
        device: GPU device
        step: Current training step
        race_cfg: Race configuration
        rank: Current rank

    Returns:
        Tuple of (batch tensors, datadist_stream or None if not racing)
        The datadist_stream is returned so caller can decide whether to sync
    """
    global _pending_datadist_work, _pending_datadist_tail_work

    if not race_cfg.datadist_racing:
        # Normal path: no datadist racing
        return batch, None

    if step < race_cfg.datadist_racing_start_step:
        # Before racing starts: skip
        return batch, None

    if not dist.is_initialized():
        # No distributed environment: can't do all_to_all
        log.debug("DATADIST RACING: rank=%d step=%d skipped (dist not initialized)", rank, step)
        return batch, None

    world_size = dist.get_world_size()
    if world_size < 2:
        # Single rank: all_to_all is a no-op
        return batch, None

    # Racing path: use separate datadist stream for all_to_all
    datadist_stream = get_datadist_stream(device)

    # Get dense tensor from batch to create real data dependency
    dense = batch.get("dense")
    if dense is None or not isinstance(dense, torch.Tensor):
        # Fallback: find any tensor from batch
        for key, tensor in batch.items():
            if isinstance(tensor, torch.Tensor) and tensor.numel() > 0:
                dense = tensor
                break

    if dense is None:
        log.warning("DATADIST RACING: No suitable tensor found in batch, skipping")
        return batch, None

    # Prepare tensors for all_to_all (on default stream first for allocation)
    # Use fixed size if configured, otherwise use dense.numel()
    # Fixed size is important when dense_dim is large - otherwise all_to_all
    # takes too long and completes before forward starts (eliminating the race!)
    dense_numel = dense.numel()
    if race_cfg.datadist_tensor_size is not None:
        # Use fixed size for consistent all_to_all duration
        # This decouples datadist timing from dense_dim setting
        chunk_size = max(1, race_cfg.datadist_tensor_size // world_size)
        total_size = chunk_size * world_size
        log.debug(
            "DATADIST RACING: Using fixed tensor size=%d (dense_numel=%d)",
            total_size, dense_numel
        )
    else:
        # Original behavior: base size on dense tensor
        chunk_size = max(1, dense_numel // world_size)
        total_size = chunk_size * world_size

    # Create tensors for all_to_all on the device
    input_tensor = torch.randn(
        total_size, device=device, dtype=dense.dtype
    )
    output_tensor = torch.empty_like(input_tensor)

    # Track tail offset when using split all_to_all
    tail_offset: Optional[int] = None

    # Perform all_to_all on datadist_stream to simulate TorchRec pattern
    with torch.cuda.stream(datadist_stream):
        # Reset pending work list for this step
        _pending_datadist_work = []
        _pending_datadist_tail_work = None

        if race_cfg.datadist_split_alltoall and total_size >= world_size * 2:
            # Split into two sequential collectives so the tail chunk is launched last.
            tail_frac = max(0.0, min(1.0, race_cfg.datadist_tail_fraction))
            tail_size = int(total_size * tail_frac)
            tail_size = max(world_size, (tail_size // world_size) * world_size)
            prefix_size = total_size - tail_size
            if prefix_size >= world_size:
                tail_offset = prefix_size
                prefix_in = input_tensor[:prefix_size]
                prefix_out = output_tensor[:prefix_size]
                tail_in = input_tensor[tail_offset:]
                tail_out = output_tensor[tail_offset:]
                work0 = dist.all_to_all_single(
                    prefix_out,
                    prefix_in,
                    async_op=True,
                )
                work1 = dist.all_to_all_single(
                    tail_out,
                    tail_in,
                    async_op=True,
                )
                _pending_datadist_work.extend([work0, work1])
                _pending_datadist_tail_work = work1
            else:
                # Fallback to single collective if split would be empty
                work = dist.all_to_all_single(
                    output_tensor,
                    input_tensor,
                    async_op=True,
                )
                _pending_datadist_work.append(work)
                _pending_datadist_tail_work = work
        else:
            # Perform all_to_all on datadist stream (racing with default stream)
            # CLIENT PATTERN: Use async_op=True so CPU returns immediately
            # This allows forward to start on default_stream while all_to_all
            # is still running on datadist_stream - creating the race window.
            #
            # We store the work handle and wait for it at the end of the step
            # to prevent NCCL collective sequence desync between steps.
            work = dist.all_to_all_single(
                output_tensor,
                input_tensor,
                async_op=True,  # CPU returns immediately - matches client pattern!
            )
            _pending_datadist_work.append(work)
            _pending_datadist_tail_work = work

    # EXIT datadist_stream context - now on DEFAULT stream!
    #
    # CRITICAL FOR PARTIAL READ RACE:
    # The all_to_all is running on datadist_stream, writing to output_tensor.
    # Below code runs on DEFAULT stream, reading from output_tensor.
    # Since there's NO sync between streams, default stream may read output_tensor
    # WHILE datadist_stream's all_to_all is still writing = PARTIAL READ = NaN!
    #
    # This is the exact pattern that causes NaN in the client's TorchRec workload:
    # - datadist_stream: all_to_all writes to embedding buffer
    # - default_stream: forward reads from embedding buffer (potentially mid-write)
    # - Result: garbage/NaN values from reading partially written memory

    if race_cfg.datadist_use_real_dependency:
        # Read output_tensor ON DEFAULT STREAM while all_to_all may still be writing!
        # This creates the TRUE partial read race condition.
        #
        # Handle size mismatch: output_tensor may be smaller than dense when using
        # fixed datadist_tensor_size. We tile/repeat the output to match dense size.
        source_tensor = output_tensor
        if race_cfg.datadist_read_tail_only and tail_offset is not None:
            source_tensor = output_tensor[tail_offset:]

        # Schedule repeated in-flight reads to detect instability
        if (race_cfg.inflight_read_check_enabled and
            race_cfg.inflight_read_repeats > 0):
            from aorta.race.inflight_checks import schedule_inflight_check
            schedule_inflight_check(
                name="datadist_tail",
                tensor=source_tensor,
                sample_size=race_cfg.inflight_read_sample_size,
                repeats=race_cfg.inflight_read_repeats,
                step=step,
                rank=rank,
            )

        output_numel = source_tensor.numel()
        if output_numel >= dense_numel:
            # Output is large enough, slice it
            noise = source_tensor[:dense_numel].view_as(dense)
        else:
            # Output is smaller, tile it to match dense size
            # This still creates the race - we're reading output_tensor mid-write
            repeat_times = (dense_numel + output_numel - 1) // output_numel
            tiled = source_tensor.repeat(repeat_times)[:dense_numel]
            noise = tiled.view_as(dense)

        # Check if all_to_all is still pending RIGHT AFTER reading output_tensor
        # This is the critical race point - we just read data that may be mid-write
        #
        # IMPORTANT: We use is_completed() which is a NON-BLOCKING polling query.
        # We do NOT use .item() here as that would cause a CUDA sync and mask the race!
        if _pending_datadist_work is not None:
            try:
                read_during_write = not _pending_datadist_work.is_completed()
                if read_during_write:
                    log.info(
                        "DATADIST_READ_RACE: step=%d rank=%d output_tensor read while all_to_all STILL PENDING - TRUE RACE!",
                        step, rank
                    )
                    # NOTE: We intentionally do NOT check for NaN/Inf here!
                    # Calling .item() would sync CUDA and mask the race.
                    # NaN check will happen later in the loss check (after forward).
                else:
                    log.warning(
                        "DATADIST_READ_RACE: step=%d rank=%d all_to_all COMPLETED before output read - NO RACE at read point!",
                        step, rank
                    )
            except AttributeError:
                pass  # is_completed not available

        # Add noise to dense - forward will read this potentially corrupted data
        batch["dense"] = dense + noise

        # NOTE: We do NOT check for NaN/Inf here to avoid CUDA sync (.item() is blocking!)
        # The NaN check happens in the loss check after forward, which doesn't mask the race
        # because forward has already started by then.

        log.debug(
            "DATADIST RACING: rank=%d step=%d TRUE PARTIAL READ - "
            "reading output_tensor (size=%d) on default_stream while all_to_all writes on datadist_stream",
            rank, step, output_numel
        )
    else:
        log.debug(
            "DATADIST RACING: rank=%d step=%d all_to_all ASYNC (no real dependency)",
            rank, step
        )

    # NO sync between datadist_stream and default_stream!
    # The race window is maximized: all_to_all on datadist_stream races with
    # the noise read above AND the forward pass that follows

    return batch, datadist_stream


def wait_pending_datadist_work() -> None:
    """
    Wait for any pending async datadist work to complete.

    This should be called at the END of each training step to ensure
    the async all_to_all collective completes before the next step.
    This prevents NCCL collective sequence desync between steps.

    The race window is preserved because:
    1. all_to_all launches async on datadist_stream
    2. Forward/backward run on default_stream (racing with datadist_stream)
    3. At end of step, we wait for the collective
    4. Next step starts fresh

    Calling this does NOT eliminate the race - it just ensures NCCL
    stability between steps.
    """
    global _pending_datadist_work, _pending_datadist_tail_work

    if _pending_datadist_work:
        for work in _pending_datadist_work:
            work.wait()
        _pending_datadist_work = []
        _pending_datadist_tail_work = None


def is_datadist_work_pending() -> bool:
    """
    Check if there is pending async datadist work that has NOT yet completed.

    This uses the work handle's is_completed() method to check if the
    all_to_all collective is still in-flight on the GPU.

    IMPORTANT: This is a non-blocking query that does NOT synchronize streams.
    It's safe to call this without affecting the race condition.

    Returns:
        True if there is pending work that has NOT completed yet.
        False if no pending work or work has already completed.
    """
    global _pending_datadist_work

    if not _pending_datadist_work:
        return False

    # Use is_completed() which is a non-blocking query
    # Returns True if the operation has completed, False otherwise
    for work in _pending_datadist_work:
        try:
            if not work.is_completed():
                return True
        except AttributeError:
            # Fallback: some work handles may not have is_completed()
            # In this case, we can't determine status without blocking
            log.debug("DATADIST DIAG: work handle does not support is_completed()")
            return True  # Assume pending to be safe
    return False


def check_nccl_async_behavior(step: int, rank: int) -> Tuple[bool, str]:
    """
    Diagnostic: Check if NCCL is truly running asynchronously.

    This checks if the pending all_to_all work is still in-flight immediately
    after launching. If it's already completed, NCCL may be synchronizing
    internally (which would mask the race condition).

    IMPORTANT: This is non-blocking and does NOT affect the race.

    Args:
        step: Current training step
        rank: Current rank

    Returns:
        Tuple of (is_async, message):
        - is_async: True if NCCL appears to be running asynchronously
        - message: Diagnostic message for logging
    """
    global _pending_datadist_work, _pending_datadist_tail_work, _nccl_sync_warning_logged

    if not _pending_datadist_work:
        return True, "no pending work"

    work = _pending_datadist_tail_work or _pending_datadist_work[-1]
    try:
        is_completed = work.is_completed()
    except AttributeError:
        return True, "work.is_completed() not available"

    if is_completed:
        # Work completed immediately after launch - NCCL may be synchronous!
        msg = (
            f"NCCL SYNC WARNING: all_to_all work.is_completed()=True immediately after launch! "
            f"NCCL may have internal synchronization that prevents race condition."
        )
        if not _nccl_sync_warning_logged:
            log.warning(msg)
            _nccl_sync_warning_logged = True
        return False, "COMPLETED (sync!)"
    else:
        # Work is still pending - NCCL is truly async, race window is open
        return True, "PENDING (async)"


def get_pending_datadist_work() -> List[Any]:
    """
    Get the pending datadist work handle (if any).

    This is useful for external code that needs to query the work handle
    directly for diagnostics.

    Returns:
        A list of pending work handles (empty if none).
    """
    return list(_pending_datadist_work)


def should_skip_datadist_sync(step: int, race_cfg: RaceConfig) -> bool:
    """
    Determine if datadist sync should be skipped (causing race condition).

    When True, FSDP collective will race with all_to_all - THIS IS THE BUG!

    Args:
        step: Current training step
        race_cfg: Race configuration

    Returns:
        True if sync should be skipped (race enabled)
    """
    return (
        race_cfg.datadist_racing
        and race_cfg.datadist_skip_sync_before_collective
        and step >= race_cfg.datadist_racing_start_step
    )
