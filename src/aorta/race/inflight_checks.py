"""
In-flight read instability checks.

This module provides utilities for detecting torn reads by performing
repeated reads on tensor regions that may be actively being written.
If values change between reads, it indicates a race condition.

These checks do NOT add any cross-stream synchronization - they only
issue reads on the current stream (typically default stream) while
the racing stream may still be writing.
"""

import logging
from typing import List, Dict, Any

import torch

log = logging.getLogger(__name__)


# Pending check results: list of dicts with name, diff, max_diff, step, rank, etc.
_pending_inflight_checks: List[Dict[str, Any]] = []


def schedule_inflight_check(
    name: str,
    tensor: torch.Tensor,
    sample_size: int,
    repeats: int,
    step: int,
    rank: int,
) -> None:
    """
    Schedule repeated reads on a tensor tail to detect instability.

    This reads a small sample from the tensor multiple times on the default
    stream while the racing stream may still be writing. If values change
    between reads, it indicates a torn read (race condition).

    IMPORTANT: This does NOT add any cross-stream synchronization. The reads
    are issued on the default stream which is already racing with the write.

    Args:
        name: Identifier for this check (e.g., "h2d_dense", "datadist_tail")
        tensor: The tensor being read (should be the tail region)
        sample_size: Number of elements to sample from the tensor
        repeats: Number of repeated reads to perform
        step: Current training step
        rank: Current rank
    """
    global _pending_inflight_checks

    if repeats <= 0 or tensor.numel() == 0:
        return

    # Sample from the start of the tensor (which is the tail region)
    actual_sample_size = min(sample_size, tensor.numel())
    sample_slice = tensor.reshape(-1)[:actual_sample_size]

    # First read: capture reference sample (on default stream)
    sample0 = sample_slice.clone()

    # Track if any difference is detected
    diff_detected = torch.zeros(1, dtype=torch.bool, device=tensor.device)
    max_abs_diff = torch.zeros(1, dtype=tensor.dtype, device=tensor.device)

    # Repeated reads: check for instability
    for _ in range(repeats):
        # Read the same slice again
        current_sample = sample_slice.clone()

        # Check for any difference (bitwise comparison)
        diff = (current_sample != sample0).any()
        diff_detected = diff_detected | diff

        # Track maximum absolute difference
        if tensor.is_floating_point():
            abs_diff = (current_sample - sample0).abs().max()
            max_abs_diff = torch.maximum(max_abs_diff, abs_diff)

    # Store results for later logging (after race window closes)
    _pending_inflight_checks.append({
        "name": name,
        "diff": diff_detected,
        "max_diff": max_abs_diff,
        "step": step,
        "rank": rank,
        "sample_size": actual_sample_size,
        "repeats": repeats,
    })


def flush_inflight_checks(step: int, rank: int) -> int:
    """
    Flush and log any pending in-flight check results.

    This should be called at the end of each training step, after the
    race window has closed (e.g., after wait_pending_datadist_work()).
    It performs a minimal sync to read the result tensors and logs any
    detected mismatches.

    Args:
        step: Current training step
        rank: Current rank

    Returns:
        Number of mismatches detected
    """
    global _pending_inflight_checks

    if not _pending_inflight_checks:
        return 0

    mismatch_count = 0

    for check in _pending_inflight_checks:
        # Sync to read the boolean result (minimal overhead)
        diff_val = check["diff"].item()
        max_diff_val = check["max_diff"].item() if check["max_diff"].numel() > 0 else 0.0

        if diff_val:
            mismatch_count += 1
            log.info(
                "INFLIGHT_MISMATCH: step=%d rank=%d path=%s diff=True max_abs_diff=%.6e "
                "sample_size=%d repeats=%d - TORN READ DETECTED!",
                check["step"], check["rank"], check["name"],
                max_diff_val, check["sample_size"], check["repeats"]
            )
        else:
            log.debug(
                "INFLIGHT_CHECK: step=%d rank=%d path=%s diff=False (no instability detected)",
                check["step"], check["rank"], check["name"]
            )

    # Clear pending checks
    _pending_inflight_checks = []

    return mismatch_count


def clear_inflight_checks() -> None:
    """Clear any pending in-flight checks without logging."""
    global _pending_inflight_checks
    _pending_inflight_checks = []
