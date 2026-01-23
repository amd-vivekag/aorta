"""
H2D (Host-to-Device) memcpy racing module.

This module implements H2D racing patterns where the race occurs when:

1. H2D memcpy happens on memcpy_stream (not default stream)
2. Forward pass starts before H2D completes
3. RCCL collective reads potentially incomplete data

Using the default stream for memcpy avoids this race. This module
intentionally uses a separate stream to simulate the race condition.
"""

import logging
from typing import Dict, Tuple, Optional

import torch

from aorta.race.config import RaceConfig


log = logging.getLogger(__name__)


# Global memcpy stream for H2D racing (created lazily per device)
_memcpy_streams: Dict[torch.device, torch.cuda.Stream] = {}

# Global H2D completion event for diagnostics
_h2d_completion_event: Optional[torch.cuda.Event] = None

# Per-tensor completion events for targeted diagnostics
_h2d_tensor_events: Dict[str, torch.cuda.Event] = {}


def get_memcpy_stream(device: torch.device) -> torch.cuda.Stream:
    """
    Get or create the memcpy stream for a device.

    The memcpy stream is used for H2D copies in the racing pattern.
    It's created lazily and cached per device.

    Args:
        device: GPU device

    Returns:
        GPU stream for memcpy operations
    """
    if device not in _memcpy_streams:
        _memcpy_streams[device] = torch.cuda.Stream(device=device)
    return _memcpy_streams[device]


def move_batch_to_device(
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Move batch to device using default stream (safe, no race).

    Args:
        batch: CPU batch tensors
        device: Target GPU device

    Returns:
        Batch tensors on GPU
    """
    return {
        key: tensor.to(device, non_blocking=True) if isinstance(tensor, torch.Tensor) else tensor
        for key, tensor in batch.items()
    }


def move_batch_to_device_racing(
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    step: int,
    race_cfg: RaceConfig,
    rank: int,
) -> Tuple[Dict[str, torch.Tensor], Optional[torch.cuda.Stream]]:
    """
    Move batch to device using separate memcpy stream (racing pattern).

    This simulates a race condition where:
    1. H2D memcpy happens on memcpy_stream (not default stream)
    2. Forward pass may start before H2D completes
    3. RCCL collective reads potentially incomplete/torn data

    Using the default stream for memcpy avoids this race.
    This function intentionally uses a separate stream to reproduce the issue.

    Args:
        batch: CPU batch tensors
        device: Target GPU device
        step: Current training step
        race_cfg: Race configuration
        rank: Current rank

    Returns:
        Tuple of (batch tensors on GPU, memcpy_stream or None if not racing)
        The memcpy_stream is returned so caller can decide whether to sync
    """
    global _h2d_completion_event

    if not race_cfg.h2d_memcpy_racing:
        # Normal path: use default stream
        return move_batch_to_device(batch, device), None

    if step < race_cfg.h2d_racing_start_step:
        # Before racing starts: use default stream
        return move_batch_to_device(batch, device), None

    # Racing path: use separate memcpy stream
    memcpy_stream = get_memcpy_stream(device)

    result = {}
    with torch.cuda.stream(memcpy_stream):
        for key, tensor in batch.items():
            if isinstance(tensor, torch.Tensor):
                # H2D copy on memcpy_stream (NOT default stream)
                # This creates potential for torn reads if forward starts
                # before this copy completes
                if key == "dense" and race_cfg.h2d_split_dense_copy and tensor.numel() > 1:
                    # Copy dense in two chunks so the tail is copied last.
                    dest = torch.empty_like(tensor, device=device)
                    src_flat = tensor.reshape(-1)
                    dest_flat = dest.reshape(-1)
                    total = dest_flat.numel()
                    tail_frac = max(0.0, min(1.0, race_cfg.h2d_dense_tail_fraction))
                    split_idx = int(total * (1.0 - tail_frac))
                    split_idx = max(1, min(total - 1, split_idx))
                    dest_flat[:split_idx].copy_(src_flat[:split_idx], non_blocking=True)
                    dest_flat[split_idx:].copy_(src_flat[split_idx:], non_blocking=True)
                    result[key] = dest
                    evt = torch.cuda.Event()
                    evt.record()
                    _h2d_tensor_events[key] = evt
                else:
                    result[key] = tensor.to(device, non_blocking=True)
                    if key == "dense":
                        evt = torch.cuda.Event()
                        evt.record()
                        _h2d_tensor_events[key] = evt
            else:
                result[key] = tensor

        # Record completion event on memcpy_stream AFTER all H2D copies
        # This can be queried later to check if H2D is still in-flight
        _h2d_completion_event = torch.cuda.Event()
        _h2d_completion_event.record()

    # CRITICAL: We intentionally DO NOT synchronize memcpy_stream here
    # The caller may or may not wait for memcpy_stream before forward pass
    # If h2d_skip_sync_before_forward=True, forward will race with H2D
    # This can result in torn reads where some data is still being copied

    log.debug(
        "H2D RACING: rank=%d step=%d batch on memcpy_stream (NOT synced)",
        rank, step
    )

    return result, memcpy_stream


def should_skip_h2d_sync(step: int, race_cfg: RaceConfig) -> bool:
    """
    Determine if H2D sync should be skipped (causing race condition).

    When True, forward pass will race with H2D copy - THIS IS THE BUG!

    Args:
        step: Current training step
        race_cfg: Race configuration

    Returns:
        True if sync should be skipped (race enabled)
    """
    return (
        race_cfg.h2d_memcpy_racing
        and race_cfg.h2d_skip_sync_before_forward
        and step >= race_cfg.h2d_racing_start_step
    )


def is_h2d_still_in_flight() -> bool:
    """
    Check if H2D copy is still in-flight (not completed).

    This uses the completion event recorded after all H2D copies to check
    if the copies have finished. This is a non-blocking query.

    Returns:
        True if H2D is still in-flight (not completed).
        False if H2D has completed or no H2D event was recorded.
    """
    global _h2d_completion_event

    if _h2d_completion_event is None:
        return False

    # query() returns True if the event has been recorded AND completed
    return not _h2d_completion_event.query()


def is_h2d_tensor_in_flight(key: str) -> bool:
    """
    Check if a specific H2D tensor copy is still in-flight.

    This uses a per-tensor completion event recorded on the memcpy stream.
    It is a non-blocking query.
    """
    evt = _h2d_tensor_events.get(key)
    if evt is None:
        return False
    return not evt.query()


def check_h2d_race_status(step: int, rank: int, context: str = "") -> Tuple[bool, str]:
    """
    Diagnostic: Check if H2D is still in-flight (race window open).

    Args:
        step: Current training step
        rank: Current rank
        context: Context string for logging (e.g., "before forward")

    Returns:
        Tuple of (is_racing, message)
    """
    global _h2d_completion_event

    if _h2d_completion_event is None:
        return False, "no H2D event recorded"

    is_in_flight = not _h2d_completion_event.query()

    if is_in_flight:
        msg = f"H2D still IN-FLIGHT {context} - RACE POSSIBLE!"
        log.info("H2D_RACE: step=%d rank=%d %s", step, rank, msg)
        return True, "IN-FLIGHT (race!)"
    else:
        msg = f"H2D COMPLETED {context} - no race at this point"
        log.debug("H2D_RACE: step=%d rank=%d %s", step, rank, msg)
        return False, "COMPLETED (no race)"


