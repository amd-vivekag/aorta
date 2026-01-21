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


def get_memcpy_stream(device: torch.device) -> torch.cuda.Stream:
    """
    Get or create the memcpy stream for a device.

    The memcpy stream is used for H2D copies in the racing pattern.
    It's created lazily and cached per device.

    Args:
        device: CUDA device

    Returns:
        CUDA stream for memcpy operations
    """
    if device not in _memcpy_streams:
        _memcpy_streams[device] = torch.cuda.Stream(device=device)
    return _memcpy_streams[device]


def clear_memcpy_streams() -> None:
    """Clear all cached memcpy streams (for testing)."""
    _memcpy_streams.clear()


def move_batch_to_device(
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Move batch to device using default stream (safe, no race).

    Args:
        batch: CPU batch tensors
        device: Target CUDA device

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
        device: Target CUDA device
        step: Current training step
        race_cfg: Race configuration
        rank: Current rank

    Returns:
        Tuple of (batch tensors on GPU, memcpy_stream or None if not racing)
        The memcpy_stream is returned so caller can decide whether to sync
    """
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
                result[key] = tensor.to(device, non_blocking=True)
            else:
                result[key] = tensor

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
