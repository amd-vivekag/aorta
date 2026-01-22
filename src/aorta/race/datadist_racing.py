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
"""

import logging
from typing import Dict, Tuple, Optional

import torch
import torch.distributed as dist

from aorta.race.config import RaceConfig


log = logging.getLogger(__name__)


# Global datadist stream for racing (created lazily per device)
_datadist_streams: Dict[torch.device, torch.cuda.Stream] = {}


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

    IMPORTANT: We use async_op=False because even in racing mode, we need all
    ranks to complete the collective before proceeding. The race window is
    created by the separate stream + optional sync skip, NOT by dangling work
    handles which would cause NCCL deadlocks.

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

    # Perform all_to_all on datadist_stream to simulate TorchRec pattern
    with torch.cuda.stream(datadist_stream):
        # Find a suitable tensor from the batch to simulate data distribution
        sample_tensor = None
        for key, tensor in batch.items():
            if isinstance(tensor, torch.Tensor) and tensor.numel() > 0:
                sample_tensor = tensor
                break

        if sample_tensor is not None:
            # Create input/output tensors for all_to_all
            # Each rank sends a chunk to every other rank
            numel = sample_tensor.numel()
            chunk_size = max(1, numel // world_size)
            total_size = chunk_size * world_size

            # Create tensors for all_to_all
            input_tensor = torch.randn(
                total_size, device=device, dtype=sample_tensor.dtype
            )
            output_tensor = torch.empty_like(input_tensor)

            # Perform all_to_all on datadist stream (racing with default stream)
            # IMPORTANT: Use async_op=False to ensure the collective completes
            # within this stream context. The race condition is created by:
            # 1. Running on a separate stream (datadist_stream)
            # 2. Optionally skipping sync before FSDP collective
            #
            # Using async_op=True with a discarded work handle causes NCCL
            # deadlocks because ranks get out of sync in the collective sequence.
            dist.all_to_all_single(
                output_tensor,
                input_tensor,
                async_op=False,  # Must complete within stream to avoid NCCL deadlock
            )

            log.debug(
                "DATADIST RACING: rank=%d step=%d all_to_all on datadist_stream (stream NOT synced with default)",
                rank, step
            )

    # CRITICAL: We intentionally DO NOT synchronize datadist_stream with default stream here
    # The caller may or may not wait for datadist_stream before FSDP collective
    # If datadist_skip_sync_before_collective=True, default stream will race with all_to_all
    # The race condition is: FSDP collective on default stream may start while
    # datadist_stream's all_to_all is still in progress

    return batch, datadist_stream


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
