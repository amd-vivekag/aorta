"""
Stream conflict injection module.

This module implements synthetic stream conflicts where multiple streams
race with RCCL collectives.

NOTE: This is a synthetic version. For realistic testing, prefer
h2d_memcpy_racing which injects the race during actual data loading.

The race occurs with conflicts between:
1. Memcpy stream: H2D copy on a separate stream
2. Datadist stream: all_to_all simulation on another stream
3. Default stream: RCCL collective (where race occurs)

Using the default stream for memcpy avoids NaN. Putting H2D and data
distribution on the same new stream still causes NaN. This suggests the
issue is about streams not synchronizing with the default stream.
"""

import logging
from typing import Dict

import torch
import torch.distributed as dist

from aorta.race.config import RaceConfig


log = logging.getLogger(__name__)


def inject_stream_conflict(
    device: torch.device,
    batch: Dict[str, torch.Tensor],
    race_cfg: RaceConfig,
    step: int,
    rank: int,
) -> Dict[str, torch.Tensor]:
    """
    Simulate stream conflicts to reproduce NaN race condition.

    This function injects stream conflicts AFTER the batch is already on GPU.
    For realistic testing, prefer h2d_memcpy_racing which races during data loading.

    Args:
        device: CUDA device
        batch: Current batch tensors (already on GPU)
        race_cfg: Race configuration
        step: Current training step
        rank: Current rank

    Returns:
        Modified batch (may have racing tensors)
    """
    if not race_cfg.stream_conflict_test:
        return batch

    if step < race_cfg.stream_conflict_start_step:
        return batch

    # Create separate streams to simulate multi-stream pattern
    memcpy_stream = torch.cuda.Stream(device=device)
    datadist_stream = torch.cuda.Stream(device=device)
    # Note: default_stream is torch.cuda.current_stream() - stream 0

    modified_batch = {}

    # Pattern 1: Memcpy stream racing with default stream
    # This simulates data loading on a separate stream
    if race_cfg.stream_conflict_memcpy_racing:
        with torch.cuda.stream(memcpy_stream):
            for key, tensor in batch.items():
                if isinstance(tensor, torch.Tensor):
                    # Simulate H2D copy by creating a CPU tensor and copying back
                    # This exercises the memcpy path that caused issues
                    cpu_copy = tensor.cpu()
                    # Copy back to device on memcpy stream (racing with default stream)
                    modified_batch[key] = cpu_copy.to(device, non_blocking=True)
                else:
                    modified_batch[key] = tensor
        # DO NOT synchronize memcpy_stream here - let it race with default stream
        log.debug(
            "STREAM CONFLICT: rank=%d step=%d memcpy_stream racing (no sync)",
            rank, step
        )
    else:
        modified_batch = dict(batch)

    # Pattern 2: Datadist stream with all_to_all simulation
    # This simulates data distribution on a separate stream
    if race_cfg.stream_conflict_datadist_racing and dist.is_initialized():
        world_size = dist.get_world_size()
        with torch.cuda.stream(datadist_stream):
            # Simulate data distribution pattern with all_to_all
            # Create a tensor that will be distributed
            sample_tensor = next(iter(modified_batch.values()))
            if isinstance(sample_tensor, torch.Tensor):
                # Create input/output tensors for all_to_all
                # Each rank sends a chunk to every other rank
                chunk_size = sample_tensor.numel() // world_size
                if chunk_size > 0:
                    input_tensor = torch.randn(
                        world_size * chunk_size, device=device, dtype=sample_tensor.dtype
                    )
                    output_tensor = torch.empty_like(input_tensor)

                    # Perform all_to_all on datadist stream (racing with default stream)
                    # This is async because we're on a non-default stream
                    dist.all_to_all_single(
                        output_tensor,
                        input_tensor,
                        async_op=True,  # Async to maximize race potential
                    )
                    log.debug(
                        "STREAM CONFLICT: rank=%d step=%d datadist_stream all_to_all racing (async)",
                        rank, step
                    )
        # DO NOT synchronize datadist_stream here - let it race with default stream

    # Pattern 3: Issue all_reduce on default stream WITHOUT waiting for other streams
    # This is where the race condition manifests - default stream starts collective
    # while memcpy_stream and datadist_stream are still running
    if dist.is_initialized():
        sample_tensor = next(iter(modified_batch.values()))
        if isinstance(sample_tensor, torch.Tensor):
            # All-reduce on default stream - races with memcpy_stream and datadist_stream
            sync_tensor = sample_tensor.clone()
            dist.all_reduce(sync_tensor, op=dist.ReduceOp.SUM)
            log.debug(
                "STREAM CONFLICT: rank=%d step=%d default_stream all_reduce (racing with other streams)",
                rank, step
            )

    return modified_batch
