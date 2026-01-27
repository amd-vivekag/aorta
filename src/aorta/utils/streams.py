"""CUDA/HIP stream management utilities.

This module provides utilities for:
- Stream creation and management
- Stream synchronization
- GPU warmup operations
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Generator, List, Optional, Tuple

import torch


def create_streams(
    count: int, device: str = "cuda:0", priorities: Optional[List[int]] = None
) -> List[torch.cuda.Stream]:
    """Create a list of CUDA/HIP streams.

    Args:
        count: Number of streams to create
        device: Target device
        priorities: Optional list of priorities for each stream.
                   Lower values = higher priority. If None, default priority is used.

    Returns:
        List of torch.cuda.Stream objects
    """
    device_obj = torch.device(device)

    if priorities is not None and len(priorities) != count:
        raise ValueError(f"priorities length ({len(priorities)}) must match count ({count})")

    streams = []
    for i in range(count):
        priority = priorities[i] if priorities is not None else 0
        # Note: PyTorch stream priority support is limited
        # On ROCm, this maps to HIP stream priorities
        stream = torch.cuda.Stream(device=device_obj, priority=priority)
        streams.append(stream)

    return streams


def sync_all_streams(streams: List[torch.cuda.Stream]) -> None:
    """Synchronize all streams.

    Args:
        streams: List of streams to synchronize
    """
    for stream in streams:
        stream.synchronize()


def sync_stream(stream: torch.cuda.Stream) -> None:
    """Synchronize a single stream.

    Args:
        stream: Stream to synchronize
    """
    stream.synchronize()


@contextmanager
def cuda_stream_context(stream: torch.cuda.Stream) -> Generator[torch.cuda.Stream, None, None]:
    """Context manager that sets the current CUDA stream.

    Args:
        stream: Stream to use as current

    Yields:
        The stream
    """
    with torch.cuda.stream(stream):
        yield stream


def get_stream_id(stream: torch.cuda.Stream) -> int:
    """Get the underlying stream ID (pointer/handle).

    This can be useful for debugging and correlating with profiler output.

    Args:
        stream: PyTorch CUDA stream

    Returns:
        Integer representation of the stream handle
    """
    return stream.cuda_stream


def warmup_gpu(device: str = "cuda:0", iterations: int = 10) -> None:
    """Warm up the GPU with simple operations.

    This helps ensure consistent timing by:
    - Triggering GPU frequency scaling
    - Warming up the memory subsystem
    - Initializing any lazy GPU state

    Args:
        device: Device to warm up
        iterations: Number of warmup iterations
    """
    device_obj = torch.device(device)

    # Create some tensors and do operations
    for _ in range(iterations):
        a = torch.randn(1024, 1024, device=device_obj)
        b = torch.randn(1024, 1024, device=device_obj)
        c = torch.mm(a, b)
        torch.cuda.synchronize(device_obj)

    # Clean up
    del a, b, c
    torch.cuda.empty_cache()


def get_available_devices() -> List[str]:
    """Get a list of all available CUDA/HIP devices.

    Returns:
        List of device strings (e.g., ["cuda:0", "cuda:1", ...])
    """
    if not torch.cuda.is_available():
        return []
    return [f"cuda:{i}" for i in range(torch.cuda.device_count())]


def create_multi_gpu_streams(
    total_stream_count: int,
    devices: Optional[List[str]] = None,
    priorities: Optional[List[int]] = None,
) -> Tuple[List[torch.cuda.Stream], Dict[int, str]]:
    """Create streams distributed across multiple GPUs with round-robin assignment.

    Each stream is created on its assigned GPU device. The streams are distributed
    in round-robin fashion across the available GPUs.

    Args:
        total_stream_count: Total number of streams to create
        devices: Optional list of devices to use. If None, uses all available GPUs.
        priorities: Optional list of priorities for each stream.
                   Lower values = higher priority. If None, default priority is used.

    Returns:
        Tuple of:
            - List of torch.cuda.Stream objects (flat list, ordered by stream index)
            - Dict mapping stream index to device string

    Example:
        >>> streams, stream_to_device = create_multi_gpu_streams(8)
        >>> # With 2 GPUs: streams 0,2,4,6 on cuda:0; streams 1,3,5,7 on cuda:1
        >>> stream_to_device[0]
        'cuda:0'
        >>> stream_to_device[1]
        'cuda:1'
    """
    if devices is None:
        devices = get_available_devices()
        if not devices:
            # Fallback to cuda:0 if no devices detected
            devices = ["cuda:0"]

    if not devices:
        raise ValueError("No CUDA devices available")

    if priorities is not None and len(priorities) != total_stream_count:
        raise ValueError(
            f"priorities length ({len(priorities)}) must match total_stream_count ({total_stream_count})"
        )

    num_devices = len(devices)
    streams: List[torch.cuda.Stream] = []
    stream_to_device: Dict[int, str] = {}

    for stream_idx in range(total_stream_count):
        # Round-robin device assignment
        device_idx = stream_idx % num_devices
        device = devices[device_idx]
        device_obj = torch.device(device)

        priority = priorities[stream_idx] if priorities is not None else 0
        stream = torch.cuda.Stream(device=device_obj, priority=priority)

        streams.append(stream)
        stream_to_device[stream_idx] = device

    return streams, stream_to_device


def create_streams_per_device(
    streams_per_device: int,
    devices: Optional[List[str]] = None,
    priorities: Optional[List[int]] = None,
) -> Dict[str, List[torch.cuda.Stream]]:
    """Create a pool of streams for each GPU device.

    Unlike create_multi_gpu_streams which creates a flat list with round-robin
    assignment, this creates separate stream pools for each device.

    Args:
        streams_per_device: Number of streams to create per device
        devices: Optional list of devices to use. If None, uses all available GPUs.
        priorities: Optional list of priorities (applied to each device's streams).
                   Length must match streams_per_device.

    Returns:
        Dict mapping device string to list of streams for that device

    Example:
        >>> device_streams = create_streams_per_device(4)
        >>> # With 2 GPUs: {"cuda:0": [s0,s1,s2,s3], "cuda:1": [s0,s1,s2,s3]}
    """
    if devices is None:
        devices = get_available_devices()
        if not devices:
            devices = ["cuda:0"]

    if priorities is not None and len(priorities) != streams_per_device:
        raise ValueError(
            f"priorities length ({len(priorities)}) must match streams_per_device ({streams_per_device})"
        )

    device_streams: Dict[str, List[torch.cuda.Stream]] = {}

    for device in devices:
        device_streams[device] = create_streams(
            count=streams_per_device,
            device=device,
            priorities=priorities,
        )

    return device_streams


def warmup_all_gpus(devices: Optional[List[str]] = None, iterations: int = 10) -> None:
    """Warm up all available GPUs.

    Args:
        devices: Optional list of devices to warm up. If None, warms up all available.
        iterations: Number of warmup iterations per device
    """
    if devices is None:
        devices = get_available_devices()
        if not devices:
            devices = ["cuda:0"]

    for device in devices:
        warmup_gpu(device=device, iterations=iterations)


__all__ = [
    "create_streams",
    "create_multi_gpu_streams",
    "create_streams_per_device",
    "get_available_devices",
    "sync_all_streams",
    "sync_stream",
    "cuda_stream_context",
    "get_stream_id",
    "warmup_gpu",
    "warmup_all_gpus",
]
