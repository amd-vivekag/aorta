"""Utility helpers for the AORTA toolkit.

This module provides shared utilities for:
- Device discovery and GPU properties
- Stream creation and management
- GPU timing utilities
- Configuration loading
- Logging setup

Torch-dependent symbols (device, distributed, streams, timing) are imported
lazily so that ``aorta.utils.config`` and ``aorta.utils.logging`` remain
usable in environments where PyTorch is not installed.
"""

from .config import load_config, merge_cli_overrides
from .logging import setup_logging

def __getattr__(name: str):
    """Lazy-load torch-dependent symbols on first access."""
    _distributed_names = {
        "cleanup_distributed", "create_process_groups", "get_local_rank",
        "get_rank", "get_world_size", "init_distributed", "is_distributed",
        "parse_process_groups",
    }
    _device_names = {
        "BACKEND_NAME", "IS_ROCM", "Accelerator", "DeviceProperties",
        "detect_accelerator", "ensure_gpu_available", "get_device",
        "get_device_properties", "get_distributed_backend", "get_driver_info",
        "get_memory_stats", "get_rocm_env_info", "get_system_info",
        "log_environment_info", "reset_memory_stats",
    }
    _streams_names = {
        "create_multi_gpu_streams", "create_streams", "create_streams_per_device",
        "cuda_stream_context", "get_available_devices", "get_stream_id",
        "sync_all_streams", "sync_stream", "warmup_all_gpus", "warmup_gpu",
    }
    _timing_names = {
        "CPUTimer", "EventTiming", "StreamTimer", "TimingContext",
    }

    if name in _distributed_names:
        from . import distributed
        return getattr(distributed, name)
    if name in _device_names:
        from . import device
        return getattr(device, name)
    if name in _streams_names:
        from . import streams
        return getattr(streams, name)
    if name in _timing_names:
        from . import timing
        return getattr(timing, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Config
    "load_config",
    "merge_cli_overrides",
    # Distributed
    "init_distributed",
    "cleanup_distributed",
    "is_distributed",
    "get_rank",
    "get_world_size",
    "get_local_rank",
    "parse_process_groups",
    "create_process_groups",
    # Logging
    "setup_logging",
    # Device - constants
    "IS_ROCM",
    "BACKEND_NAME",
    "Accelerator",
    # Device - classes
    "DeviceProperties",
    # Device - functions (original aorta)
    "detect_accelerator",
    "get_device",
    "get_distributed_backend",
    # Device - functions (extended)
    "get_device_properties",
    "ensure_gpu_available",
    "get_memory_stats",
    "reset_memory_stats",
    "get_rocm_env_info",
    "get_driver_info",
    "get_system_info",
    "log_environment_info",
    # Streams
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
    # Timing
    "EventTiming",
    "TimingContext",
    "StreamTimer",
    "CPUTimer",
]
