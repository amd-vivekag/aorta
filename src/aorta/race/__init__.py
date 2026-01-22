"""
Race condition injection module for testing distributed training robustness.

This module provides tools to inject controlled race conditions to simulate
scenarios where H2D memcpy and RCCL collectives race on different GPU streams.

Three categories of race conditions:
1. H2D Race - Races H2D memcpy with forward pass (realistic pattern)
2. Datadist Race - Races all_to_all with FSDP collectives (TorchRec-style)
3. Timing Skew - Controlled delays to demonstrate NaN progression
"""

from aorta.race.config import RaceConfig
from aorta.race.injectors import (
    inject_h2d_racing,
    inject_datadist_racing,
    inject_timing_skew,
    should_skip_h2d_sync,
    should_skip_datadist_sync,
    get_memcpy_stream,
    get_datadist_stream,
    setup_gpu_max_hw_queues,
    check_hw_queues_warning,
    log_race_config_status,
    check_loss_for_nan,
    check_gradients_for_nan,
)

__all__ = [
    "RaceConfig",
    "inject_h2d_racing",
    "inject_datadist_racing",
    "inject_timing_skew",
    "should_skip_h2d_sync",
    "should_skip_datadist_sync",
    "get_memcpy_stream",
    "get_datadist_stream",
    "setup_gpu_max_hw_queues",
    "check_hw_queues_warning",
    "log_race_config_status",
    "check_loss_for_nan",
    "check_gradients_for_nan",
]
