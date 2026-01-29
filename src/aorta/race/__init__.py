"""
Race condition injection module for testing distributed training robustness.

This module provides tools to inject controlled race conditions to simulate
scenarios where H2D memcpy and RCCL collectives race on different GPU streams.

Four categories of functionality:

1. H2D Race - Races H2D memcpy with forward pass (realistic pattern)
2. Datadist Race - Races all_to_all with FSDP collectives (TorchRec-style)
3. Timing Skew - Controlled delays to demonstrate NaN progression
4. Correctness Verification - Detect silent corruption without NaN
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
from aorta.race.correctness_verification import (
    VerificationResult,
    StabilityReport,
    set_deterministic_mode,
    run_reference_step,
    verify_deterministic_comparison,
    compute_variance_over_runs,
    verify_variance,
    track_gradient_norms,
    verify_gradient_norm_outliers,
    run_all_verifications,
    check_determinism,
    check_gradient_health,
    check_loss_health,
    verify_numerical_stability,
)
from aorta.race.minimal_reproducer import (
    ReproducerConfig,
    ReproducerResult,
    MinimalReproducer,
    run_reproducer,
)

__all__ = [
    # Config
    "RaceConfig",
    # Race injection
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
    # Correctness verification
    "VerificationResult",
    "StabilityReport",
    "set_deterministic_mode",
    "run_reference_step",
    "verify_deterministic_comparison",
    "compute_variance_over_runs",
    "verify_variance",
    "track_gradient_norms",
    "verify_gradient_norm_outliers",
    "run_all_verifications",
    # Numerical stability
    "check_determinism",
    "check_gradient_health",
    "check_loss_health",
    "verify_numerical_stability",
    # Minimal reproducer (runtime bug detection)
    "ReproducerConfig",
    "ReproducerResult",
    "MinimalReproducer",
    "run_reproducer",
]
