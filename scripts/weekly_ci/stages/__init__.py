"""
Pipeline stages for Weekly CI Kickoff.

This module exports all pipeline stage functions.
"""

from __future__ import annotations

from .analysis import (
    stage_compare_all_analysis,
    stage_cross_timestamp_comparison,
    stage_pairwise_analysis,
)
from .build import (
    stage_build_rccl,
    stage_install_dependencies,
    verify_rccl_installation,
)
from .docker import stage_cleanup, stage_docker_setup
from .test import (
    stage_find_baseline_experiment_dir,
    stage_find_experiment_dir,
    stage_run_performance_tests,
    validate_experiment_configs,
)
from .validate import stage_validate_environment

__all__ = [
    # Validation
    "stage_validate_environment",
    # Docker
    "stage_docker_setup",
    "stage_cleanup",
    # Build
    "stage_build_rccl",
    "stage_install_dependencies",
    "verify_rccl_installation",
    # Test
    "stage_run_performance_tests",
    "stage_find_experiment_dir",
    "stage_find_baseline_experiment_dir",
    "validate_experiment_configs",
    # Analysis
    "stage_pairwise_analysis",
    "stage_compare_all_analysis",
    "stage_cross_timestamp_comparison",
]
