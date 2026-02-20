"""
Pipeline stages for Weekly CI Kickoff.

This module exports all pipeline stage functions.
"""

from __future__ import annotations

from .build import (
    stage_build_rccl,
    stage_install_dependencies,
    verify_rccl_installation,
)
from .docker import stage_cleanup, stage_docker_setup
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
]
