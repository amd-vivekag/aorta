"""
RCCL Race Condition Reproducer - Standalone test module.

This module provides a minimal, standalone test for detecting runtime-level bugs
in RCCL/HIP that manifest in multi-node distributed training with overlapping streams.

Usage:
    torchrun --nproc_per_node=8 -m aorta.race --warmup 100 --verify 10000
"""

from aorta.race.config import RaceConfig
from aorta.race.minimal_reproducer import (
    ReproducerConfig,
    ReproducerResult,
    MinimalReproducer,
    run_reproducer,
)

__all__ = [
    # Config
    "RaceConfig",
    # Minimal reproducer (runtime bug detection)
    "ReproducerConfig",
    "ReproducerResult",
    "MinimalReproducer",
    "run_reproducer",
]
