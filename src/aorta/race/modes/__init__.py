"""
Reproducer modes package.

This package contains different distributed training simulation modes:
- default: TorchRec-like pattern with all_to_all + all_reduce
- ddp: DDP pattern with gradient all_reduce and H2D prefetch

Use create_reproducer() factory function to instantiate the appropriate mode.
"""

from typing import TYPE_CHECKING

from ..base import BaseReproducer
from ..config import ReproducerConfig

if TYPE_CHECKING:
    pass


def create_reproducer(
    config: ReproducerConfig, 
    rank: int, 
    world_size: int
) -> BaseReproducer:
    """
    Factory function to create the appropriate reproducer mode.
    
    Args:
        config: Reproducer configuration (mode field determines which class).
        rank: Current process rank.
        world_size: Total number of processes.
        
    Returns:
        BaseReproducer subclass instance for the specified mode.
        
    Raises:
        ValueError: If mode is not recognized.
    """
    # Import here to avoid circular imports
    from .ddp import DDPModeReproducer
    from .default import DefaultModeReproducer
    
    mode = config.mode.lower()
    
    if mode == "ddp":
        return DDPModeReproducer(config, rank, world_size)
    elif mode == "default":
        return DefaultModeReproducer(config, rank, world_size)
    else:
        available = ["default", "ddp"]
        raise ValueError(
            f"Unknown mode: {config.mode}. Available modes: {available}"
        )


# Lazy imports to avoid circular dependencies
def __getattr__(name: str):
    if name == "DefaultModeReproducer":
        from .default import DefaultModeReproducer
        return DefaultModeReproducer
    elif name == "DDPModeReproducer":
        from .ddp import DDPModeReproducer
        return DDPModeReproducer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "create_reproducer",
    "DefaultModeReproducer",
    "DDPModeReproducer",
]
