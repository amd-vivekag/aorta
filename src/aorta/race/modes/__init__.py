"""
Reproducer modes package.

This package contains different distributed training simulation modes:
- default: TorchRec-like pattern with all_to_all + all_reduce
- ddp: DDP pattern with gradient all_reduce and H2D prefetch
- fsdp: FSDP pattern with per-layer all_gather + reduce_scatter
- eval_pipelined: Pipelined eval loop for NaN investigation (Experiments A/B)
- stress: All patterns combined with many streams + embedding simulation

Use create_reproducer() factory function to instantiate the appropriate mode.
"""

import logging
from typing import TYPE_CHECKING

from ..base import BaseReproducer
from ..config import ReproducerConfig

if TYPE_CHECKING:
    pass

_log = logging.getLogger(__name__)


def _try_import(registry: dict, mode_name: str, import_fn) -> None:
    """Import a mode class, logging failures instead of silently swallowing them."""
    try:
        cls = import_fn()
        registry[mode_name] = cls
    except ImportError as exc:
        _log.debug("Mode %r unavailable: %s", mode_name, exc)
    except Exception:
        _log.exception("Failed to import mode %r", mode_name)
        raise


def create_reproducer(
    config: ReproducerConfig,
    rank: int,
    world_size: int,
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

    Available modes:
        - "default": TorchRec-like (H2D + all_to_all + all_reduce)
        - "ddp": DDP (H2D prefetch + gradient all_reduce)
        - "fsdp": FSDP (per-layer all_gather + reduce_scatter)
        - "eval_pipelined": Pipelined eval loop (Experiments A/B)
        - "stress": All patterns combined with many streams + embeddings
    """
    def _lazy_registry() -> dict:
        registry: dict = {}
        _try_import(registry, "default",
                    lambda: __import__("aorta.race.modes.default", fromlist=["DefaultModeReproducer"]).DefaultModeReproducer)
        _try_import(registry, "ddp",
                    lambda: __import__("aorta.race.modes.ddp", fromlist=["DDPModeReproducer"]).DDPModeReproducer)
        _try_import(registry, "fsdp",
                    lambda: __import__("aorta.race.modes.fsdp", fromlist=["FSDPModeReproducer"]).FSDPModeReproducer)
        _try_import(registry, "eval_pipelined",
                    lambda: __import__("aorta.race.modes.eval_pipelined", fromlist=["EvalPipelinedReproducer"]).EvalPipelinedReproducer)
        _try_import(registry, "stress",
                    lambda: __import__("aorta.race.modes.stress", fromlist=["StressModeReproducer"]).StressModeReproducer)
        return registry

    MODE_REGISTRY = _lazy_registry()

    mode = config.mode.lower()

    if mode not in MODE_REGISTRY:
        available = list(MODE_REGISTRY.keys())
        raise ValueError(
            f"Unknown mode: {config.mode}. Available modes: {available}"
        )

    return MODE_REGISTRY[mode](config, rank, world_size)


# Lazy imports to avoid circular dependencies
def __getattr__(name: str):
    if name == "DefaultModeReproducer":
        from .default import DefaultModeReproducer
        return DefaultModeReproducer
    elif name == "DDPModeReproducer":
        from .ddp import DDPModeReproducer
        return DDPModeReproducer
    elif name == "FSDPModeReproducer":
        from .fsdp import FSDPModeReproducer
        return FSDPModeReproducer
    elif name == "EvalPipelinedReproducer":
        from .eval_pipelined import EvalPipelinedReproducer
        return EvalPipelinedReproducer
    elif name == "StressModeReproducer":
        from .stress import StressModeReproducer
        return StressModeReproducer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "create_reproducer",
    "DefaultModeReproducer",
    "DDPModeReproducer",
    "FSDPModeReproducer",
    "EvalPipelinedReproducer",
    "StressModeReproducer",
]
