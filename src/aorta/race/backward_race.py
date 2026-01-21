"""
Backward/clip race injection module.

This module implements race conditions between backward() and clip_grad_norm_().
NOTE: This is a different code path than H2D racing. H2D memcpy racing with
RCCL collectives is in h2d_racing.py.

The backward race affects the gradient flow rather than the input data flow.

The race occurs when:
1. backward() triggers reduce-scatter on the RCCL stream
2. clip_grad_norm_() starts on a fresh stream without waiting
3. Gradient clipping reads incomplete gradients → NaN

Racing ranks: local ranks 1, 2 on Node 0 only (global ranks 1, 2)
"""

import logging
from typing import Optional

import torch
from torch.nn.utils import clip_grad_norm_

from aorta.race.config import RaceConfig


log = logging.getLogger(__name__)


def is_racing_rank(rank: int, gpus_per_node: int = 8) -> bool:
    """
    Determine if this rank should be a racing rank.

    Racing ranks are: local ranks 1, 2 on Node 0 only (global ranks 1, 2)

    Args:
        rank: Global rank
        gpus_per_node: Number of GPUs per node (default 8)

    Returns:
        True if this rank should race
    """
    local_rank = rank % gpus_per_node
    node_id = rank // gpus_per_node
    return node_id == 0 and local_rank in (1, 2)


def get_racing_rank_info(rank: int, gpus_per_node: int = 8) -> tuple:
    """
    Get racing rank information.

    Args:
        rank: Global rank
        gpus_per_node: Number of GPUs per node

    Returns:
        Tuple of (is_racing, node_id, local_rank)
    """
    local_rank = rank % gpus_per_node
    node_id = rank // gpus_per_node
    is_racing = node_id == 0 and local_rank in (1, 2)
    return is_racing, node_id, local_rank


def should_enable_force_async(
    step: int,
    race_cfg: RaceConfig,
    rank: int,
    gpus_per_node: int = 8,
    start_step: int = 3,
) -> bool:
    """
    Determine if force_async should be enabled for reduce-scatter.

    When enabled, reduce-scatter returns before completion, creating a race window.

    Args:
        step: Current training step
        race_cfg: Race configuration
        rank: Global rank
        gpus_per_node: Number of GPUs per node
        start_step: Step to start racing

    Returns:
        True if force_async should be enabled
    """
    if not race_cfg.race_force_async:
        return False
    if step < start_step:
        return False
    return is_racing_rank(rank, gpus_per_node)


def should_sync_after_backward(
    step: int,
    race_cfg: RaceConfig,
    rank: int,
    gpus_per_node: int = 8,
    start_step: int = 3,
) -> bool:
    """
    Determine if non-racing ranks should sync after backward.

    When force_async is enabled, non-racing ranks need to explicitly sync
    to wait for reduce-scatter to complete before gradient clipping.

    Args:
        step: Current training step
        race_cfg: Race configuration
        rank: Global rank
        gpus_per_node: Number of GPUs per node
        start_step: Step to start racing

    Returns:
        True if sync should be performed
    """
    if not race_cfg.race_force_async:
        return False
    if step < start_step:
        return False
    return not is_racing_rank(rank, gpus_per_node)


def clip_gradients_racing(
    model: torch.nn.Module,
    grad_clip_norm: float,
    device: torch.device,
    step: int,
    race_cfg: RaceConfig,
    rank: int,
    gpus_per_node: int = 8,
    start_step: int = 3,
) -> Optional[float]:
    """
    Clip gradients with optional race injection for racing ranks.

    Racing ranks use a fresh CUDA stream with no dependencies, causing
    gradient clipping to race with reduce-scatter.

    Args:
        model: Model with gradients
        grad_clip_norm: Max gradient norm
        device: CUDA device
        step: Current training step
        race_cfg: Race configuration
        rank: Global rank
        gpus_per_node: Number of GPUs per node
        start_step: Step to start racing

    Returns:
        Gradient norm (may be NaN if race condition hit)
    """
    is_racing, node_id, local_rank = get_racing_rank_info(rank, gpus_per_node)
    race_active = (
        step >= start_step
        and race_cfg.race_fresh_stream
        and is_racing
    )

    if race_active:
        # Mechanism 2: Racing ranks use a fresh stream with NO dependencies
        race_stream = torch.cuda.Stream(device=device)
        log.warning(
            "RACE INJECTION: rank=%d (node=%d, local=%d) using FRESH stream - racing",
            rank, node_id, local_rank
        )
        with torch.cuda.stream(race_stream):
            grad_norm = clip_grad_norm_(model.parameters(), grad_clip_norm)
        # Do NOT synchronize race_stream here - let it race
        return grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
    else:
        # Safe path
        return None  # Caller should use normal gradient clipping


def inject_safe_rank_delay(
    device: torch.device,
    step: int,
    race_cfg: RaceConfig,
    rank: int,
    gpus_per_node: int = 8,
    start_step: int = 3,
) -> None:
    """
    Inject artificial GPU delay for safe (non-racing) ranks.

    This widens the race window by slowing down safe ranks, giving
    racing ranks more time to hit the race condition.

    Args:
        device: CUDA device
        step: Current training step
        race_cfg: Race configuration
        rank: Global rank
        gpus_per_node: Number of GPUs per node
        start_step: Step to start delay injection
    """
    is_racing, node_id, local_rank = get_racing_rank_info(rank, gpus_per_node)

    if step >= start_step and race_cfg.race_delay_safe_ranks and not is_racing:
        # Add artificial GPU computation to delay safe ranks
        delay_tensor = torch.randn(1024 * 1024, device=device)
        for _ in range(100):
            delay_tensor = delay_tensor * 1.0001
        torch.cuda.synchronize()
        log.warning(
            "RACE INJECTION: rank=%d (node=%d, local=%d) SYNCHRONIZED with delay - safe",
            rank, node_id, local_rank
        )
