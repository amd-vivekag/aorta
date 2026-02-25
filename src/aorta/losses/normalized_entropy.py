"""Normalized Entropy (NE) loss for click/acceptance-rate prediction.

Normalized Entropy is a calibration-aware metric widely used in
recommendation systems (e.g., friend-suggestion acceptance, ad click
prediction). It is defined as:

    NE = average_logloss / background_entropy

where:
    average_logloss = -(1/N) * sum[ y*log(p) + (1-y)*log(1-p) ]
    background_entropy = -( p_bar*log(p_bar) + (1-p_bar)*log(1-p_bar) )
    p_bar = windowed average of positive labels (background CTR)

Interpretation:
    NE = 1.0  ->  model is no better than predicting the background rate
    NE < 1.0  ->  model outperforms the background-rate baseline
    NE > 1.0  ->  model is worse than the naive baseline

The module maintains a sliding window of recent labels to estimate p_bar,
making it suitable for non-stationary click distributions.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _binary_entropy(p: float, eps: float = 1e-7) -> float:
    """Compute binary entropy H(p) = -(p*log(p) + (1-p)*log(1-p))."""
    p = max(eps, min(1.0 - eps, p))
    return -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))


class NormalizedEntropyLoss(nn.Module):
    """Normalized Entropy loss for binary click/accept prediction.

    During training, the forward pass returns a differentiable loss equal to
    BCE divided by the current background entropy estimate. Gradients flow
    through the BCE numerator; the denominator is treated as a (slowly
    changing) constant.

    The background CTR (p_bar) is estimated from a sliding window of recent
    mini-batch label means. This makes the normalization adaptive to
    distribution shifts in the acceptance rate.

    Args:
        window_size: Number of recent mini-batches used to estimate
            the background CTR. Larger windows give a more stable
            estimate; smaller windows react faster to shifts.
        initial_ctr: Initial background CTR used before the window is
            populated. A reasonable default is the expected global
            acceptance rate for friend suggestions (e.g., 0.05 - 0.15).
        eps: Small constant for numerical stability in log and
            entropy computations.
        reduction: ``'mean'`` (default) or ``'none'``.  When ``'mean'``,
            returns a scalar NE loss. When ``'none'``, returns per-sample
            BCE divided by background entropy (useful for weighted losses).
    """

    def __init__(
        self,
        window_size: int = 100,
        initial_ctr: float = 0.1,
        eps: float = 1e-7,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        if not 0.0 < initial_ctr < 1.0:
            raise ValueError(f"initial_ctr must be in (0, 1), got {initial_ctr}")
        if reduction not in ("mean", "none"):
            raise ValueError(f"reduction must be 'mean' or 'none', got {reduction}")

        self.window_size = window_size
        self.eps = eps
        self.reduction = reduction

        # Sliding window of per-batch positive rates (Python floats, not tensors)
        self._label_means: deque[float] = deque(maxlen=window_size)
        self._initial_ctr = initial_ctr

    @property
    def background_ctr(self) -> float:
        """Current estimate of the background click-through rate."""
        if not self._label_means:
            return self._initial_ctr
        return sum(self._label_means) / len(self._label_means)

    @property
    def background_entropy(self) -> float:
        """Binary entropy of the current background CTR."""
        return _binary_entropy(self.background_ctr, self.eps)

    @property
    def normalized_entropy(self) -> Optional[float]:
        """Most recent NE value (available after at least one forward pass)."""
        return getattr(self, "_last_ne", None)

    def reset_window(self) -> None:
        """Clear the sliding window (e.g., at epoch boundaries)."""
        self._label_means.clear()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute Normalized Entropy loss.

        Args:
            logits: Raw model output (pre-sigmoid), shape ``[B, T]`` or
                ``[B]``.
            targets: Binary labels (0 or 1), same shape as logits.
            weight: Optional per-sample importance weights, same shape
                as logits.

        Returns:
            Scalar NE loss (if reduction='mean') or per-element NE
            (if reduction='none').
        """
        targets = targets.to(logits.dtype)

        # --- update sliding window (no grad) ---
        with torch.no_grad():
            batch_positive_rate = targets.mean().item()
            self._label_means.append(batch_positive_rate)

        # --- BCE numerator (differentiable) ---
        if weight is not None:
            bce = F.binary_cross_entropy_with_logits(
                logits, targets, weight=weight, reduction=self.reduction,
            )
        else:
            bce = F.binary_cross_entropy_with_logits(
                logits, targets, reduction=self.reduction,
            )

        # --- background entropy denominator (constant w.r.t. params) ---
        bg_entropy = self.background_entropy
        bg_entropy = max(bg_entropy, self.eps)

        ne = bce / bg_entropy

        # cache scalar NE for logging
        with torch.no_grad():
            self._last_ne = ne.detach().mean().item() if ne.dim() > 0 else ne.detach().item()

        return ne

    def extra_repr(self) -> str:
        return (
            f"window_size={self.window_size}, "
            f"initial_ctr={self._initial_ctr}, "
            f"reduction={self.reduction!r}"
        )


__all__ = ["NormalizedEntropyLoss"]
