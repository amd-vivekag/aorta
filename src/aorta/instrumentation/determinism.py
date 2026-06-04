"""Best-effort PyTorch determinism setup for replay-style workloads.

Not all kernels have deterministic implementations; ``warn_only=True``
lets a workload run instead of hard-failing on the first
nondeterministic op. The companion checksum compare in
:mod:`aorta.instrumentation.checksum` is what actually proves bitwise
equality between two replays.

Does NOT touch ``torch.compile`` or HIP graphs — the llm_determinism
recipe explicitly avoids both at the call site and we don't want this
helper silently re-enabling them anywhere it's reused.
"""

from __future__ import annotations

import os
import random

import torch

CUBLAS_WORKSPACE_ENV = "CUBLAS_WORKSPACE_CONFIG"
CUBLAS_WORKSPACE_VALUE = ":4096:8"


def enable_deterministic(seed: int) -> None:
    """Seed RNGs and flip PyTorch into deterministic mode.

    Sets ``CUBLAS_WORKSPACE_CONFIG`` if unset — cuBLAS/hipBLAS require it
    before the first CUDA context, otherwise ``use_deterministic_algorithms``
    raises at the first matmul.

    Seeds only the CPU + Python RNGs here. The per-rank CUDA device must
    be seeded by the caller AFTER ``torch.cuda.set_device(LOCAL_RANK)``,
    using ``torch.cuda.manual_seed`` (NOT ``manual_seed_all``) so we don't
    initialize CUDA contexts on every visible GPU in every torchrun rank.
    """
    os.environ.setdefault(CUBLAS_WORKSPACE_ENV, CUBLAS_WORKSPACE_VALUE)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


__all__ = ["CUBLAS_WORKSPACE_ENV", "CUBLAS_WORKSPACE_VALUE", "enable_deterministic"]
