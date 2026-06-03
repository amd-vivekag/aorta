"""Bit-exact tensor checksums for nondeterminism detection.

The checksum is a **bit-pattern** sum, not a numeric sum: tensor storage is
re-interpreted (``view``, not ``to``) as the signed integer of matching
element size, then accumulated into ``int64``. Two tensors with identical
bits produce identical checksums; two tensors that differ in a single bit
(including NaN payload bits and the +0 vs -0 distinction that a numeric
sum would erase) produce different checksums.

Use for war-room replay checks where the question is "did this kernel
produce the same bytes twice", not "are the values numerically close".
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# Each floating dtype maps to the signed integer of the same element size.
# Lookup by dtype handles bf16/fp16 (both 2-byte) -> int16, fp32 -> int32, etc.
_VIEW_DTYPE: dict[torch.dtype, torch.dtype] = {
    torch.bfloat16: torch.int16,
    torch.float16: torch.int16,
    torch.float32: torch.int32,
    torch.float64: torch.int64,
    torch.int8: torch.int8,
    torch.int16: torch.int16,
    torch.int32: torch.int32,
    torch.int64: torch.int64,
    torch.bool: torch.int8,
}


def tensor_checksum(t: torch.Tensor) -> int:
    """Return a bit-exact int64 checksum of ``t``'s storage.

    Casts the dtype with ``view`` (zero-copy bit reinterpretation), not
    ``to`` (numeric conversion). The accumulator is int64 so the sum
    wraps modulo 2**64 deterministically — wraparound is fine because we
    only ever compare two checksums, never reason about the magnitude.
    """
    if t.numel() == 0:
        return 0
    view_dtype = _VIEW_DTYPE.get(t.dtype)
    if view_dtype is None:
        raise TypeError(f"tensor_checksum: unsupported dtype {t.dtype}")
    flat = t.detach().contiguous().view(-1)
    as_int = flat.view(view_dtype) if view_dtype != flat.dtype else flat
    return int(as_int.to(torch.int64).sum().item())


def state_checksum(named: dict[str, torch.Tensor]) -> dict[str, int]:
    """Per-tensor checksums for a name->tensor mapping (e.g. ``state_dict``)."""
    return {name: tensor_checksum(t) for name, t in named.items()}


def global_checksum(local: int, group: object | None = None) -> int:
    """All-reduce a local checksum into a rank-agnostic global checksum.

    Sum-reduces in int64 so the result is order-independent across ranks.
    Returns the local value unchanged when torch.distributed isn't
    initialised, which keeps single-process callers working.
    """
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return local
    buf = torch.tensor([local], dtype=torch.int64)
    if torch.cuda.is_available():
        # current_device() respects LOCAL_RANK; the default `.cuda()` would
        # land on cuda:0 on every rank, which NCCL rejects on multi-GPU runs.
        buf = buf.to(torch.device("cuda", torch.cuda.current_device()))
    dist.all_reduce(buf, op=dist.ReduceOp.SUM, group=group)
    return int(buf.item())


@dataclass
class ChecksumSet:
    """Loss + grad + output checksums for one replay of one step.

    Fields are plain ints so ``a == b`` is the entire comparison the
    caller needs. ``grads`` and ``params`` are per-tensor so a divergence
    report can name *which* parameter drifted.
    """

    loss_bits: int
    output_bits: int
    grads: dict[str, int]
    params: dict[str, int]
    global_bits: int | None = None


def compare(a: ChecksumSet, b: ChecksumSet) -> list[str]:
    """Return a list of human-readable divergence reasons; empty == match."""
    reasons: list[str] = []
    if a.loss_bits != b.loss_bits:
        reasons.append(f"loss_bits {a.loss_bits} != {b.loss_bits}")
    if a.output_bits != b.output_bits:
        reasons.append(f"output_bits {a.output_bits} != {b.output_bits}")
    if a.global_bits != b.global_bits:
        reasons.append(f"global_bits {a.global_bits} != {b.global_bits}")
    for name in sorted(set(a.grads) | set(b.grads)):
        if a.grads.get(name) != b.grads.get(name):
            reasons.append(f"grad[{name}] {a.grads.get(name)} != {b.grads.get(name)}")
    for name in sorted(set(a.params) | set(b.params)):
        if a.params.get(name) != b.params.get(name):
            reasons.append(f"param[{name}] {a.params.get(name)} != {b.params.get(name)}")
    return reasons


__all__ = [
    "ChecksumSet",
    "compare",
    "global_checksum",
    "state_checksum",
    "tensor_checksum",
]
