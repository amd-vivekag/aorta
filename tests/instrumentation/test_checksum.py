"""Bit-equality and divergence detection for the checksum helper."""

import pytest
import torch

from aorta.instrumentation.checksum import (
    ChecksumSet,
    compare,
    state_checksum,
    tensor_checksum,
)


def test_identical_tensors_match() -> None:
    a = torch.randn(64, 64, dtype=torch.bfloat16)
    assert tensor_checksum(a) == tensor_checksum(a.clone())


def test_one_bit_flip_diverges() -> None:
    a = torch.zeros(8, dtype=torch.float32)
    b = a.clone()
    b[0] = 1.0
    assert tensor_checksum(a) != tensor_checksum(b)


def test_plus_zero_vs_minus_zero_diverges() -> None:
    # Numeric sum would say these are equal; bit checksum must not.
    a = torch.zeros(1, dtype=torch.float32)
    b = torch.tensor([-0.0], dtype=torch.float32)
    assert tensor_checksum(a) != tensor_checksum(b)


def test_state_dict_checksum_keyed_per_tensor() -> None:
    sd = {"w": torch.ones(4, dtype=torch.bfloat16), "b": torch.zeros(4, dtype=torch.bfloat16)}
    cs = state_checksum(sd)
    assert set(cs) == {"w", "b"} and cs["w"] != cs["b"]


def test_compare_reports_divergent_keys() -> None:
    a = ChecksumSet(loss_bits=1, output_bits=2, grads={"x": 10}, params={"y": 20})
    b = ChecksumSet(loss_bits=1, output_bits=2, grads={"x": 11}, params={"y": 20})
    reasons = compare(a, b)
    assert len(reasons) == 1 and "grad[x]" in reasons[0]


def test_unsupported_dtype_raises() -> None:
    with pytest.raises(TypeError):
        tensor_checksum(torch.tensor([1 + 2j], dtype=torch.complex64))
