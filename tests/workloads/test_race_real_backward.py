"""CPU test for the real-backward core logic (Option 1).

Tests the grad-enabled backward that _backward_layer runs when
config.real_backward=True, in isolation (no GPU, no torch.distributed):
proves real gradient kernels run over the shared block and produce real,
finite, non-zero input grads -- unlike the no_grad timing proxy which
builds no graph and produces no grads.

Run:  python -m pytest tests/workloads/test_race_real_backward.py -v
"""

import pytest

torch = pytest.importorskip("torch")

from aorta.models import BlockConfig, RepeatedTransformerBlock


HIDDEN = 64
FFN = 128
NUM_HEADS = 4
SEQ = 8
BATCH = 1


def _block() -> RepeatedTransformerBlock:
    cfg = BlockConfig(hidden_size=HIDDEN, ffn_size=FFN, num_heads=NUM_HEADS,
                      num_layers=1, seq_len=SEQ, vocab_size=16)
    return RepeatedTransformerBlock(cfg).eval()


def test_real_backward_produces_real_input_grads() -> None:
    block = _block()
    reference_input = torch.randn(BATCH, SEQ, HIDDEN)
    ri = reference_input.detach().requires_grad_(True)
    out = block(ri)
    loss = out.float().sum()
    loss.backward()
    assert ri.grad is not None
    assert torch.isfinite(ri.grad).all()
    assert ri.grad.abs().sum() > 0


def test_no_grad_proxy_builds_no_graph() -> None:
    block = _block()
    reference_input = torch.randn(BATCH, SEQ, HIDDEN)
    with torch.no_grad():
        out = block(reference_input)
    assert out.requires_grad is False
