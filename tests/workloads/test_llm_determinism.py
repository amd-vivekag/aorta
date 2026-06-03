"""Config parsing + per-block replay-equality contract on a tiny model (CPU)."""

import pytest
import torch

from aorta.instrumentation.checksum import tensor_checksum
from aorta.instrumentation.determinism import enable_deterministic
from aorta.models import BlockConfig, RepeatedBlockModel
from aorta.workloads.llm_determinism import (
    LlmDeterminismConfig,
    _BlockHookManager,
    _compare_block_lists,
)


def test_config_from_dict_picks_known_keys() -> None:
    cfg = LlmDeterminismConfig.from_dict({"num_layers": 4, "dtype": "fp32", "unknown": "ignored"})
    assert cfg.num_layers == 4 and cfg.dtype == "fp32"


@pytest.mark.parametrize("bad", [{"dtype": "fp8"}, {"checksum_mode": "everyone"}, {"steps": 0}])
def test_config_validation_rejects_garbage(bad: dict) -> None:
    with pytest.raises(ValueError):
        LlmDeterminismConfig.from_dict(bad)


def _tiny_model() -> RepeatedBlockModel:
    return RepeatedBlockModel(BlockConfig(
        vocab_size=128, hidden_size=64, ffn_size=128, num_heads=4,
        num_layers=3, seq_len=16,
    )).to(torch.float32)


def test_moe_path_runs_and_routes_all_tokens() -> None:
    cfg = BlockConfig(vocab_size=64, hidden_size=32, ffn_size=64, num_heads=4,
                      num_layers=1, seq_len=8, num_experts=4)
    m = RepeatedBlockModel(cfg).to(torch.float32)
    y = m(torch.randint(0, 64, (1, 8)))
    assert y.shape == (1, 8, 64) and torch.isfinite(y).all()


def test_per_block_replay_is_bit_identical_on_cpu() -> None:
    enable_deterministic(seed=7)
    model = _tiny_model()
    hooks = _BlockHookManager(list(model.blocks))
    ids = torch.randint(0, 128, (1, 16))
    snap = {n: p.detach().clone() for n, p in model.named_parameters()}

    def step() -> tuple[int, list[int], list[int]]:
        model.zero_grad(set_to_none=True)
        cs = hooks.start_capture()
        logits = model(ids)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, 128).float(), ids.view(-1))
        loss.backward()
        hooks.stop_capture()
        return tensor_checksum(logits.detach()), list(cs.pre), list(cs.post)

    o1, pre1, post1 = step()
    with torch.no_grad():
        for n, p in model.named_parameters():
            p.copy_(snap[n])
    o2, pre2, post2 = step()

    hooks.remove()
    assert o1 == o2
    assert pre1 == pre2 and post1 == post2
    assert len(pre1) == 3 and len(post1) == 3


def test_block_list_compare_flags_per_block_divergence() -> None:
    from aorta.workloads.llm_determinism import _BlockChecksums
    a = _BlockChecksums(pre=[1, 2, 3], post=[10, 20, 30])
    b = _BlockChecksums(pre=[1, 99, 3], post=[10, 20, 30])
    reasons = _compare_block_lists(a, b)
    assert len(reasons) == 1 and "block[1].pre" in reasons[0]
