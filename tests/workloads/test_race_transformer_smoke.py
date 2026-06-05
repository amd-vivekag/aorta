"""CPU smoke test for the shared-weight transformer compute + checksum detector.

Runs the REAL pieces end-to-end without a GPU or torch.distributed:
  * the borrowed RepeatedTransformerBlock (same model llm_determinism uses),
  * the shared-block + fixed-reference-input invariant from setup_buffers,
  * the real FSDPModeReproducer._checksum and _verify_layer_checksums.

It proves three things a green cluster run alone cannot:
  1. a real transformer block forward actually runs (not mm+gelu, not a no-op),
  2. with shared weights + same input, every layer's output is byte-identical,
  3. the per-layer checksum detector PASSES when clean and FIRES (localized to
     compute) when a layer's output is corrupted.

Run:  python -m pytest tests/workloads/test_race_transformer_smoke.py -v
"""

import pytest

torch = pytest.importorskip("torch")

from aorta.models import BlockConfig, RepeatedTransformerBlock
from aorta.race.modes.fsdp import FSDPModeReproducer


HIDDEN = 64
NUM_LAYERS = 4
NUM_HEADS = 4
SEQ = 8
BATCH = 1
DTYPE = torch.bfloat16


def _build_shared_block_and_input():
    """Mirror setup_buffers' shared-weight transformer construction (CPU)."""
    cfg = BlockConfig(
        hidden_size=HIDDEN,
        num_heads=NUM_HEADS,
        num_layers=1,
        ffn_size=HIDDEN * 4,
        num_experts=1,
    )
    # Fixed seed -> deterministic, reproducible weights (CPU analogue of the
    # fork_rng + cuda.manual_seed(0) used on device).
    torch.manual_seed(0)
    block = RepeatedTransformerBlock(cfg).to(DTYPE)
    block.eval()
    g = torch.Generator()
    g.manual_seed(1)
    reference_input = torch.randn(BATCH, SEQ, HIDDEN, dtype=DTYPE, generator=g)
    return block, reference_input


def _run_layers(block, reference_input):
    """Per-layer forward + 4 checksums, exactly like _forward_layer's shared path."""
    layer_checksums = []
    for _ in range(NUM_LAYERS):
        comm_input = FSDPModeReproducer._checksum(reference_input)
        comm_output = comm_input  # no real all_gather on CPU; identical by construction
        compute_input = FSDPModeReproducer._checksum(reference_input)
        with torch.no_grad():
            out = block(reference_input)
        compute_output = FSDPModeReproducer._checksum(out)
        layer_checksums.append(
            {
                "comm_input": comm_input,
                "comm_output": comm_output,
                "compute_input": compute_input,
                "compute_output": compute_output,
            }
        )
    return layer_checksums


def _verifier(layer_checksums):
    """Minimal FSDPModeReproducer carrying just what _verify_layer_checksums reads."""
    r = object.__new__(FSDPModeReproducer)
    r.layer_checksums = layer_checksums
    r.rank = 0
    r.corruption_details = []
    r.layers_verified = 0
    r.layer_checksum_mismatches = 0
    return r


def test_num_heads_auto_derived_when_zero():
    """num_heads=0 must resolve to model_dim//128 (the recipe relies on this)."""
    # mirrors fsdp.setup_buffers derivation
    hidden = 1024
    cfg_num_heads = 0
    resolved = cfg_num_heads or (hidden // 128)
    assert resolved == 8
    # and the block accepts it
    cfg = BlockConfig(hidden_size=hidden, num_heads=resolved, num_layers=1,
                      ffn_size=hidden * 4, num_experts=1)
    assert cfg.hidden_size % cfg.num_heads == 0


def test_real_transformer_block_runs_on_cpu():
    """A real RepeatedTransformerBlock forward executes and returns the right shape."""
    block, ref = _build_shared_block_and_input()
    with torch.no_grad():
        out = block(ref)
    assert out.shape == (BATCH, SEQ, HIDDEN)
    assert out.dtype == DTYPE
    # Not a trivial no-op: output differs from input.
    assert FSDPModeReproducer._checksum(out) != FSDPModeReproducer._checksum(ref)


def test_shared_weights_make_layers_identical_and_detector_passes():
    """Clean path: shared block + same input -> identical layers -> detector PASSES."""
    block, ref = _build_shared_block_and_input()
    layer_checksums = _run_layers(block, ref)

    # Invariant: every layer's compute_output checksum is identical.
    outs = {c["compute_output"] for c in layer_checksums}
    assert len(outs) == 1, "shared weights + same input must yield identical layer outputs"

    r = _verifier(layer_checksums)
    assert r._verify_layer_checksums(iteration=0) is True
    assert r.layer_checksum_mismatches == 0
    assert r.layers_verified == NUM_LAYERS - 1  # layers 1..N compared to layer 0


def test_injected_compute_corruption_is_detected_and_localized():
    """Corrupt one layer's compute_output -> detector FIRES, localized to compute."""
    block, ref = _build_shared_block_and_input()
    layer_checksums = _run_layers(block, ref)

    bad_layer = 2
    layer_checksums[bad_layer]["compute_output"] += 1  # flip the checksum

    r = _verifier(layer_checksums)
    assert r._verify_layer_checksums(iteration=0) is False
    assert r.layer_checksum_mismatches == 1
    detail = r.corruption_details[0]
    assert detail["type"] == "layer_checksum_mismatch_compute_output"
    assert detail["layer_cmp"] == bad_layer
    assert "comm" not in detail["type"]  # localized to compute, not the NIC path
