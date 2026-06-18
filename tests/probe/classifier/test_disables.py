"""Tests for the detector-disable knob (issue #229).

Covers the token-validation layer (:mod:`aorta.probe.classifier.disables`)
and its effect inside :func:`aorta.probe.classifier.classify_trial`: a
disabled detector must never flip the verdict nor appear in the fired
lists, and a disabled whole tier must skip evaluation entirely.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from aorta.probe.classifier import TrialContext, classify_trial
from aorta.probe.classifier.disables import (
    DetectorSpecError,
    normalize_detector_id,
    normalize_detector_ids,
    normalize_tier,
    normalize_tiers,
)
from aorta.probe.classifier.tier3_kernel import Tier3State
from aorta.probe.classifier.tier5_custom import CompiledPattern


# --------------------------------------------------------------------------
# Token validation
# --------------------------------------------------------------------------
@pytest.mark.parametrize("tok", ["tier1", "tier2", "tier3", "tier4", "tier5"])
def test_normalize_tier_accepts_known(tok: str) -> None:
    assert normalize_tier(tok) == tok


def test_normalize_tier_is_case_insensitive_and_trims() -> None:
    assert normalize_tier("  TIER3 ") == "tier3"


@pytest.mark.parametrize("tok", ["tier0", "tier6", "hang", "tier3:hang", ""])
def test_normalize_tier_rejects_unknown(tok: str) -> None:
    with pytest.raises(DetectorSpecError):
        normalize_tier(tok)


@pytest.mark.parametrize(
    "tok",
    ["tier2:hang", "tier3:vram_growth", "tier4:hip_error", "custom:my_id"],
)
def test_normalize_detector_id_accepts_known_prefixes(tok: str) -> None:
    assert normalize_detector_id(tok) == tok


def test_normalize_detector_id_lowercases_prefix_only() -> None:
    # Prefix canonicalised; the id half is preserved verbatim because
    # custom-pattern ids are user-named and case-sensitive.
    assert normalize_detector_id("TIER2:Hang") == "tier2:Hang"


@pytest.mark.parametrize("tok", ["hang", "tier2:", ":hang", "tier9:foo", "tierX:y"])
def test_normalize_detector_id_rejects_malformed(tok: str) -> None:
    with pytest.raises(DetectorSpecError):
        normalize_detector_id(tok)


def test_normalize_lists_dedupe_preserve_order() -> None:
    assert normalize_tiers(["tier3", "tier2", "tier3"]) == ("tier3", "tier2")
    assert normalize_detector_ids(["tier2:hang", "tier2:hang"]) == ("tier2:hang",)


def test_normalize_lists_none_is_empty() -> None:
    assert normalize_tiers(None) == ()
    assert normalize_detector_ids(None) == ()


@pytest.mark.parametrize("bad", ["tier3", 5, {"a": 1}])
def test_normalize_lists_reject_non_list(bad: object) -> None:
    # A bare string is a common mistake (forgetting the YAML list dash);
    # it must be rejected rather than iterated character-by-character.
    with pytest.raises(DetectorSpecError):
        normalize_tiers(bad)


# --------------------------------------------------------------------------
# classify_trial integration
# --------------------------------------------------------------------------
def _ctx(tmp_path: Path, **kw: object) -> TrialContext:
    base: dict[str, object] = {
        "exit_code": 0,
        "timed_out": False,
        "walltime_sec": 1.0,
        "trial_dir": tmp_path,
        "log_text": "",
    }
    base.update(kw)
    return TrialContext(**base)  # type: ignore[arg-type]


def test_hang_fires_when_not_disabled(tmp_path: Path) -> None:
    verdict, _ = classify_trial(_ctx(tmp_path, hang_detected=True))
    assert verdict.verdict == "fail"
    assert "tier2:hang" in verdict.failure_detectors_fired


def test_disable_tier2_suppresses_hang(tmp_path: Path) -> None:
    verdict, _ = classify_trial(
        _ctx(tmp_path, hang_detected=True, disabled_tiers=frozenset({"tier2"}))
    )
    assert verdict.verdict == "pass"
    assert "tier2:hang" not in verdict.failure_detectors_fired


def test_disable_detector_id_suppresses_hang(tmp_path: Path) -> None:
    verdict, _ = classify_trial(
        _ctx(tmp_path, hang_detected=True, disabled_detectors=frozenset({"tier2:hang"}))
    )
    assert verdict.verdict == "pass"
    assert "tier2:hang" not in verdict.failure_detectors_fired


def test_disable_tier4_suppresses_log_pattern(tmp_path: Path) -> None:
    log = "Traceback... HIP error: out of memory\n"
    fired, _ = classify_trial(_ctx(tmp_path, log_text=log))
    assert fired.verdict == "fail"  # baseline: tier4 fires

    verdict, _ = classify_trial(
        _ctx(tmp_path, log_text=log, disabled_tiers=frozenset({"tier4"}))
    )
    assert "tier4:hip_error" not in verdict.failure_detectors_fired


def test_disable_specific_tier4_id_keeps_others(tmp_path: Path) -> None:
    log = "HIP error: boom\ncudaError_LaunchFailure\n"
    verdict, _ = classify_trial(
        _ctx(tmp_path, log_text=log, disabled_detectors=frozenset({"tier4:hip_error"}))
    )
    assert "tier4:hip_error" not in verdict.failure_detectors_fired
    assert "tier4:cuda_error" in verdict.failure_detectors_fired
    assert verdict.verdict == "fail"


def test_disable_tier3_skips_dmesg_scan(tmp_path: Path) -> None:
    dmesg = "amdgpu: GPU reset begin!\n"
    baseline, _ = classify_trial(
        _ctx(tmp_path, dmesg_text=dmesg, tier3_state=Tier3State())
    )
    assert "tier3:amdgpu_reset" in baseline.failure_detectors_fired

    verdict, _ = classify_trial(
        _ctx(
            tmp_path,
            dmesg_text=dmesg,
            tier3_state=Tier3State(),
            disabled_tiers=frozenset({"tier3"}),
        )
    )
    assert "tier3:amdgpu_reset" not in verdict.failure_detectors_fired
    assert verdict.verdict == "pass"


def test_disable_custom_pattern(tmp_path: Path) -> None:
    pat = CompiledPattern(
        detector_id="custom:boom",
        regex=re.compile("boom"),
        condition_code=None,
        condition_source=None,
        on_match="fail",
        required_for_pass=False,
    )
    log = "...boom...\n"
    baseline, _ = classify_trial(_ctx(tmp_path, log_text=log, custom_patterns=(pat,)))
    assert "custom:boom" in baseline.failure_detectors_fired
    assert baseline.verdict == "fail"

    verdict, _ = classify_trial(
        _ctx(
            tmp_path,
            log_text=log,
            custom_patterns=(pat,),
            disabled_detectors=frozenset({"custom:boom"}),
        )
    )
    assert "custom:boom" not in verdict.failure_detectors_fired
    assert verdict.verdict == "pass"


def test_disable_custom_pattern_leaves_no_capture_side_effect(tmp_path: Path) -> None:
    # A disabled custom detector must be truly *not evaluated*: its
    # ``capture`` named groups must not surface (and, by the same token,
    # its sandbox ``condition`` must not run). Filtering happens before
    # the Tier-5 scan, so the silenced detector leaves no trace.
    pat = CompiledPattern(
        detector_id="custom:metric",
        regex=re.compile(r"loss=(?P<loss>[0-9.]+)"),
        condition_code=None,
        condition_source=None,
        on_match="info",
        required_for_pass=False,
    )
    log = "step 1 loss=3.14\n"
    baseline, _ = classify_trial(_ctx(tmp_path, log_text=log, custom_patterns=(pat,)))
    assert baseline.capture.get("loss") == "3.14"

    verdict, _ = classify_trial(
        _ctx(
            tmp_path,
            log_text=log,
            custom_patterns=(pat,),
            disabled_detectors=frozenset({"custom:metric"}),
        )
    )
    assert "loss" not in verdict.capture


def test_disable_required_custom_pattern_does_not_synthesize_missing_signal(
    tmp_path: Path,
) -> None:
    # A required-for-pass pattern that the operator disables must not
    # leave behind a meta:missing_pass_signal failure -- disabling means
    # "ignore this detector entirely", required-ness included.
    pat = CompiledPattern(
        detector_id="custom:must_see",
        regex=re.compile("never-matches-zzz"),
        condition_code=None,
        condition_source=None,
        on_match="fail",
        required_for_pass=True,
    )
    verdict, _ = classify_trial(
        _ctx(
            tmp_path,
            log_text="nothing here\n",
            custom_patterns=(pat,),
            disabled_detectors=frozenset({"custom:must_see"}),
        )
    )
    assert verdict.verdict == "pass"
    assert all("missing_pass_signal" not in d for d in verdict.failure_detectors_fired)
