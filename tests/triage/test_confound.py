"""Tests for src/aorta/triage/confound.py: baseline resolution + classify()."""

from __future__ import annotations

import pytest

from aorta.triage.confound import (
    CONFOUND_BASELINE,
    CONFOUND_DID_NOT_RUN,
    CONFOUND_ERROR,
    CONFOUND_NA,
    CONFOUND_NEUTRAL,
    CONFOUND_NO_EFFECT,
    classify,
    classify_all,
    is_did_not_run_cell,
    resolve_baseline,
)
from aorta.triage.matrix import (
    OUTCOME_COMPLETED,
    OUTCOME_DID_NOT_RUN,
    CellStats,
)
from aorta.triage.recipe import Cell, RecipeCellError


def _stats(
    name: str,
    mean_step_time_ms: float = 100.0,
    passed_count: int = 0,
    trials: int = 8,
    error: str | None = None,
    step_time_source: str = "per_step",
    outcome_counts: dict[str, int] | None = None,
) -> CellStats:
    """Build a synthetic CellStats for classify() unit tests.

    ``step_time_source`` defaults to ``"per_step"`` -- the realistic value
    for a workload that emits its own per-iteration clocks. Tests that
    exercise the new mixed-source path pass an explicit value so the
    mismatch surfaces.
    """
    return CellStats(
        name=name,
        mitigations=("none",),
        environment="local",
        extra_env={},
        resolved_env_vars={},
        trials=trials,
        passed_count=passed_count,
        failed_count=trials - passed_count,
        mean_step_time_ms=mean_step_time_ms,
        std_step_time_ms=0.0,
        min_step_time_ms=mean_step_time_ms,
        max_step_time_ms=mean_step_time_ms,
        p50_step_time_ms=mean_step_time_ms,
        p90_step_time_ms=mean_step_time_ms,
        p99_step_time_ms=mean_step_time_ms,
        mean_wall_clock_sec=1.0,
        exit_status_counts={},
        step_times_ms=[mean_step_time_ms],
        trial_paths=[],
        error=error,
        step_time_source=step_time_source,  # type: ignore[arg-type]
        outcome_counts=dict(outcome_counts) if outcome_counts is not None else {},
    )


# ---- resolve_baseline -----------------------------------------------------


def test_explicit_baseline_name_wins():
    cells = [
        Cell(name="tf32-local", mitigations=("tf32_off",), environment="local"),
        Cell(name="baseline-local", mitigations=("none",), environment="local"),
    ]
    chosen = resolve_baseline(cells, explicit_name="tf32-local")
    assert chosen.name == "tf32-local"


def test_explicit_baseline_name_not_found_raises():
    cells = [Cell(name="a", mitigations=("none",), environment="local")]
    with pytest.raises(RecipeCellError, match="does not match any cell"):
        resolve_baseline(cells, explicit_name="nope")


def test_default_picks_first_baseline_dash_prefix():
    cells = [
        Cell(name="tf32-local", mitigations=("tf32_off",), environment="local"),
        Cell(name="baseline-local", mitigations=("none",), environment="local"),
        Cell(name="baseline-docker", mitigations=("none",), environment="local"),
    ]
    assert resolve_baseline(cells, explicit_name=None).name == "baseline-local"


def test_default_falls_back_to_mitigations_none():
    cells = [
        Cell(name="tf32-local", mitigations=("tf32_off",), environment="local"),
        Cell(name="vanilla", mitigations=("none",), environment="local"),
    ]
    assert resolve_baseline(cells, explicit_name=None).name == "vanilla"


def test_single_cell_is_its_own_baseline():
    cells = [Cell(name="only", mitigations=("tf32_off",), environment="local")]
    assert resolve_baseline(cells, explicit_name=None).name == "only"


def test_no_baseline_resolution_raises():
    cells = [
        Cell(name="tf32-local", mitigations=("tf32_off",), environment="local"),
        Cell(name="xnack-local", mitigations=("xnack",), environment="local"),
    ]
    with pytest.raises(RecipeCellError, match="cannot resolve baseline cell"):
        resolve_baseline(cells, explicit_name=None)


# ---- classify -------------------------------------------------------------


def test_classify_baseline():
    base = _stats("b", mean_step_time_ms=100.0)
    tag, ratio = classify(base, base, threshold=1.15)
    assert tag == CONFOUND_BASELINE
    assert ratio is None


def test_classify_speed_confound_plus_25_percent():
    base = _stats("b", mean_step_time_ms=400.0, passed_count=4)  # failure_rate=0.5
    slow = _stats("tf32-local", mean_step_time_ms=500.0, passed_count=8)  # failure_rate=0
    tag, ratio = classify(slow, base, threshold=1.15)
    assert tag == "speed (+25%)"
    assert ratio is not None and abs(ratio - 1.25) < 1e-9


def test_classify_neutral_when_ratio_one_and_failure_rate_drops():
    base = _stats("b", mean_step_time_ms=100.0, passed_count=0)  # failure_rate 1.0
    cell = _stats("c", mean_step_time_ms=100.0, passed_count=8)  # failure_rate 0.0
    tag, ratio = classify(cell, base, threshold=1.15)
    assert tag == CONFOUND_NEUTRAL
    assert ratio == 1.0


def test_classify_no_effect_when_failure_rate_unchanged_and_no_slowdown():
    base = _stats("b", mean_step_time_ms=100.0, passed_count=0)  # failure_rate 1.0
    cell = _stats("c", mean_step_time_ms=105.0, passed_count=0)  # ratio 1.05, failure_rate 1.0
    tag, ratio = classify(cell, base, threshold=1.15)
    assert tag == CONFOUND_NO_EFFECT
    assert ratio == 1.05


def test_classify_error_cell_tag():
    base = _stats("b", mean_step_time_ms=100.0)
    err = _stats("c", error="docker pull failed")
    tag, ratio = classify(err, base, threshold=1.15)
    assert tag == CONFOUND_ERROR
    assert ratio is None


def test_classify_baseline_errored_forces_no_ratio():
    """Baseline crashed -> non-baseline cells must NOT get the trustworthy '-' tag.

    Pin the round-6 fix: previously this returned ``CONFOUND_NEUTRAL`` (the
    "mitigation works without speed cost" tag), which silently labelled
    every other cell as trustworthy even though no comparison was possible.
    The distinct ``CONFOUND_NA`` tag exists to keep that case visible.
    """
    base = _stats("b", mean_step_time_ms=0.0, error="baseline crashed")
    cell = _stats("c", mean_step_time_ms=100.0, passed_count=8)
    tag, ratio = classify(cell, base, threshold=1.15)
    assert tag == CONFOUND_NA
    assert tag != CONFOUND_NEUTRAL  # cosmetically equal? no: distinct contract.
    assert ratio is None


def test_classify_baseline_zero_step_time_forces_no_ratio():
    """Baseline with no usable timing -> non-baseline cells get CONFOUND_NA.

    Same contract as the errored-baseline case: zero step-time means the
    workload didn't emit timing, so a step-time ratio cannot be computed
    against this baseline. Pin that the tag is NOT the neutral '-'.
    """
    base = _stats("b", mean_step_time_ms=0.0)
    cell = _stats("c", mean_step_time_ms=100.0, passed_count=8)
    tag, ratio = classify(cell, base, threshold=1.15)
    assert tag == CONFOUND_NA
    assert ratio is None


def test_confound_na_is_distinct_from_neutral():
    """Schema-level pin: the two tags must render to distinct strings.

    Reusing CONFOUND_NEUTRAL for the unclassifiable case would let
    matrix.md readers parse 'no comparison possible' as 'mitigation works
    without speed cost'. The distinctness is enforced at the constant
    level so any future renderer change can't reintroduce the conflation.
    """
    assert CONFOUND_NA != CONFOUND_NEUTRAL
    assert CONFOUND_NA == "n/a"
    assert CONFOUND_NEUTRAL == "-"


# ---- per-step instrumentation required (smoke-3) -------------------------
#
# The 2026-05-13 smoke matrix produced ``speed (+3500%)`` tags from ratios
# between two cells whose step times were ``wall_clock_total / configured_steps``
# (the import-time wall-clock of a setup crash divided by a never-reached
# step count). Those ratios are dominated by setup / teardown / crash time,
# not per-step cost. Smoke-3's rule: confound classification requires
# ``step_time_source == "per_step"`` on BOTH the cell and the baseline; any
# fallback branch (``elapsed_per_iter`` or ``wall_clock_total``) on either
# side collapses to ``CONFOUND_NA``. Strictly stronger than the older
# "sources must match" rule from PR #160.


def test_classify_returns_na_when_only_baseline_has_per_step():
    """Baseline measured cleanly, cell fell back to elapsed_per_iter.
    The fallback folds workload warm-up / shutdown into the per-iter number,
    so the ratio is not a per-step comparison even though both sides have
    *some* timing."""
    base = _stats("b", mean_step_time_ms=100.0, step_time_source="per_step")
    cell = _stats("c", mean_step_time_ms=110.0, step_time_source="elapsed_per_iter")
    tag, ratio = classify(cell, base, threshold=1.15)
    assert tag == CONFOUND_NA
    assert ratio is None


def test_classify_returns_na_when_only_cell_has_per_step():
    """Symmetric: baseline fell back, cell measured cleanly. The baseline's
    number folds setup time, so the ratio's denominator isn't iteration time."""
    base = _stats("b", mean_step_time_ms=100.0, step_time_source="wall_clock_total")
    cell = _stats("c", mean_step_time_ms=120.0, step_time_source="per_step")
    tag, ratio = classify(cell, base, threshold=1.15)
    assert tag == CONFOUND_NA
    assert ratio is None


def test_classify_returns_na_when_both_sources_are_wall_clock_total():
    """The smoke-3 regression case: both sides fell back to wall-clock and the
    old classifier (sources match) happily emitted a ``speed (+N%)`` tag.
    Under the new rule, two wall-clock numbers can't anchor a per-step ratio
    even when they agree on the fallback branch."""
    base = _stats(
        "b",
        mean_step_time_ms=400.0,
        step_time_source="wall_clock_total",
        passed_count=4,
    )
    cell = _stats(
        "tf32-local",
        mean_step_time_ms=500.0,
        step_time_source="wall_clock_total",
        passed_count=8,
    )
    tag, ratio = classify(cell, base, threshold=1.15)
    assert tag == CONFOUND_NA
    assert ratio is None


def test_classify_returns_na_when_both_sources_are_elapsed_per_iter():
    """Even the higher-fidelity fallback (elapsed/iterations from the workload's
    own report) is excluded -- it still folds the workload's warm-up and any
    untimed shutdown into the per-iter number."""
    base = _stats("b", mean_step_time_ms=100.0, step_time_source="elapsed_per_iter")
    cell = _stats("c", mean_step_time_ms=120.0, step_time_source="elapsed_per_iter")
    tag, ratio = classify(cell, base, threshold=1.15)
    assert tag == CONFOUND_NA
    assert ratio is None


def test_classify_computes_ratio_when_both_sources_are_per_step():
    """Sanity check: per_step on both sides goes through the normal classifier."""
    base = _stats("b", mean_step_time_ms=400.0, step_time_source="per_step", passed_count=4)
    slow = _stats(
        "tf32-local",
        mean_step_time_ms=500.0,
        step_time_source="per_step",
        passed_count=8,
    )
    tag, ratio = classify(slow, base, threshold=1.15)
    assert tag == "speed (+25%)"
    assert ratio is not None and abs(ratio - 1.25) < 1e-9


def test_classify_returns_na_when_cell_has_missing_timing():
    """A cell with no usable timing (source='missing') can't anchor a ratio
    even if the baseline measured cleanly. matrix.md must mark it n/a, not
    coast through to a divide-by-non-comparable comparison."""
    base = _stats("b", mean_step_time_ms=100.0, step_time_source="per_step")
    cell = _stats("c", mean_step_time_ms=0.0, step_time_source="missing", passed_count=8)
    tag, ratio = classify(cell, base, threshold=1.15)
    assert tag == CONFOUND_NA
    assert ratio is None


# ---- classify_all ---------------------------------------------------------


def test_classify_all_returns_tag_per_cell():
    base = _stats("baseline-local", mean_step_time_ms=400.0, passed_count=4)
    slow = _stats("tf32-local", mean_step_time_ms=500.0, passed_count=8)
    neutral = _stats("xnack-local", mean_step_time_ms=400.0, passed_count=8)
    tags = classify_all([base, slow, neutral], baseline_name="baseline-local", threshold=1.15)
    assert tags["baseline-local"][0] == CONFOUND_BASELINE
    assert tags["tf32-local"][0] == "speed (+25%)"
    assert tags["xnack-local"][0] == CONFOUND_NEUTRAL


def test_classify_all_missing_baseline_raises():
    cell = _stats("c", mean_step_time_ms=100.0)
    with pytest.raises(RecipeCellError, match="baseline cell"):
        classify_all([cell], baseline_name="not_present", threshold=1.15)


# ---- did_not_run handling (issue #173) -----------------------------------
#
# Cells whose every trial died before the workload's primary work phase
# do not participate in confound classification at all. The classify()
# function short-circuits to CONFOUND_DID_NOT_RUN; the runner uses
# is_did_not_run_cell() to decide whether to disqualify the baseline
# (hard error if explicitly named, soft warning if auto-resolved).


def test_classify_did_not_run_cell_short_circuits():
    """Even when the baseline measured cleanly, a did-not-run cell carries
    its own tag instead of going through ratio classification."""
    base = _stats("b", mean_step_time_ms=100.0)
    cell = _stats(
        "nan-repro",
        mean_step_time_ms=0.0,
        step_time_source="missing",
        passed_count=0,
        trials=2,
        outcome_counts={OUTCOME_DID_NOT_RUN: 2},
    )
    tag, ratio = classify(cell, base, threshold=1.15)
    assert tag == CONFOUND_DID_NOT_RUN
    assert ratio is None


def test_classify_did_not_run_takes_precedence_over_baseline_match():
    """A baseline cell that itself ended up did-not-run renders as
    did_not_run, not (baseline). The matrix.md reader sees an explicit
    'this row produced no measurable work' tag rather than a misleading
    '(baseline)' label on a row that didn't actually anchor anything."""
    cell = _stats(
        "baseline-nan-repro",
        mean_step_time_ms=0.0,
        step_time_source="missing",
        trials=2,
        outcome_counts={OUTCOME_DID_NOT_RUN: 2},
    )
    tag, ratio = classify(cell, cell, threshold=1.15)
    assert tag == CONFOUND_DID_NOT_RUN
    assert ratio is None


def test_classify_mixed_outcome_cell_is_not_did_not_run():
    """At least one trial reached main work -> the cell is NOT did_not_run.
    Falls through to the normal ratio path against the baseline."""
    base = _stats("b", mean_step_time_ms=100.0)
    cell = _stats(
        "c",
        mean_step_time_ms=120.0,
        passed_count=4,
        trials=8,
        outcome_counts={OUTCOME_DID_NOT_RUN: 4, OUTCOME_COMPLETED: 4},
    )
    tag, ratio = classify(cell, base, threshold=1.15)
    assert tag != CONFOUND_DID_NOT_RUN
    # Ratio path should fire: 1.20 < threshold 1.15? No, 1.20 > 1.15 -> speed.
    assert tag == "speed (+20%)"


def test_is_did_not_run_cell_true_for_uniform_did_not_run():
    cell = _stats("c", trials=2, outcome_counts={OUTCOME_DID_NOT_RUN: 2})
    assert is_did_not_run_cell(cell) is True


def test_is_did_not_run_cell_false_for_mixed_outcomes():
    cell = _stats("c", trials=2, outcome_counts={OUTCOME_DID_NOT_RUN: 1, OUTCOME_COMPLETED: 1})
    assert is_did_not_run_cell(cell) is False


def test_is_did_not_run_cell_false_when_count_does_not_cover_all_trials():
    """Defensive guard against a partial / truncated histogram. Normal
    aggregation invariantly produces a histogram totalling ``trials``,
    but a CellStats loaded from external matrix.json or built by hand
    could carry an inconsistent count -- e.g. ``{"did_not_run": 1}``
    on a 2-trial cell -- and we must NOT flag that as did-not-run on
    the strength of the only present key happening to be did_not_run.
    Round-3 Copilot catch on PR #175.
    """
    cell = _stats("c", trials=2, outcome_counts={OUTCOME_DID_NOT_RUN: 1})
    assert is_did_not_run_cell(cell) is False


def test_is_did_not_run_cell_false_for_legacy_workload_with_empty_counts():
    """Workload that hasn't been updated to the new contract -> empty
    outcome_counts. Must NOT be flagged as did_not_run; the runner relies
    on this to keep legacy runs out of the disqualification path."""
    cell = _stats("c")
    assert cell.outcome_counts == {}
    assert is_did_not_run_cell(cell) is False


def test_confound_did_not_run_distinct_from_other_tags():
    """Schema-level pin: the did_not_run tag must render to a string that
    no other tag uses, so matrix.md / matrix.json consumers can switch on
    it unambiguously."""
    distinct_tags = {
        CONFOUND_BASELINE,
        CONFOUND_NEUTRAL,
        CONFOUND_NO_EFFECT,
        CONFOUND_ERROR,
        CONFOUND_NA,
        CONFOUND_DID_NOT_RUN,
    }
    assert len(distinct_tags) == 6
    assert CONFOUND_DID_NOT_RUN == "did_not_run"
