#!/usr/bin/env python3
"""Compare loss curves across TF32 precision experiments.

Reads rank_00_metrics.jsonl from each experiment directory, computes pairwise
relative error in loss, and reports whether each mode meets the client's
0.1% max error tolerance.

Usage:
    python scripts/compare_precision_runs.py \
        --baseline experiments/multinode_*_precision_tf32x3/*/rank_00_metrics.jsonl \
        --compare  experiments/multinode_*_precision_tf32x1/*/rank_00_metrics.jsonl \
                   experiments/multinode_*_precision_native_tf32/*/rank_00_metrics.jsonl

    # Or auto-discover from experiment directories:
    python scripts/compare_precision_runs.py \
        --baseline-dir experiments/multinode_56ch_256th_*_precision_tf32x3 \
        --compare-dir  experiments/multinode_56ch_256th_*_precision_tf32x1 \
                       experiments/multinode_56ch_256th_*_precision_native_tf32
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Optional


def load_losses(jsonl_path: Path) -> list[float]:
    """Load per-step loss values from a metrics JSONL file."""
    losses = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                loss = record.get("loss")
                if loss is not None:
                    losses.append(float(loss))
            except (json.JSONDecodeError, ValueError):
                continue
    return losses


def find_metrics_file(experiment_dir: Path) -> Optional[Path]:
    """Find rank_00_metrics.jsonl inside an experiment directory."""
    for candidate in experiment_dir.rglob("rank_00_metrics.jsonl"):
        return candidate
    return None


def compute_error_metrics(baseline: list[float], compare: list[float]) -> dict:
    """Compute per-step relative error between two loss curves.

    Uses the shorter of the two sequences.
    """
    n = min(len(baseline), len(compare))
    if n == 0:
        return {"steps": 0, "mean_rel_error_pct": float("nan"), "max_rel_error_pct": float("nan"),
                "max_error_step": -1}

    rel_errors = []
    max_error = 0.0
    max_error_step = 0

    for i in range(n):
        ref = abs(baseline[i])
        if ref < 1e-12:
            continue
        err = abs(compare[i] - baseline[i]) / ref
        rel_errors.append(err)
        if err > max_error:
            max_error = err
            max_error_step = i

    if not rel_errors:
        return {"steps": n, "mean_rel_error_pct": 0.0, "max_rel_error_pct": 0.0, "max_error_step": 0}

    mean_err = sum(rel_errors) / len(rel_errors)
    return {
        "steps": n,
        "mean_rel_error_pct": mean_err * 100,
        "max_rel_error_pct": max_error * 100,
        "max_error_step": max_error_step,
    }


def resolve_dir(pattern: str) -> Optional[Path]:
    """Resolve a glob pattern to a single directory."""
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None
    return Path(matches[-1])


def main():
    parser = argparse.ArgumentParser(description="Compare loss curves across TF32 precision runs")
    parser.add_argument("--baseline", type=str, help="Path to baseline rank_00_metrics.jsonl")
    parser.add_argument("--compare", type=str, nargs="+", help="Paths to comparison rank_00_metrics.jsonl files")
    parser.add_argument("--baseline-dir", type=str, help="Glob pattern for baseline experiment directory")
    parser.add_argument("--compare-dir", type=str, nargs="+", help="Glob patterns for comparison experiment directories")
    parser.add_argument("--tolerance", type=float, default=0.1, help="Max error tolerance in %% (default: 0.1)")
    parser.add_argument("--skip-warmup", type=int, default=50, help="Skip first N steps as warmup (default: 50)")
    args = parser.parse_args()

    # Resolve baseline
    if args.baseline:
        baseline_path = Path(args.baseline)
    elif args.baseline_dir:
        bdir = resolve_dir(args.baseline_dir)
        if bdir is None:
            print(f"ERROR: No directory matches baseline pattern: {args.baseline_dir}", file=sys.stderr)
            sys.exit(1)
        baseline_path = find_metrics_file(bdir)
        if baseline_path is None:
            print(f"ERROR: No rank_00_metrics.jsonl found in {bdir}", file=sys.stderr)
            sys.exit(1)
    else:
        print("ERROR: Provide --baseline or --baseline-dir", file=sys.stderr)
        sys.exit(1)

    # Resolve comparison files
    compare_paths = []
    if args.compare:
        compare_paths = [Path(p) for p in args.compare]
    elif args.compare_dir:
        for pattern in args.compare_dir:
            cdir = resolve_dir(pattern)
            if cdir is None:
                print(f"WARNING: No directory matches pattern: {pattern}", file=sys.stderr)
                continue
            cpath = find_metrics_file(cdir)
            if cpath is None:
                print(f"WARNING: No rank_00_metrics.jsonl in {cdir}", file=sys.stderr)
                continue
            compare_paths.append(cpath)
    else:
        print("ERROR: Provide --compare or --compare-dir", file=sys.stderr)
        sys.exit(1)

    if not compare_paths:
        print("ERROR: No valid comparison files found", file=sys.stderr)
        sys.exit(1)

    # Load baseline
    print(f"Baseline: {baseline_path}")
    baseline_losses = load_losses(baseline_path)
    print(f"  Total steps: {len(baseline_losses)}")

    if args.skip_warmup > 0:
        baseline_losses = baseline_losses[args.skip_warmup:]
        print(f"  After warmup skip ({args.skip_warmup}): {len(baseline_losses)} steps")

    print()
    print(f"{'Experiment':<60} {'Steps':>6} {'Mean Err%':>10} {'Max Err%':>10} {'MaxStep':>8} {'Status':>10}")
    print("-" * 110)

    all_pass = True
    for cpath in compare_paths:
        label = str(cpath.parent.parent.name) if cpath.parent.name.endswith("channels") else str(cpath.parent.name)
        compare_losses = load_losses(cpath)

        if args.skip_warmup > 0:
            compare_losses = compare_losses[args.skip_warmup:]

        metrics = compute_error_metrics(baseline_losses, compare_losses)
        passed = metrics["max_rel_error_pct"] <= args.tolerance
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False

        print(f"{label:<60} {metrics['steps']:>6} {metrics['mean_rel_error_pct']:>9.4f}% {metrics['max_rel_error_pct']:>9.4f}% {metrics['max_error_step']:>8} {status:>10}")

    print("-" * 110)
    print(f"Tolerance: {args.tolerance}% max relative error (end-to-end, including outliers)")
    print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
