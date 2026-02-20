"""
Reporting stages for Weekly CI Kickoff.

This module provides:
- Summary report generation
- Dashboard creation
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from ..utils import get_config_dir_name, parse_config_pairs


def stage_generate_summary(
    experiment_dir: str,
    repo_root: Path,
    config_pairs: str,
    baseline: str,
    rccl_branch: str,
    gpu_target: str,
    baseline_experiment_dir: Optional[str],
    logger: logging.Logger,
) -> Path:
    """Generate a summary report for the experiment.

    Creates a summary.txt file in the experiment directory with:
    - Configuration information
    - Test configurations run
    - Generated artifacts list
    - Key metrics (if available)

    Args:
        experiment_dir: Path to the experiment directory (relative to repo_root).
        repo_root: Path to the aorta repository root.
        config_pairs: Space-separated CU,threads pairs.
        baseline: Baseline configuration (CU,threads format).
        rccl_branch: RCCL branch tested.
        gpu_target: GPU architecture target.
        baseline_experiment_dir: Path to baseline experiment for cross-timestamp (if any).
        logger: Logger instance.

    Returns:
        Path to the generated summary file.
    """
    logger.info("Generating summary report...")

    exp_path = repo_root / experiment_dir
    summary_path = exp_path / "summary.txt"

    # Parse configurations
    pairs = parse_config_pairs(config_pairs)

    # Build summary content
    lines = []
    lines.append("=" * 70)
    lines.append("RCCL Warp Speed Performance Analysis Summary")
    lines.append("=" * 70)
    lines.append("")

    # Timestamp
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Configuration section
    lines.append("-" * 70)
    lines.append("Configuration")
    lines.append("-" * 70)
    lines.append(f"Experiment Directory: {experiment_dir}")
    lines.append(f"RCCL Branch: {rccl_branch}")
    lines.append(f"GPU Target: {gpu_target}")
    lines.append(f"Baseline: {baseline} (CU,Threads)")
    lines.append("")

    # Tested configurations
    lines.append("-" * 70)
    lines.append("Tested Configurations")
    lines.append("-" * 70)
    for cu, threads in pairs:
        config_dir = get_config_dir_name(cu, threads)
        config_path = exp_path / config_dir
        status = "✓" if config_path.exists() else "✗"
        baseline_marker = " (baseline)" if f"{cu},{threads}" == baseline else ""
        lines.append(f"  {status} CU={cu}, Threads={threads}{baseline_marker}")
    lines.append("")

    # Cross-timestamp comparison
    if baseline_experiment_dir:
        lines.append("-" * 70)
        lines.append("Cross-Timestamp Comparison")
        lines.append("-" * 70)
        lines.append(f"Baseline Experiment: {baseline_experiment_dir}")
        lines.append(f"Current Experiment: {experiment_dir}")
        lines.append("")

    # Generated artifacts
    lines.append("-" * 70)
    lines.append("Generated Artifacts")
    lines.append("-" * 70)

    artifacts = _scan_artifacts(exp_path, logger)
    for artifact_type, artifact_list in artifacts.items():
        if artifact_list:
            lines.append(f"  {artifact_type}:")
            for artifact in artifact_list[:5]:  # Limit to 5 per type
                lines.append(f"    - {artifact}")
            if len(artifact_list) > 5:
                lines.append(f"    ... and {len(artifact_list) - 5} more")
    lines.append("")

    # Key metrics summary (if available)
    metrics_summary = _extract_key_metrics(exp_path, pairs, baseline, logger)
    if metrics_summary:
        lines.append("-" * 70)
        lines.append("Key Metrics Summary")
        lines.append("-" * 70)
        for line in metrics_summary:
            lines.append(line)
        lines.append("")

    lines.append("=" * 70)
    lines.append("End of Summary")
    lines.append("=" * 70)

    # Write summary file
    summary_content = "\n".join(lines)
    summary_path.write_text(summary_content)

    logger.info(f"  ✓ Summary written to: {summary_path}")

    # Also log summary to console
    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    for line in lines[4:20]:  # Print first few lines
        logger.info(f"  {line}")
    if len(lines) > 20:
        logger.info("  ...")
        logger.info(f"  (Full summary in {summary_path})")
    logger.info("=" * 60)

    return summary_path


def _scan_artifacts(exp_path: Path, logger: logging.Logger) -> dict[str, list[str]]:
    """Scan experiment directory for generated artifacts.

    Args:
        exp_path: Path to the experiment directory.
        logger: Logger instance.

    Returns:
        Dictionary of artifact types to list of artifact paths.
    """
    artifacts: dict[str, list[str]] = {
        "Summary Reports": [],
        "Comparison Results": [],
        "Cross-Timestamp Results": [],
        "Excel Reports": [],
        "HTML Reports": [],
        "Visualizations": [],
    }

    if not exp_path.exists():
        return artifacts

    # Scan for summaries in config directories
    for item in exp_path.iterdir():
        if item.is_dir():
            summary_dir = item / "summary"
            if summary_dir.exists():
                artifacts["Summary Reports"].append(f"{item.name}/summary/")

    # Comparison results
    comparison_dir = exp_path / "comparison_results"
    if comparison_dir.exists():
        for item in comparison_dir.iterdir():
            if item.is_dir():
                artifacts["Comparison Results"].append(f"comparison_results/{item.name}/")

    # Cross-timestamp comparison
    cross_ts_dir = exp_path / "cross_timestamp_comparison"
    if cross_ts_dir.exists():
        for item in cross_ts_dir.iterdir():
            if item.is_dir():
                artifacts["Cross-Timestamp Results"].append(
                    f"cross_timestamp_comparison/{item.name}/"
                )

    # Compare all runs
    compare_all_dir = exp_path / "compare_all_runs"
    if compare_all_dir.exists():
        artifacts["Comparison Results"].append("compare_all_runs/")

    # Excel and HTML reports
    for item in exp_path.rglob("*.xlsx"):
        rel_path = item.relative_to(exp_path)
        artifacts["Excel Reports"].append(str(rel_path))

    for item in exp_path.rglob("*.html"):
        rel_path = item.relative_to(exp_path)
        artifacts["HTML Reports"].append(str(rel_path))

    # Visualizations (PNG files)
    for item in exp_path.rglob("*.png"):
        rel_path = item.relative_to(exp_path)
        artifacts["Visualizations"].append(str(rel_path))

    return artifacts


def _extract_key_metrics(
    exp_path: Path,
    pairs: list[tuple[str, str]],
    baseline: str,
    logger: logging.Logger,
) -> list[str]:
    """Extract key metrics from experiment results.

    Args:
        exp_path: Path to the experiment directory.
        pairs: List of (cu, threads) tuples.
        baseline: Baseline configuration string.
        logger: Logger instance.

    Returns:
        List of summary lines for key metrics.
    """
    lines = []

    baseline_parts = baseline.split(",")
    if len(baseline_parts) != 2:
        return lines

    baseline_cu, baseline_threads = baseline_parts
    baseline_dir = get_config_dir_name(baseline_cu, baseline_threads)

    # Try to find metrics.json files
    for cu, threads in pairs:
        config_dir = get_config_dir_name(cu, threads)
        metrics_path = exp_path / config_dir / "metrics.json"

        if metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text())
                # Extract key metrics if available
                if "iteration_time_ms" in metrics:
                    iter_time = metrics["iteration_time_ms"]
                    is_baseline = config_dir == baseline_dir
                    marker = " (baseline)" if is_baseline else ""
                    lines.append(f"  {config_dir}: {iter_time:.2f} ms/iteration{marker}")
            except (json.JSONDecodeError, KeyError) as e:
                logger.debug(f"Could not parse metrics from {metrics_path}: {e}")

    # Try to find comparison summary
    comparison_dir = exp_path / "comparison_results"
    if comparison_dir.exists():
        for item in comparison_dir.iterdir():
            summary_file = item / "summary.json"
            if summary_file.exists():
                try:
                    summary = json.loads(summary_file.read_text())
                    if "performance_change" in summary:
                        change = summary["performance_change"]
                        lines.append(f"  {item.name}: {change:+.1f}% vs baseline")
                except (json.JSONDecodeError, KeyError):
                    pass

    return lines


def generate_dashboard_entry(
    experiment_dir: str,
    baseline_experiment_dir: Optional[str],
    repo_root: Path,
    config_pairs: str,
    baseline: str,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Generate a dashboard entry for this experiment run.

    Creates a structured entry that can be added to a dashboard JSON file.

    Args:
        experiment_dir: Path to current experiment directory.
        baseline_experiment_dir: Path to baseline experiment (for cross-timestamp).
        repo_root: Path to aorta repository root.
        config_pairs: Space-separated CU,threads pairs.
        baseline: Baseline configuration.
        logger: Logger instance.

    Returns:
        Dashboard entry dictionary.
    """
    logger.info("Generating dashboard entry...")

    exp_path = repo_root / experiment_dir

    # Extract timestamp from experiment directory name
    # Format: experiments/rccl_warp_speed_YYYYMMDD_HHMMSS
    exp_name = Path(experiment_dir).name
    timestamp_str = exp_name.replace("rccl_warp_speed_", "")

    try:
        timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
    except ValueError:
        timestamp = datetime.now()

    entry = {
        "date": timestamp.strftime("%Y-%m-%d"),
        "time": timestamp.strftime("%H:%M:%S"),
        "experiment_dir": experiment_dir,
        "baseline_config": baseline,
        "configurations": [],
        "cross_timestamp_baseline": baseline_experiment_dir,
    }

    # Add configuration results
    pairs = parse_config_pairs(config_pairs)
    for cu, threads in pairs:
        config_dir = get_config_dir_name(cu, threads)
        config_path = exp_path / config_dir

        config_entry = {
            "cu": int(cu),
            "threads": int(threads),
            "exists": config_path.exists(),
            "metrics": {},
        }

        # Try to load metrics
        metrics_path = config_path / "metrics.json"
        if metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text())
                config_entry["metrics"] = metrics
            except json.JSONDecodeError:
                pass

        entry["configurations"].append(config_entry)

    logger.info(f"  ✓ Dashboard entry generated for {timestamp.strftime('%Y-%m-%d')}")

    return entry


def update_dashboard_file(
    dashboard_entry: dict[str, Any],
    aorta_report_dir: Path,
    logger: logging.Logger,
) -> Path:
    """Update the dashboard JSON file with a new entry.

    Args:
        dashboard_entry: Dashboard entry to add.
        aorta_report_dir: Path to aorta-report repository.
        logger: Logger instance.

    Returns:
        Path to the dashboard file.
    """
    dashboard_path = aorta_report_dir / "dashboard" / "rccl_warp_speed.json"
    dashboard_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing dashboard
    if dashboard_path.exists():
        try:
            dashboard = json.loads(dashboard_path.read_text())
        except json.JSONDecodeError:
            dashboard = {"entries": []}
    else:
        dashboard = {"entries": []}

    # Add new entry (avoid duplicates by date)
    entry_date = dashboard_entry["date"]
    existing_dates = [e.get("date") for e in dashboard.get("entries", [])]

    if entry_date in existing_dates:
        # Update existing entry
        for i, entry in enumerate(dashboard["entries"]):
            if entry.get("date") == entry_date:
                dashboard["entries"][i] = dashboard_entry
                logger.info(f"  Updated dashboard entry for {entry_date}")
                break
    else:
        # Add new entry
        dashboard["entries"].append(dashboard_entry)
        logger.info(f"  Added new dashboard entry for {entry_date}")

    # Sort by date (newest first)
    dashboard["entries"].sort(key=lambda x: x.get("date", ""), reverse=True)

    # Keep only last 30 entries
    if len(dashboard["entries"]) > 30:
        dashboard["entries"] = dashboard["entries"][:30]

    # Write dashboard
    dashboard_path.write_text(json.dumps(dashboard, indent=2))

    logger.info(f"  ✓ Dashboard updated: {dashboard_path}")

    return dashboard_path

