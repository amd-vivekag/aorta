"""
Test execution stages for Weekly CI Kickoff.

This module provides:
- Performance test execution (RCCL warp speed comparison)
- Experiment directory discovery
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from ..utils import (
    docker_exec,
    find_latest_experiment_dir,
    find_second_latest_experiment_dir,
    parse_config_pairs,
)


def stage_run_performance_tests(
    container_name: str,
    config_pairs: str,
    training_config: str,
    logger: logging.Logger,
) -> None:
    """Run RCCL warp speed comparison tests inside the Docker container.

    This stage executes the run_rccl_warp_speed_comparison.sh script which:
    1. Runs training with different CU/thread configurations
    2. Collects performance traces for each configuration
    3. Stores results in experiments/rccl_warp_speed_<timestamp>/ directory

    Args:
        container_name: Name of the Docker container.
        config_pairs: Space-separated CU,threads pairs (e.g., "56,256 37,384 32,512").
        training_config: Path to training config YAML (inside container).
        logger: Logger instance.

    Raises:
        RuntimeError: If performance tests fail.
    """
    logger.info("Running RCCL warp speed comparison tests...")
    logger.info(f"  Config pairs: {config_pairs}")
    logger.info(f"  Training config: {training_config}")

    # Parse and display configurations
    pairs = parse_config_pairs(config_pairs)
    for cu, threads in pairs:
        logger.info(f"    - {cu} CUs, {threads} threads")

    # Build the test script to run inside container
    test_script = f"""
        set -e

        # Set RCCL library path
        export LD_LIBRARY_PATH=/rccl/rccl/build/release:$LD_LIBRARY_PATH

        echo "Starting RCCL warp speed comparison tests..."
        echo "Config pairs: {config_pairs}"
        echo "Training config: {training_config}"

        # Run the RCCL warp speed comparison script
        bash ./scripts/tracelens_single_config/run_rccl_warp_speed_comparison.sh \\
            -p "{config_pairs}" \\
            -c {training_config}

        echo "Performance tests completed!"
    """

    logger.info("  Starting performance tests...")
    logger.info("  (This may take 30-60 minutes depending on configurations)")

    try:
        docker_exec(
            container_name,
            test_script,
            logger,
            workdir="/workspace",
            check=True,
        )
        logger.info("  ✓ Performance tests completed successfully")
    except Exception as e:
        raise RuntimeError(f"Performance tests failed: {e}") from e


def stage_find_experiment_dir(
    repo_root: Path,
    logger: logging.Logger,
) -> Optional[str]:
    """Find the most recently created experiment directory.

    Searches for experiment directories matching the pattern:
    experiments/rccl_warp_speed_<timestamp>/

    Args:
        repo_root: Path to the repository root.
        logger: Logger instance.

    Returns:
        Relative path to the experiment directory (e.g., "experiments/rccl_warp_speed_20260220_143000"),
        or None if no experiment directory found.

    Raises:
        RuntimeError: If no experiment directory is found.
    """
    logger.info("Finding experiment directory...")

    experiments_dir = repo_root / "experiments"
    experiment_dir = find_latest_experiment_dir(experiments_dir)

    if experiment_dir is None:
        raise RuntimeError(
            f"No experiment directory found in {experiments_dir}. "
            "Make sure performance tests have been run first."
        )

    # Return relative path
    relative_path = experiment_dir.relative_to(repo_root)
    logger.info(f"  Found experiment directory: {relative_path}")

    return str(relative_path)


def stage_find_baseline_experiment_dir(
    repo_root: Path,
    baseline_experiment: str,
    logger: logging.Logger,
) -> Optional[str]:
    """Find baseline experiment directory for cross-timestamp comparison.

    If baseline_experiment is provided, uses that directory.
    Otherwise, auto-detects the second most recent experiment directory.

    Args:
        repo_root: Path to the repository root.
        baseline_experiment: Explicit baseline experiment path (empty for auto-detect).
        logger: Logger instance.

    Returns:
        Relative path to the baseline experiment directory, or None if not found.
    """
    logger.info("Finding baseline experiment directory for cross-timestamp comparison...")

    if baseline_experiment:
        # Use explicitly provided baseline
        baseline_path = repo_root / baseline_experiment
        if baseline_path.exists():
            logger.info(f"  Using provided baseline: {baseline_experiment}")
            return baseline_experiment
        else:
            logger.warning(f"  Provided baseline not found: {baseline_experiment}")
            logger.info("  Attempting auto-detection...")

    # Auto-detect second most recent experiment
    experiments_dir = repo_root / "experiments"
    baseline_dir = find_second_latest_experiment_dir(experiments_dir)

    if baseline_dir is None:
        logger.warning("  No baseline experiment directory found for cross-timestamp comparison")
        logger.warning("  (Need at least 2 experiment runs for comparison)")
        return None

    # Return relative path
    relative_path = baseline_dir.relative_to(repo_root)
    logger.info(f"  Found baseline experiment directory: {relative_path}")

    return str(relative_path)


def validate_experiment_configs(
    experiment_dir: Path,
    config_pairs: str,
    logger: logging.Logger,
) -> list[str]:
    """Validate that experiment directory contains expected configuration subdirectories.

    Args:
        experiment_dir: Path to the experiment directory.
        config_pairs: Space-separated CU,threads pairs.
        logger: Logger instance.

    Returns:
        List of found configuration directory names.
    """
    pairs = parse_config_pairs(config_pairs)
    found_configs = []
    missing_configs = []

    for cu, threads in pairs:
        config_dir_name = f"{cu}cu_{threads}threads"
        config_path = experiment_dir / config_dir_name

        if config_path.exists() and config_path.is_dir():
            found_configs.append(config_dir_name)
            logger.debug(f"  Found: {config_dir_name}")
        else:
            missing_configs.append(config_dir_name)
            logger.warning(f"  Missing: {config_dir_name}")

    if missing_configs:
        logger.warning(
            f"  Warning: {len(missing_configs)} configuration(s) missing: "
            f"{', '.join(missing_configs)}"
        )

    return found_configs

