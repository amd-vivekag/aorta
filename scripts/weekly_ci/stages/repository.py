"""
Repository management stages for Weekly CI Kickoff.

This module provides:
- aorta-report repository checkout/update
- Results pushing to aorta-report
"""

from __future__ import annotations

import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..utils import run_command


def stage_checkout_aorta_report(
    aorta_report_path: str,
    repo_root: Path,
    logger: logging.Logger,
    git_token: Optional[str] = None,
) -> Path:
    """Checkout or update the aorta-report repository.

    If the repository exists, performs a git fetch and pull.
    If it doesn't exist, clones it from GitHub.

    Args:
        aorta_report_path: Path to the aorta-report repository (relative or absolute).
        repo_root: Path to the aorta repository root.
        logger: Logger instance.
        git_token: Optional GitHub token for authentication.

    Returns:
        Path to the aorta-report repository.

    Raises:
        RuntimeError: If checkout/update fails.
    """
    logger.info("Checking out aorta-report repository...")

    # Resolve path
    if os.path.isabs(aorta_report_path):
        aorta_report_dir = Path(aorta_report_path)
    else:
        aorta_report_dir = repo_root / aorta_report_path

    aorta_report_dir = aorta_report_dir.resolve()
    logger.info(f"  Target path: {aorta_report_dir}")

    if aorta_report_dir.exists() and (aorta_report_dir / ".git").exists():
        # Repository exists, update it
        logger.info("  Repository exists, updating...")
        try:
            run_command(
                "git fetch origin",
                logger,
                cwd=aorta_report_dir,
                capture_output=True,
                check=True,
            )
            run_command(
                "git pull --rebase origin main",
                logger,
                cwd=aorta_report_dir,
                capture_output=True,
                check=True,
            )
            logger.info("  ✓ Repository updated successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to update aorta-report repository: {e}") from e
    else:
        # Clone repository
        logger.info("  Repository not found, cloning...")

        # Create parent directory if needed
        aorta_report_dir.parent.mkdir(parents=True, exist_ok=True)

        # Build clone URL
        if git_token:
            clone_url = f"https://{git_token}@github.com/ROCm/aorta-report.git"
        else:
            # Try environment variable
            env_token = os.environ.get("AORTA_REPORT_GITHUB_TOKEN", "")
            if env_token:
                clone_url = f"https://{env_token}@github.com/ROCm/aorta-report.git"
            else:
                # Use SSH (requires SSH key setup)
                clone_url = "git@github.com:ROCm/aorta-report.git"
                logger.info("  Note: Using SSH authentication (no token provided)")

        try:
            run_command(
                f"git clone {clone_url} {aorta_report_dir}",
                logger,
                capture_output=True,
                check=True,
            )
            logger.info("  ✓ Repository cloned successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to clone aorta-report repository: {e}") from e

    return aorta_report_dir


def stage_push_results(
    aorta_report_dir: Path,
    experiment_dir: str,
    repo_root: Path,
    logger: logging.Logger,
    git_user_name: str = "Weekly CI Bot",
    git_user_email: str = "weekly-ci@aorta.local",
) -> None:
    """Copy experiment results to aorta-report and push.

    Creates a date-based directory structure in aorta-report:
    aorta-report/YYYY-MM-DD/rccl-warp-speed/

    Args:
        aorta_report_dir: Path to the aorta-report repository.
        experiment_dir: Path to the experiment directory (relative to repo_root).
        repo_root: Path to the aorta repository root.
        logger: Logger instance.
        git_user_name: Git user name for commit.
        git_user_email: Git user email for commit.

    Raises:
        RuntimeError: If push fails.
    """
    logger.info("Pushing results to aorta-report...")

    # Get today's date for directory name
    today = datetime.now().strftime("%Y-%m-%d")
    target_dir = aorta_report_dir / today / "rccl-warp-speed"

    logger.info(f"  Date directory: {today}")
    logger.info(f"  Target path: {target_dir}")

    # Source experiment directory
    source_dir = repo_root / experiment_dir
    if not source_dir.exists():
        raise RuntimeError(f"Experiment directory not found: {source_dir}")

    try:
        # Create target directory
        target_dir.mkdir(parents=True, exist_ok=True)

        # Copy experiment results
        logger.info("  Copying experiment results...")
        _copy_experiment_results(source_dir, target_dir, logger)

        # Configure git user
        run_command(
            f'git config user.name "{git_user_name}"',
            logger,
            cwd=aorta_report_dir,
            capture_output=True,
            check=True,
        )
        run_command(
            f'git config user.email "{git_user_email}"',
            logger,
            cwd=aorta_report_dir,
            capture_output=True,
            check=True,
        )

        # Pull latest changes
        logger.info("  Pulling latest changes...")
        run_command(
            "git pull --rebase origin main",
            logger,
            cwd=aorta_report_dir,
            capture_output=True,
            check=False,  # May fail if no upstream
        )

        # Add and commit
        logger.info("  Committing changes...")
        run_command(
            f"git add {today}",
            logger,
            cwd=aorta_report_dir,
            capture_output=True,
            check=True,
        )

        # Check if there are changes to commit
        result = run_command(
            "git status --porcelain",
            logger,
            cwd=aorta_report_dir,
            capture_output=True,
            check=True,
        )

        if result.stdout.strip():
            run_command(
                f'git commit -m "Add RCCL warp speed results for {today}"',
                logger,
                cwd=aorta_report_dir,
                capture_output=True,
                check=True,
            )

            # Push
            logger.info("  Pushing to remote...")
            run_command(
                "git push origin main",
                logger,
                cwd=aorta_report_dir,
                capture_output=True,
                check=True,
            )
            logger.info("  ✓ Results pushed successfully")
        else:
            logger.info("  No changes to commit (results may already exist)")

    except Exception as e:
        raise RuntimeError(f"Failed to push results: {e}") from e


def _copy_experiment_results(
    source_dir: Path,
    target_dir: Path,
    logger: logging.Logger,
) -> None:
    """Copy experiment results to target directory.

    Copies important result files while excluding large trace files.

    Args:
        source_dir: Source experiment directory.
        target_dir: Target directory in aorta-report.
        logger: Logger instance.
    """
    # Patterns to exclude (large files we don't want in the report)
    exclude_patterns = [
        "*.rpd",  # ROCm profiler data
        "*.sqlite",
        "*.db",
        "traces/",  # Raw trace directories
    ]

    # Patterns to include (prioritize these)
    include_patterns = [
        "*.json",
        "*.csv",
        "*.xlsx",
        "*.html",
        "*.png",
        "*.md",
        "*.txt",
        "summary/",
        "comparison_results/",
        "cross_timestamp_comparison/",
        "compare_all_runs/",
    ]

    items_copied = 0

    for item in source_dir.iterdir():
        # Check exclusions
        should_exclude = False
        for pattern in exclude_patterns:
            if pattern.endswith("/"):
                # Directory pattern
                if item.is_dir() and item.name == pattern[:-1]:
                    should_exclude = True
                    break
            elif item.match(pattern):
                should_exclude = True
                break

        if should_exclude:
            logger.debug(f"    Excluding: {item.name}")
            continue

        # Copy item
        dest = target_dir / item.name
        try:
            if item.is_dir():
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(item, dest, ignore=shutil.ignore_patterns(*exclude_patterns))
                logger.debug(f"    Copied directory: {item.name}")
            else:
                shutil.copy2(item, dest)
                logger.debug(f"    Copied file: {item.name}")
            items_copied += 1
        except Exception as e:
            logger.warning(f"    Failed to copy {item.name}: {e}")

    logger.info(f"  Copied {items_copied} items")

