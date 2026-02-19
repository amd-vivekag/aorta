#!/usr/bin/env python3
"""
Weekly CI Kickoff - RCCL Warp Speed Performance Analysis

This script replicates the CI workflow locally for:
- Building Docker containers
- Cloning and building RCCL
- Running performance tests
- Performing pairwise and cross-timestamp analysis
- Pushing results to aorta-report repository

Usage:
    python scripts/weekly_ci_kickoff.py [OPTIONS]

Examples:
    # Run with defaults
    python scripts/weekly_ci_kickoff.py

    # Skip docker and rccl build (container already running)
    python scripts/weekly_ci_kickoff.py --skip-docker-setup --skip-rccl-build

    # Custom config pairs
    python scripts/weekly_ci_kickoff.py --config-pairs "56,256 37,384"

    # Use custom config file
    python scripts/weekly_ci_kickoff.py --config my_config.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add scripts directory to path for imports
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from weekly_ci import (
    Config,
    load_config_file,
    log_stage_complete,
    log_stage_error,
    log_stage_skip,
    log_stage_start,
    merge_config,
    parse_args,
    setup_logging,
)


def main() -> int:
    """Main entry point for Weekly CI Kickoff.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    # Parse arguments
    args = parse_args()

    # Load configuration
    config = Config()
    yaml_data = load_config_file(args.config)
    config = merge_config(config, yaml_data, args)

    # Setup logging
    logger = setup_logging(config.output.log_dir, config.output.log_level)

    logger.info("=" * 60)
    logger.info("Weekly CI Kickoff - RCCL Warp Speed Performance Analysis")
    logger.info("=" * 60)
    logger.info("")

    # Log configuration
    logger.info("Configuration:")
    logger.info(f"  Config file: {args.config}")
    logger.info(f"  RCCL branch: {config.rccl.branch}")
    logger.info(f"  GPU target: {config.rccl.gpu_target}")
    logger.info(f"  Config pairs: {config.test.config_pairs}")
    logger.info(f"  Baseline: {config.test.baseline}")
    logger.info(f"  Docker compose: {config.docker.compose_file}")
    logger.info(f"  Container: {config.docker.container_name}")
    logger.info("")

    # Log skip configuration
    logger.info("Stages to skip:")
    if config.skip.docker_setup:
        logger.info("  - docker_setup")
    if config.skip.rccl_build:
        logger.info("  - rccl_build")
    if config.skip.install_deps:
        logger.info("  - install_deps")
    if config.skip.performance_tests:
        logger.info("  - performance_tests")
    if config.skip.pairwise_analysis:
        logger.info("  - pairwise_analysis")
    if config.skip.compare_all_analysis:
        logger.info("  - compare_all_analysis")
    if config.skip.checkout_aorta_report:
        logger.info("  - checkout_aorta_report")
    if config.skip.cross_timestamp_comparison:
        logger.info("  - cross_timestamp_comparison")
    if config.skip.push_results:
        logger.info("  - push_results")
    if config.skip.cleanup:
        logger.info("  - cleanup")
    logger.info("")

    # Track overall success
    success = True

    try:
        # Stage 1: Validate Environment
        log_stage_start(logger, "1. Validate Environment")
        # TODO: Implement in Phase 2
        logger.info("Stage implementation pending (Phase 2)")
        log_stage_complete(logger, "Validate Environment")

        # Stage 2: Docker Setup
        if config.skip.docker_setup:
            log_stage_skip(logger, "2. Docker Setup")
        else:
            log_stage_start(logger, "2. Docker Setup")
            # TODO: Implement in Phase 2
            logger.info("Stage implementation pending (Phase 2)")
            log_stage_complete(logger, "Docker Setup")

        # Stage 3: Build RCCL
        if config.skip.rccl_build:
            log_stage_skip(logger, "3. Build RCCL")
        else:
            log_stage_start(logger, "3. Build RCCL")
            # TODO: Implement in Phase 2
            logger.info("Stage implementation pending (Phase 2)")
            log_stage_complete(logger, "Build RCCL")

        # Stage 4: Install Dependencies
        if config.skip.install_deps:
            log_stage_skip(logger, "4. Install Dependencies")
        else:
            log_stage_start(logger, "4. Install Dependencies")
            # TODO: Implement in Phase 2
            logger.info("Stage implementation pending (Phase 2)")
            log_stage_complete(logger, "Install Dependencies")

        # Stage 5: Run Performance Tests
        if config.skip.performance_tests:
            log_stage_skip(logger, "5. Run Performance Tests")
        else:
            log_stage_start(logger, "5. Run Performance Tests")
            # TODO: Implement in Phase 3
            logger.info("Stage implementation pending (Phase 3)")
            log_stage_complete(logger, "Run Performance Tests")

        # Stage 6: Find Experiment Directory
        log_stage_start(logger, "6. Find Experiment Directory")
        # TODO: Implement in Phase 3
        logger.info("Stage implementation pending (Phase 3)")
        log_stage_complete(logger, "Find Experiment Directory")

        # Stage 7: Pairwise Analysis
        if config.skip.pairwise_analysis:
            log_stage_skip(logger, "7. Pairwise Analysis")
        else:
            log_stage_start(logger, "7. Pairwise Analysis")
            # TODO: Implement in Phase 3
            logger.info("Stage implementation pending (Phase 3)")
            log_stage_complete(logger, "Pairwise Analysis")

        # Stage 8: Compare All Analysis
        if config.skip.compare_all_analysis:
            log_stage_skip(logger, "8. Compare All Analysis")
        else:
            log_stage_start(logger, "8. Compare All Analysis")
            # TODO: Implement in Phase 3
            logger.info("Stage implementation pending (Phase 3)")
            log_stage_complete(logger, "Compare All Analysis")

        # Stage 9: Checkout aorta-report
        if config.skip.checkout_aorta_report:
            log_stage_skip(logger, "9. Checkout aorta-report")
        else:
            log_stage_start(logger, "9. Checkout aorta-report")
            # TODO: Implement in Phase 4
            logger.info("Stage implementation pending (Phase 4)")
            log_stage_complete(logger, "Checkout aorta-report")

        # Stage 10: Cross-Timestamp Comparison
        if config.skip.cross_timestamp_comparison:
            log_stage_skip(logger, "10. Cross-Timestamp Comparison")
        else:
            log_stage_start(logger, "10. Cross-Timestamp Comparison")
            # TODO: Implement in Phase 4
            logger.info("Stage implementation pending (Phase 4)")
            log_stage_complete(logger, "Cross-Timestamp Comparison")

        # Stage 11: Generate Summary
        log_stage_start(logger, "11. Generate Summary")
        # TODO: Implement in Phase 4
        logger.info("Stage implementation pending (Phase 4)")
        log_stage_complete(logger, "Generate Summary")

        # Stage 12: Push Results
        if config.skip.push_results:
            log_stage_skip(logger, "12. Push Results")
        else:
            log_stage_start(logger, "12. Push Results")
            # TODO: Implement in Phase 4
            logger.info("Stage implementation pending (Phase 4)")
            log_stage_complete(logger, "Push Results")

        # Stage 13: Cleanup
        if config.skip.cleanup:
            log_stage_skip(logger, "13. Cleanup")
        else:
            log_stage_start(logger, "13. Cleanup")
            # TODO: Implement in Phase 2
            logger.info("Stage implementation pending (Phase 2)")
            log_stage_complete(logger, "Cleanup")

    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user")
        success = False
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        success = False

    # Final summary
    logger.info("")
    logger.info("=" * 60)
    if success:
        logger.info("✅ Weekly CI Kickoff completed successfully!")
    else:
        logger.error("❌ Weekly CI Kickoff failed!")
    logger.info("=" * 60)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

