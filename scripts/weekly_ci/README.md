# Weekly CI Kickoff

A standalone Python script that replicates the RCCL Warp Speed Performance Analysis CI workflow for local execution.

## Overview

This script automates the entire performance testing and analysis pipeline:
1. Docker container setup
2. RCCL library build
3. Performance test execution
4. Pairwise and cross-timestamp analysis
5. Summary report generation
6. Results publishing to aorta-report repository

## Prerequisites

### System Requirements

- **Docker** (with Docker Compose v2)
- **Python 3.8+**
- **Git** (for aorta-report operations)
- **GPU**: AMD GPU with ROCm support (gfx950 or gfx942)

### Python Dependencies

The script uses only standard library modules plus PyYAML:

```bash
pip install pyyaml
```

Or install the full aorta package:

```bash
pip install -e .
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `DOCKER_PASSWORD` | For private images | Docker registry password for pulling base images |
| `AORTA_REPORT_GITHUB_TOKEN` | For push | GitHub token for pushing results to aorta-report |

### Setting Environment Variables

```bash
# Docker registry authentication (for pulling private base images)
export DOCKER_PASSWORD="your-docker-registry-token"

# GitHub token for pushing results (optional, only if --skip-push is not used)
export AORTA_REPORT_GITHUB_TOKEN="ghp_your_github_token"
```

## Quick Start

### Basic Usage

```bash
# Run from repository root with all defaults
python scripts/weekly_ci_kickoff.py

# Run with custom config file
python scripts/weekly_ci_kickoff.py --config path/to/config.yaml
```

### Common Scenarios

#### First-time setup (full run)
```bash
python scripts/weekly_ci_kickoff.py --docker-build
```

#### Skip container rebuild (container already running)
```bash
python scripts/weekly_ci_kickoff.py --skip-docker-setup --skip-rccl-build
```

#### Run only performance tests (dependencies already installed)
```bash
python scripts/weekly_ci_kickoff.py \
    --skip-docker-setup \
    --skip-rccl-build \
    --skip-install-deps
```

#### Run analysis on existing experiment
```bash
python scripts/weekly_ci_kickoff.py \
    --skip-docker-setup \
    --skip-rccl-build \
    --skip-install-deps \
    --skip-performance-tests
```

#### Custom configuration pairs
```bash
python scripts/weekly_ci_kickoff.py \
    --config-pairs "56,256 37,384 32,512" \
    --baseline "56,256"
```

## Configuration

### Configuration File

Default configuration is at `config/weekly_ci.yaml`:

```yaml
# RCCL Build Configuration
rccl:
  branch: "warp_speed_v1"
  gpu_target: "gfx950"

# Test Configuration
test:
  config_pairs: "56,256 37,384 32,512"  # CU,threads pairs
  baseline: "56,256"                     # Baseline for comparisons
  training_config: "config/single_node/gemm_overlap_comm.yaml"
  experiment_dir: ""                     # Explicit dir (auto-detect if empty)

# Docker Configuration
docker:
  compose_file: "docker/rccl_test/docker-compose.rocm70_9-1.yaml"
  container_name: "training-overlap-bugs-rocm70_9-1"
  registry_user: "rocmshared"
  registry_password: ""  # Use DOCKER_PASSWORD env var
  skip_build: true       # Skip docker build by default

# Stage Skip Configuration
skip:
  docker_setup: false
  rccl_build: false
  install_deps: false
  performance_tests: false
  pairwise_analysis: false
  compare_all_analysis: true    # Expensive, skip by default
  checkout_aorta_report: false
  cross_timestamp_comparison: false
  push_results: true            # Avoid accidental pushes
  cleanup: true                 # Keep container running

# Cross-Timestamp Comparison
cross_timestamp:
  baseline_experiment: ""       # Auto-detect if empty
  baseline_date: ""             # Date in aorta-report (e.g., "2026-02-19")
  aorta_report_path: "../aorta-report"

# Analysis Labels (for aorta-report commands)
analysis:
  baseline_label: ""            # e.g., "baseline", "v1.0"
  test_label: ""                # e.g., "test", "v1.1"

# Git Configuration
git:
  user_name: "Weekly CI Bot"
  user_email: "weekly-ci@aorta.local"
  github_token: ""              # Use AORTA_REPORT_GITHUB_TOKEN env var

# Output Configuration
output:
  log_dir: "logs"
  log_level: "INFO"
```

### Command-Line Arguments

All config file options can be overridden via CLI. CLI arguments take precedence.

```
Usage: python scripts/weekly_ci_kickoff.py [OPTIONS]

Test Configuration:
  --config-pairs        Space-separated CU,threads pairs (e.g., "56,256 37,384")
  --baseline            Baseline configuration (e.g., "56,256")
  --training-config     Path to training config YAML
  --experiment-dir      Explicit experiment directory (auto-detect if not specified)

RCCL Configuration:
  --rccl-branch         RCCL branch to test
  --gpu-target          GPU architecture (gfx950, gfx942)

Docker Configuration:
  --compose-file        Docker compose file path
  --container-name      Docker container name
  --docker-user         Docker registry username
  --docker-password     Docker registry password
  --docker-build        Build Docker image (default: skip)
  --no-docker-build     Skip Docker build (default behavior)

Skip Stages:
  --skip-docker-setup   Skip Docker setup
  --skip-rccl-build     Skip RCCL build
  --skip-install-deps   Skip dependency installation
  --skip-performance-tests  Skip performance tests
  --skip-pairwise-analysis  Skip pairwise analysis
  --skip-compare-all    Skip compare-all analysis (default: skipped)
  --no-skip-compare-all Enable compare-all analysis
  --skip-checkout-aorta-report  Skip aorta-report checkout
  --skip-cross-timestamp  Skip cross-timestamp comparison
  --skip-push           Skip pushing results

Cross-Timestamp Options:
  --baseline-experiment Previous experiment directory (local)
  --baseline-date       Date directory in aorta-report (e.g., "2026-02-19")
  --aorta-report-path   Path to aorta-report repository

Analysis Labels:
  --baseline-label      Label for baseline in reports (e.g., "v1.0")
  --test-label          Label for test in reports (e.g., "v1.1")

Other Options:
  --cleanup             Cleanup container after completion
  --git-user-name       Git user name for commits
  --git-user-email      Git user email for commits
  --github-token        GitHub token for aorta-report
  --log-level           Logging level (DEBUG/INFO/WARNING/ERROR)
  --log-dir             Directory for log files
```

## Pipeline Stages

The script executes these stages in order:

| Stage | Description | Skippable |
|-------|-------------|-----------|
| 1. Validate Environment | Check Docker, paths, config | No |
| 2. Docker Setup | Start container, optional build | Yes |
| 3. Build RCCL | Clone and build RCCL library | Yes |
| 4. Install Dependencies | Install Python packages | Yes |
| 5. Run Performance Tests | Execute RCCL warp speed tests | Yes |
| 6. Find Experiment Dir | Locate test results | Auto |
| 7. Pairwise Analysis | Compare baseline vs each config | Yes |
| 8. Compare All Analysis | Multi-config comparison | Yes (default: skip) |
| 9. Checkout aorta-report | Clone/update report repo | Yes |
| 10. Cross-Timestamp | Compare with previous run | Yes |
| 11. Generate Summary | Create summary report | No |
| 12. Push Results | Push to aorta-report | Yes (default: skip) |
| 13. Cleanup | Stop container | Yes (default: skip) |

### Data Flow Between Stages

Stages discover their inputs from the filesystem or previous stages:

| Stage | Input Source | Override Option |
|-------|-------------|-----------------|
| 6. Find Experiment Dir | Scans `experiments/rccl_warp_speed_*` | `--experiment-dir` |
| 7-8. Analysis | Uses experiment dir from Stage 6 | (automatic) |
| 10. Cross-Timestamp | Local experiments or aorta-report | `--baseline-experiment` or `--baseline-date` |

#### Explicit Experiment Directory

To run analysis on a specific experiment (instead of auto-detecting):

```bash
python scripts/weekly_ci_kickoff.py \
    --experiment-dir "experiments/rccl_warp_speed_20260220_100000" \
    --skip-performance-tests
```

#### Cross-Timestamp Baseline Sources

For cross-timestamp comparison, the baseline can come from:

1. **Local experiment** (default: auto-detect second-most-recent):
   ```bash
   --baseline-experiment "experiments/rccl_warp_speed_20260219_100000"
   ```

2. **aorta-report date directory**:
   ```bash
   --baseline-date "2026-02-19"
   # Looks in: aorta-report/2026-02-19/rccl-warp-speed/
   ```

#### Custom Labels for Reports

Add labels to distinguish baseline and test in reports:

```bash
python scripts/weekly_ci_kickoff.py \
    --baseline-label "v1.0-stable" \
    --test-label "v1.1-experimental"
```

## Expected Output

### Directory Structure

After a successful run, you'll find:

```
experiments/rccl_warp_speed_YYYYMMDD_HHMMSS/
├── 56cu_256threads/              # Baseline configuration
│   ├── traces/                   # Raw trace data
│   ├── logs/                     # Execution logs
│   ├── metrics.json              # Performance metrics
│   └── summary/                  # aorta-report summary
├── 37cu_384threads/              # Test configuration 1
│   └── summary/
├── 32cu_512threads/              # Test configuration 2
│   └── summary/
├── comparison_results/           # Pairwise comparisons
│   ├── baseline_vs_37cu_384threads/
│   │   ├── summary.json
│   │   ├── comparison.xlsx
│   │   └── plots/
│   └── baseline_vs_32cu_512threads/
├── cross_timestamp_comparison/   # Comparison with previous run
│   ├── 56cu_256threads/
│   ├── 37cu_384threads/
│   └── 32cu_512threads/
├── compare_all_runs/             # Optional multi-config comparison
└── summary.txt                   # Human-readable summary

logs/
├── weekly_ci_YYYYMMDD_HHMMSS.log # Full execution log
└── latest.log -> weekly_ci_*.log # Symlink to latest
```

### Log Files

Logs are written to both console (colored) and file:

```
logs/weekly_ci_20260220_143022.log
```

Log format:
```
2026-02-20 14:30:22,554 - INFO - ============================================================
2026-02-20 14:30:22,554 - INFO - STAGE: 1. Validate Environment
2026-02-20 14:30:22,554 - INFO - ============================================================
2026-02-20 14:30:22,554 - INFO - Validating execution environment...
2026-02-20 14:30:22,597 - INFO -   ✓ Docker is available and running
```

### Summary Report

A `summary.txt` is generated in the experiment directory:

```
======================================================================
RCCL Warp Speed Performance Analysis Summary
======================================================================

Generated: 2026-02-20 14:30:22

----------------------------------------------------------------------
Configuration
----------------------------------------------------------------------
Experiment Directory: experiments/rccl_warp_speed_20260220_143022
RCCL Branch: warp_speed_v1
GPU Target: gfx950
Baseline: 56,256 (CU,Threads)

----------------------------------------------------------------------
Tested Configurations
----------------------------------------------------------------------
  ✓ CU=56, Threads=256 (baseline)
  ✓ CU=37, Threads=384
  ✓ CU=32, Threads=512

----------------------------------------------------------------------
Generated Artifacts
----------------------------------------------------------------------
  Summary Reports:
    - 56cu_256threads/summary/
    - 37cu_384threads/summary/
    - 32cu_512threads/summary/
  Comparison Results:
    - comparison_results/baseline_vs_37cu_384threads/
    - comparison_results/baseline_vs_32cu_512threads/
```

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Failure (check logs for details) |

## Troubleshooting

### Docker Issues

**Error: Docker not running**
```bash
sudo systemctl start docker
```

**Error: Permission denied**
```bash
sudo usermod -aG docker $USER
# Log out and back in
```

**Error: Docker login failed**
```bash
# Set the password via environment variable
export DOCKER_PASSWORD="your-token"
# Or via CLI
python scripts/weekly_ci_kickoff.py --docker-password "your-token"
```

### Git/Push Issues

**Error: Authentication failed for aorta-report**
```bash
# Set GitHub token
export AORTA_REPORT_GITHUB_TOKEN="ghp_your_token"
# Or use SSH (requires SSH key setup)
```

**Error: aorta-report not found**
```bash
# Specify path explicitly
python scripts/weekly_ci_kickoff.py --aorta-report-path /path/to/aorta-report
```

### Performance Test Issues

**Error: No experiment directory found**
- Check that performance tests completed successfully
- Verify the container is running: `docker ps`
- Check container logs: `docker logs <container_name>`
- Use `--experiment-dir` to specify an explicit directory

### Resume After Failure

If the script fails partway through, you can resume by skipping completed stages:

```bash
# Example: Tests completed but analysis failed
python scripts/weekly_ci_kickoff.py \
    --skip-docker-setup \
    --skip-rccl-build \
    --skip-install-deps \
    --skip-performance-tests
```

## Development

### Project Structure

```
scripts/
├── weekly_ci_kickoff.py      # Main entry point
└── weekly_ci/
    ├── __init__.py           # Package exports
    ├── config.py             # Configuration management
    ├── logging_config.py     # Logging setup
    ├── utils.py              # Utility functions
    └── stages/
        ├── __init__.py       # Stage exports
        ├── validate.py       # Environment validation
        ├── docker.py         # Docker operations
        ├── build.py          # RCCL build
        ├── test.py           # Performance tests
        ├── analysis.py       # Analysis stages
        ├── repository.py     # Git operations
        └── reporting.py      # Summary generation
```

### Adding New Stages

1. Create stage function in appropriate module under `stages/`
2. Export from `stages/__init__.py`
3. Add to main script with proper skip logic
4. Update config if needed

## License

See repository LICENSE file.

