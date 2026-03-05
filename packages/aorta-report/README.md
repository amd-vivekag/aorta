# aorta-report

A CLI tool for analyzing PyTorch profiler traces, processing GPU timeline data, and generating comprehensive performance comparison reports.

For the complete command reference, all options, and advanced workflows, see the [User Guide](docs/user-guide.md).

---

## Installation

Requires Python >= 3.10 and PyTorch with ROCm support.
Works on bare-metal or inside any PyTorch ROCm container (e.g., [`rocm/pytorch`](https://hub.docker.com/r/rocm/pytorch)).

```bash
# Install PyTorch with ROCm support
# Pick the command for your ROCm version from https://pytorch.org/get-started/locally/
# For nightly (e.g. ROCm 7.2):
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/rocm7.2

# Install aorta-report with TraceLens (recommended)
cd /path/to/aorta
pip install -e "packages/aorta-report/[tracelens]"

# Verify
aorta-report --version
```

[TraceLens](https://github.com/AMD-AGI/TraceLens) is AMD-AGI's trace analysis library.
It is required for the `analyze` commands but not for `process`, `compare`, or `generate`.
If you only need post-processing on existing TraceLens reports, install without it:

```bash
pip install -e packages/aorta-report/
```

<details>
<summary>Other installation methods</summary>

```bash
# Install from GitHub (no local clone needed)
pip install "git+https://github.com/ROCm/aorta.git#subdirectory=packages/aorta-report"

# Install with uv
cd /path/to/aorta
uv pip install -e packages/aorta-report/
```

</details>

---

## Quick Start

### What are traces?

`aorta-report` analyzes **traces** — JSON files produced by [PyTorch Profiler](https://pytorch.org/docs/stable/profiler.html) during a training run. Each rank in a distributed training job produces one trace file.

To generate traces, enable profiling in your training config:

```yaml
profiling:
  enabled: true
  wait: 1
  warmup: 1
  active: 30
  repeat: 1
  record_shapes: true
  profile_memory: true
  chrome_trace: true
```

This produces a directory like:

```
my_experiment/
└── torch_profiler/
    ├── rank0/trace/pt.trace.json
    ├── rank1/trace/pt.trace.json
    └── ...
```

> **Need a working example?** The [GEMM Sweep Scripts](scripts/gemm_analysis/README.md) walk through the full flow end-to-end: running a training sweep with profiling enabled, generating TraceLens reports, and analyzing the results.

### A/B Comparison (baseline vs test)

Compare two training runs (e.g., before and after a configuration change):

```bash
aorta-report pipeline summary \
    -b /path/to/baseline_experiment/ \
    -t /path/to/test_experiment/ \
    -o ./comparison_output/
```

### Single Configuration Analysis

Analyze one training run without comparison:

```bash
aorta-report pipeline summary \
    -t /path/to/my_experiment/ \
    -o ./output/
```

### GEMM Variance Analysis

A **sweep directory** contains traces from multiple training runs with different NCCL thread/channel configurations (e.g., `256thread/nccl_28channels/`). This pipeline analyzes how GEMM kernel performance varies across those configurations:

```bash
aorta-report pipeline gemm \
    --sweep-dir /path/to/sweep/ \
    -o ./gemm_output/
```

---

## What You Get

Each pipeline generates a complete set of reports:

| Pipeline | Key Outputs |
|----------|-------------|
| `pipeline summary` | `final_analysis_report.xlsx`, comparison plots, `performance_analysis_report.html` |
| `pipeline gemm` | GEMM variance CSV, variance plots, `gemm_variance_report.html` |

HTML reports are self-contained and can be shared directly without any dependencies.

---

## Common Options

Use `--skip-tracelens` on subsequent runs if TraceLens reports already exist (saves time):

```bash
aorta-report pipeline summary \
    -b /path/to/baseline -t /path/to/test -o ./output/ \
    --skip-tracelens
```

Use `--help` on any command to see all available options:

```bash
aorta-report --help
aorta-report pipeline summary --help
aorta-report pipeline gemm --help
```

---

## Further Reading

| Document | Description |
|----------|-------------|
| [User Guide](docs/user-guide.md) | Complete command reference, all options, workflows, and troubleshooting |
| [GEMM Sweep Scripts](scripts/gemm_analysis/README.md) | GEMM sweep profiling scripts and end-to-end example |
| [Single Config Scripts](scripts/tracelens_single_config/README.md) | Single-configuration analysis scripts |
