# Magpie Integration

Aorta integrates with [Magpie](https://github.com/AMD-AGI/Magpie), AMD-AGI's GPU kernel evaluation toolkit, in one area:

- **Magpie report adapter** -- Import Magpie benchmark workspaces into aorta's report pipeline for unified analysis and comparison.

The report adapter reads Magpie's file-based workspace output and does not import any Magpie Python modules, so Magpie does not need to be installed for the adapter to work.

**GPU hardware control** (lock-clock, power-limit) is implemented natively in aorta using direct `rocm-smi` / `nvidia-smi` subprocess calls. It does not depend on Magpie.

## GPU Hardware Control (native)

GPU hardware control (lock-clock, power-limit) is implemented directly in `src/aorta/utils/gpu_control.py` using subprocess calls to `rocm-smi` (AMD) and `nvidia-smi` (NVIDIA). It does **not** depend on Magpie.

See the CLI flags `--lock-clocks` and `--power-limit` on the `run` and `sweep` commands, or use `GPUControlConfig` / `GPUControlManager` programmatically.

For full documentation, see the inline docstrings in `gpu_control.py`.

## Magpie Report Adapter

### Magpie workspace layout

Magpie benchmark runs produce workspaces with this structure:

```
benchmark_{framework}_{timestamp}/
    config.yaml
    benchmark_report.json
    inferencemax_result.json
    torch_trace/
    tracelens_rank0_csvs/
    tracelens_collective_csvs/
```

The adapter reads these workspaces and converts them into a format aorta's report pipeline can consume.

### CLI commands

All commands live under `aorta-report magpie`:

```bash
# List Magpie workspaces in a directory
aorta-report magpie list ./results

# Show normalized report for a workspace
aorta-report magpie show results/benchmark_vllm_20260301_120000

# Import a workspace into aorta-report format
aorta-report magpie import results/benchmark_vllm_20260301_120000 \
    -o aorta_output/run1

# Import and run TraceLens analysis on torch traces
aorta-report magpie import results/benchmark_vllm_20260301_120000 \
    -o aorta_output/run1 --run-tracelens

# Compare two Magpie benchmark runs
aorta-report magpie compare \
    -b results/benchmark_vllm_20260301_120000 \
    -t results/benchmark_vllm_20260301_140000

# Save comparison to JSON
aorta-report magpie compare \
    -b results/benchmark_vllm_20260301_120000 \
    -t results/benchmark_vllm_20260301_140000 \
    -o comparison.json
```

### Comparison output

The `compare` command reports throughput and latency deltas:

- **Throughput** (higher is better): `request_throughput`, `output_throughput`, `total_token_throughput`
- **Latency** (lower is better): `ttft`, `tpot`, `itl`, `e2el` (mean and p99)

Each metric is classified as `better` (>1% improvement), `worse` (>1% regression), or `similar`.

### Programmatic usage

```python
from aorta.report.magpie_adapter import (
    locate_magpie_workspaces,
    read_magpie_report,
    import_magpie_workspace,
    compare_magpie_reports,
)

# Find workspaces
workspaces = locate_magpie_workspaces("./results")

# Read a single report
report = read_magpie_report(workspaces[0])
print(report["throughput"]["request_throughput"])

# Import into aorta format
result = import_magpie_workspace(
    workspace=workspaces[0],
    output_dir="./aorta_output/run1",
    run_tracelens=True,
)

# Compare two runs
comparison = compare_magpie_reports(
    baseline_workspace=workspaces[1],
    test_workspace=workspaces[0],
)
print(comparison["summary"]["overall"])  # "improvement", "regression", or "neutral"
```

## Architecture

```
aorta
  ├── src/aorta/utils/gpu_control.py
  │     Native GPU control via rocm-smi / nvidia-smi subprocess calls
  │     Exposes GPUControlConfig + GPUControlManager
  │
  ├── src/aorta/hw_queue_eval/core/harness.py
  │     Uses GPUControlManager in run() and run_workload()
  │
  ├── src/aorta/hw_queue_eval/cli.py
  │     --lock-clocks / --power-limit flags
  │
  ├── src/aorta/report/magpie_adapter.py
  │     Reads Magpie workspace files (file I/O only, no Magpie imports)
  │
  └── src/aorta/report/cli.py
        aorta-report magpie list|show|import|compare
```

## API Reference

### `aorta.utils.gpu_control`

| Symbol | Type | Description |
| --- | --- | --- |
| `GPUControlConfig` | dataclass | Power limit, clock levels, device IDs, reset-on-exit |
| `GPUControlManager` | class | Context manager using direct subprocess calls for GPU control |
| `GPUControlManager.apply()` | method | Apply config, return hardware snapshot dict |
| `GPUControlManager.reset()` | method | Reset GPUs to defaults |
| `GPUControlManager.available` | property | `True` if control is enabled |

### `aorta.report.magpie_adapter`

| Function | Description |
| --- | --- |
| `locate_magpie_workspaces(results_dir)` | Find workspace dirs containing `benchmark_report.json` |
| `read_magpie_report(workspace)` | Read and normalize a Magpie benchmark report |
| `import_magpie_workspace(workspace, output_dir, ...)` | Copy workspace into aorta-compatible layout |
| `compare_magpie_reports(baseline, test)` | Compute throughput/latency deltas between two runs |
