# GEMM Sweep Profiling

Profile GEMM kernel performance across multiple NCCL configurations.

## Prerequisites

- AMD GPU with ROCm (latest stable recommended)
- PyTorch with ROCm support ([install guide](https://pytorch.org/get-started/locally/))
- `aorta-report` package installed
- [TraceLens](https://github.com/AMD-AGI/TraceLens) -- AMD-AGI's trace analysis library, used by `aorta-report` to generate per-rank performance reports and collective NCCL reports from PyTorch profiler or ROCprof traces

```bash
# Install aorta-report with TraceLens in one step
pip install -e "packages/aorta-report/[tracelens]"

# Or install them separately
pip install -e packages/aorta-report/
pip install git+https://github.com/AMD-AGI/TraceLens.git
```

Works on bare-metal or inside any PyTorch ROCm container (e.g., `rocm/pytorch`).

## Pipeline Steps

### 1. Run Training Sweep

```bash
bash packages/aorta-report/scripts/gemm_analysis/run_train_various_channels.sh \
  --channels 28,42,56,70 \
  --threads 256,512 \
  --config config/single_node/gemm_overlap_comm.yaml
```

#### rocprof Tracing Options

Add rocprofv3 kernel tracing to capture detailed GEMM performance. Two YAML configs are provided:

Two rocprofv3 configs are provided:

**`rocprof_cu_only.yaml`** -- Minimal, CU utilization counters only:

```yaml
jobs:
  - kernel_include_regex: "(gemm|Cijk_.*)"
    kernel_trace: true
    output_format: [csv]
    pmc:
      - SQ_BUSY_CU_CYCLES     # CU utilization
      - SQ_WAVES               # Active waves (occupancy)
      - SQ_WAVE_CYCLES         # Total wave cycles
      - SQ_INSTS_MFMA          # Matrix ops (GEMM-specific)
```

**`rocprof_input.yaml`** -- Full counters + timing stats + Perfetto traces:

```yaml
jobs:
  - kernel_include_regex: "(gemm|Cijk_.*)"
    kernel_trace: true
    stats: true                             # timing statistics
    output_format: [json, csv, perfetto]    # generates .pftrace for Chrome tracing
    sys_trace: false
    advanced_thread_trace: false            # leave false unless ATT decoder is installed
    pmc:
      - SQ_BUSY_CU_CYCLES                  # plus many more counters
      # ... (see file for full list)
```

Run the sweep with the CU-only YAML (recommended starting point):

```bash
bash packages/aorta-report/scripts/gemm_analysis/run_train_various_channels.sh \
  --rocprof \
  --rocprof-input packages/aorta-report/scripts/gemm_analysis/rocprof_cu_only.yaml \
  --channels 28,42,56 --threads 256,512 \
  --config config/single_node/gemm_overlap_comm.yaml
```

Notes:
- Kernel filtering and counters come from the YAML. The current rocprofv3 build ignores CLI kernel filters, so use the YAML to include/exclude kernels.
- Keep `advanced_thread_trace: false` unless the ATT decoder debs are installed.
- `stats: true` only collects timing statistics, NOT hardware counter metrics. For CU utilization, use the `pmc:` section.
- **Output Files**: rocprof generates 5 files per rank/process:
  - `PID_agent_info.csv`: Hardware information about CPUs and GPUs
  - `PID_counter_collection.csv`: **Main file with CU utilization metrics** (focus on this)
  - `PID_kernel_trace.csv`: Kernel execution timeline data
  - `PID_results.json`: Chrome trace format for visualization
  - `PID_results.csv`: Summary statistics

**Analyzing Unique GEMM Kernels (counter_collection.csv columns):**
- `Grid_Size`: Total number of workgroups in the kernel launch
- `Kernel_Name`: Name of the GEMM kernel (e.g., Cijk_Alik_Bljk_SB_MT128x128x32_MI32x32x1x2)
- `Workgroup_Size`: Number of work-items per workgroup
- `LDS_Block_Size`: Local Data Share memory allocation per workgroup
- `Scratch_Size`: Private memory allocation per work-item
- `VGPR_Count`: Vector General Purpose Registers used
- `Accum_VGPR_Count`: Accumulator VGPRs (for matrix operations)
- `SGPR_Count`: Scalar General Purpose Registers used
- `Counter_Name`: Performance counter being measured (e.g., SQ_BUSY_CU_CYCLES)
- `Counter_Value`: Value of the performance counter
- `Start_Timestamp` / `End_Timestamp`: Kernel execution timing

**Key Options:**
- `--rocprof` : Enable rocprofv3 tracing
- `--stats` : Include timing statistics (not CU utilization)
- `--channels VALUES` : Comma-separated NCCL channel values
- `--threads VALUES` : Comma-separated thread values

**Output:** Traces saved to `rocprof_traces/` in each run directory.

**Key Performance Counters (found in counter_collection.csv files):**
- `SQ_BUSY_CU_CYCLES`: Percentage of time CUs are active (CU utilization)
- `SQ_WAVES`: Number of active wavefronts (occupancy indicator)
- `SQ_INSTS_MFMA`: Matrix FMA instructions (critical for GEMM performance)
- `SQ_INSTS_VALU`: Vector ALU instructions (general compute)

### 2. Generate TraceLens Reports

TraceLens can analyze both PyTorch profiler traces and ROCprof traces.

**For PyTorch profiler traces (default):**
```bash
bash packages/aorta-report/scripts/gemm_analysis/run_tracelens_analysis.sh experiments/sweep_20251124_222204
```

**For ROCprof traces:**
```bash
bash packages/aorta-report/scripts/gemm_analysis/run_tracelens_analysis.sh experiments/sweep_20251217_103450 --rocprof
```

The `--rocprof` flag:
- Processes `*_results.json` files from `rocprof_traces/pass_1/` directories
- Uses `TraceLens_generate_perf_report_rocprof` command
- Generates individual reports per rank (no collective reports for ROCprof)
- Provides detailed kernel-level performance metrics including grid/block dimensions

### 3. Run GEMM Analysis Pipeline

Use the `aorta-report` CLI to run the full GEMM analysis pipeline:

```bash
aorta-report pipeline gemm \
  --sweep-dir experiments/sweep_YYYYMMDD_HHMMSS \
  -o experiments/sweep_YYYYMMDD_HHMMSS/analysis \
  -t 256 -t 512 -c 28 -c 42 -c 56 -c 70 \
  --top-k 5
```

If TraceLens reports already exist, skip re-running them:

```bash
aorta-report pipeline gemm \
  --sweep-dir experiments/sweep_YYYYMMDD_HHMMSS \
  -o experiments/sweep_YYYYMMDD_HHMMSS/analysis \
  --skip-tracelens
```

Or run individual steps:

```bash
# Extract top GEMM kernels
aorta-report analyze gemm \
  experiments/sweep_YYYYMMDD_HHMMSS/tracelens_analysis \
  -t 256 -t 512 -c 28 -c 42 -c 56 -c 70 --top-k 5

# Enhance with timestamps
aorta-report process gemm-variance \
  experiments/sweep_YYYYMMDD_HHMMSS/tracelens_analysis/top5_gemm_kernels_time_variance.csv \
  --base-path experiments/sweep_YYYYMMDD_HHMMSS

# Generate variance plots
aorta-report generate plots --type gemm \
  -i experiments/sweep_YYYYMMDD_HHMMSS/tracelens_analysis/top5_gemm_kernels_time_variance.csv \
  -o experiments/sweep_YYYYMMDD_HHMMSS/tracelens_analysis/plots

# Process GPU timeline data
aorta-report process gpu-timeline \
  experiments/sweep_YYYYMMDD_HHMMSS --mode sweep

# Process NCCL communication data
aorta-report process comms \
  experiments/sweep_YYYYMMDD_HHMMSS

# Generate sweep comparison HTML
aorta-report generate html --mode sweep \
  --sweep1 experiments/sweep1 --sweep2 experiments/sweep2 \
  --label1 "Baseline" --label2 "Optimized" -o comparison.html
```

### 4. Analyze Collective Overlap (standalone script)

This analysis is not yet in the CLI. Use the standalone script to identify NCCL
collective operations overlapping with GEMM kernels:

```bash
python packages/aorta-report/scripts/gemm_analysis/gemm_report_with_collective_overlap.py \
  --input-csv experiments/sweep_YYYYMMDD_HHMMSS/tracelens_analysis/top5_gemm_kernels_time_variance_with_timestamps.csv \
  --tracelens-path experiments/sweep_YYYYMMDD_HHMMSS/tracelens_analysis
```

Output: `top5_gemm_kernels_time_variance_with_collective_overlap.csv`

## Output Structure

```
experiments/sweep_YYYYMMDD_HHMMSS/
├── 256thread/
│   └── nccl_XXchannels/
│       ├── torch_profiler/rank*/
│       ├── rocprof_traces/           # if --rocprof flag used
│       │   ├── PID_agent_info.csv    # Hardware info for each rank
│       │   ├── PID_counter_collection.csv  # CU utilization metrics (main focus)
│       │   ├── PID_kernel_trace.csv  # Kernel execution timeline
│       │   ├── PID_results.json      # Chrome trace format
│       │   └── PID_results.csv       # Summary statistics
│       └── run_output.log
├── 512thread/
│   └── nccl_XXchannels/
└── tracelens_analysis/
    ├── 256thread/
    │   ├── individual_reports/perf_*ch_rank*.xlsx
    │   └── collective_reports/collective_*ch.xlsx
    ├── 512thread/
    ├── top5_gemm_kernels_time_variance.csv
    ├── top5_gemm_kernels_time_variance_with_timestamps.csv
    ├── top5_gemm_kernels_time_variance_with_collective_overlap.csv
    ├── gpu_timeline_all_configs_mean.xlsx
    ├── nccl_master_all_configs.xlsx
    └── plots/
        ├── variance_by_threads_boxplot.png
        ├── variance_by_channels_boxplot.png
        ├── variance_by_ranks_boxplot.png
        ├── variance_violin_combined.png
        └── variance_thread_channel_interaction.png
```

## Scripts in this Directory

| Script | Purpose |
|--------|---------|
| `run_train_various_channels.sh` | Shell orchestration for NCCL channel/thread sweep training runs |
| `run_tracelens_analysis.sh` | Shell orchestration for TraceLens report generation on sweeps |
| `gemm_report_with_collective_overlap.py` | Adds NCCL collective overlap info to GEMM variance CSV |
| `rocprof_cu_only.yaml` | rocprofv3 config for CU utilization tracing |
| `rocprof_input.yaml` | rocprofv3 config (general) |
