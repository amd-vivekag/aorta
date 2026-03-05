# aorta-hw-queue

A framework for stress-testing GPU hardware queue scheduling with workloads requiring high stream concurrency (8, 16, 32+ concurrent streams).

For the complete command reference, all options, and advanced workflows, see the [User Guide](docs/user-guide.md).

---

## Purpose

Modern GPU workloads increasingly rely on multiple concurrent streams for overlapping compute, communication, and data movement. When the number of CUDA/HIP streams exceeds the number of physical hardware queues, streams must share queues and the scheduler's mapping decisions become critical. This package measures:

- Queue switch latency overhead
- Throughput scaling with stream count
- Scheduling behavior under high concurrency (convoy effect detection)
- Communication/compute overlap effectiveness

---

## Installation

Requires Python >= 3.10 and PyTorch with ROCm support.

```bash
# Install PyTorch with ROCm support
# Pick the command for your ROCm version from https://pytorch.org/get-started/locally/

# Install aorta-hw-queue (editable)
cd /path/to/aorta
pip install -e packages/aorta-hw-queue/

# Verify
python -m aorta.hw_queue_eval --version
python -m aorta.hw_queue_eval list
```

<details>
<summary>Other installation methods</summary>

```bash
# Install with profiling extras (matplotlib, seaborn)
pip install -e "packages/aorta-hw-queue/[profiling]"

# Install with uv
cd /path/to/aorta
uv pip install -e packages/aorta-hw-queue/
```

</details>

---

## Quick Start

```bash
# List available workloads
python -m aorta.hw_queue_eval list

# Run a single workload
python -m aorta.hw_queue_eval run hetero_kernels --streams 8

# Sweep across stream counts
python -m aorta.hw_queue_eval sweep hetero_kernels --streams 1,2,4,8,16

# Run all P0 (critical) workloads
python -m aorta.hw_queue_eval run-priority P0

# Compare results for regressions
python -m aorta.hw_queue_eval compare --baseline results_a.json --test results_b.json

# Profile with PyTorch profiler (Chrome trace + TensorBoard)
python -m aorta.hw_queue_eval run hetero_kernels --streams 8 --profile

# Multi-GPU: distribute streams across GPUs
python -m aorta.hw_queue_eval run hetero_kernels --streams 16 --multi-gpu
python -m aorta.hw_queue_eval run hetero_kernels --streams 16 --num-gpus 4
```

### Distributed Mode (real NCCL/RCCL collectives)

```bash
torchrun --nproc_per_node=8 -m aorta.hw_queue_eval run comms_compute_overlap \
    --streams 4 --real-collectives --async-op --backend nccl \
    --process-groups "[0,1,2,3,4,5,6,7]" \
    --mm-dim 4096,4096,4096 --num-compute 10 --comm-size 128M
```

---

## Workloads

15 workloads across four categories, each targeting different stream concurrency patterns:

| Category | Workloads | Key Question |
|---|---|---|
| **Latency-sensitive** | `hetero_kernels`, `tiny_kernel_stress`, `large_gemm_only`, `graph_subgraphs` | Does the scheduler avoid convoy effects? |
| **Distributed** | `comms_compute_overlap`, `fsdp_tp`, `moe`, `activation_ckpt`, `grad_accum` | Can comms and compute overlap effectively? |
| **Inference** | `speculative_decode`, `continuous_batch`, `rag_pipeline` | Does stream concurrency help latency-sensitive inference? |
| **Pipeline** | `async_dataload`, `zero_offload`, `torch_compile` | Can data movement hide behind compute? |

Use `python -m aorta.hw_queue_eval list -v` for detailed descriptions and recommended stream counts.

---

## Metrics

Each run collects:

- **Throughput**: GFLOPS, TFLOPS, tokens/sec, or GB/s (workload-dependent)
- **Latency**: Mean, P50, P95, P99 per iteration
- **Queue switch overhead**: Inter-stream vs intra-stream gap estimation
- **Memory**: Peak allocated, reserved
- **Scaling analysis**: Throughput per stream count, efficiency curves, inflection point detection

---

## Scripts

| Script | Description |
|---|---|
| `scripts/run_sweep.sh` | Run a stream-count sweep for any workload |
| `scripts/profile_queues.sh` | Profile with `rocprof` for hardware queue visibility |
| `scripts/compare_baselines.py` | Compare two result files for regressions (with threshold) |

---

## Further Reading

| Document | Description |
|----------|-------------|
| [User Guide](docs/user-guide.md) | Complete command reference, workload details, and analysis workflows |
