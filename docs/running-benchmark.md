# Running the Benchmark

This guide covers different ways to launch the AORTA benchmark on CUDA and ROCm systems.

## Quick Start

### ROCm

```bash
bash scripts/launch_rocm.sh config/default.yaml
```

### CUDA

```bash
bash scripts/launch_cuda.sh config/default.yaml
```

Both scripts:
- Default to `config/default.yaml` but accept an override as the first argument
- Query `torch.cuda.device_count()` to size `--nproc_per_node`
- Fall back gracefully when detection fails
- Export `PYTHONPATH=$REPO_ROOT/src` so the `aorta` package is discoverable

## Direct Invocation

For more control over the launch:

```bash
torchrun --nproc_per_node 4 train.py --config config/default.yaml --override training.max_steps=100
```

Use dotted `--override` arguments to mutate configuration values without editing the YAML file.

## Torch Compile Acceleration

Enable AOT compilation by toggling the `compile` block or CLI overrides:

```bash
torchrun --nproc_per_node 4 train.py \
  --config config/default.yaml \
  --override compile.enabled=true compile.backend=inductor compile.mode=max-autotune
```

### Compile Behavior

- The toolkit compiles the FSDP-wrapped model and falls back gracefully if `torch.compile` raises (logging the reason).
- On ROCm, `torch.compile` with `backend=inductor` is still experimental; the launcher automatically downgrades to the safer `aot_eager` backend when necessary.
- You can override this by explicitly passing another backend (e.g., `compile.backend=aot_eager`).
- Tune `compile.fullgraph`, `compile.dynamic`, or `compile.options` (passed directly to `torch.compile`) to match your workload characteristics.
- Compilation occurs per rank, so expect extra time on the first iteration; subsequent steps reuse the optimized graph.

## SDMA Prototype Benchmark

![SDMA Benchmark](../analysis/figures/sdma_benchmark.png)

To measure theoretical compute/SDMA overlap on ROCm without modifying the full training loop:

```bash
python scripts/run_sdma_prototype.py --device 0 --matrix-size 4096 --copy-mb 64
```

The script:
- Launches GEMM-heavy kernels on one stream while issuing `hipMemcpyAsync` transfers on a high-priority stream
- Reports the average duration with and without overlap plus the estimated savings

Use `rocprofv3` (or `scripts/rocprof_capture.sh`) against this benchmark to inspect SDMA engine utilization and validate whether transfers run concurrently with compute.

## Multi-Node Training

For distributed training across multiple nodes:

```bash
# Basic multi-node launch
./scripts/multi_node/master_launch.sh --channels 28 --threads 256

# With experiment label for easy identification
./scripts/multi_node/master_launch.sh --label shampoo_test --config config/multi_node/shampoo_opt_multi_node.yaml
```

The `--label` option appends a custom suffix to the experiment directory name (e.g., `experiments/multinode_28ch_256th_20260116_123456_shampoo_test/`).

### Prerequisites

1. Configure node IPs in `scripts/multi_node/node_ip_list.txt` (one IP/hostname per line)
2. Ensure passwordless SSH access between nodes
3. Start the Docker container on all nodes

### Available Options

| Option | Description |
| --- | --- |
| `-c, --channels` | NCCL_MAX_NCHANNELS (default: 28) |
| `-t, --threads` | RCCL_THREADS_PER_BLOCK (default: 256) |
| `-f, --config` | Config file path |
| `-p, --nproc` | Processes per node (default: 8) |
| `-d, --docker` | Docker container name |
| `-l, --label` | Experiment label (appended to directory name) |
| `-r, --rocprof` | Enable rocprofv3 tracing |

## Next Steps

- [Configuration Guide](configuration.md) - Tune model and training parameters
- [Profiling Guide](profiling.md) - Capture and analyze traces
