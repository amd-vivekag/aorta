# aorta-training User Guide

Complete reference for running distributed training benchmarks with FSDP/DDP, AdamW/Shampoo optimizers, single-node and multi-node setups, profiling, and hyperparameter sweeps.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Installation](#installation)
- [Running Training](#running-training)
  - [Single-Node](#single-node)
  - [Multi-Node](#multi-node)
- [Configuration Reference](#configuration-reference)
  - [Distributed Strategy (FSDP vs DDP)](#distributed-strategy-fsdp-vs-ddp)
  - [Optimizer (AdamW vs Shampoo)](#optimizer-adamw-vs-shampoo)
  - [Mixed Precision](#mixed-precision)
  - [FSDP Sharding Strategies](#fsdp-sharding-strategies)
  - [torch.compile](#torchcompile)
  - [Profiling](#profiling)
- [Config Files](#config-files)
- [Multi-Node Setup](#multi-node-setup)
  - [Prerequisites](#prerequisites)
  - [Slurm Workflow](#slurm-workflow)
  - [Docker Images](#docker-images)
  - [NCCL/RCCL Tuning](#ncclrccl-tuning)
  - [Monitoring and Debugging](#monitoring-and-debugging)
- [Hyperparameter Sweeps](#hyperparameter-sweeps)
- [Troubleshooting](#troubleshooting)

---

## Architecture Overview

The trainer runs a **RankingTransformer** model on synthetic data to benchmark compute-communication overlap under realistic distributed training conditions. It is not a production training loop -- it is an instrumented benchmark designed to expose overlap behavior, RCCL race conditions, and performance bottlenecks on ROCm GPUs.

```
train.py (CLI entry point)
  └── aorta.training.fsdp_trainer.main()
        ├── init_distributed()          # torch.distributed setup
        ├── build_fsdp_model()          # or build_ddp_model()
        ├── configure_optimizer()       # AdamW or DistributedShampoo
        ├── configure_scheduler()       # Linear warmup + decay
        └── training_loop()
              ├── StreamProfiler         # Per-iteration overlap timing
              ├── torch.profiler          # Chrome traces / TensorBoard
              └── MetricsLogger           # JSONL per-rank metrics
```

**Key components:**

| Module | Purpose |
|--------|---------|
| `fsdp_trainer.py` | Core trainer supporting both FSDP and DDP |
| `data/synthetic_dataset.py` | Deterministic synthetic ranking dataset |
| `models/ranking_transformer.py` | Transformer encoder for ranking signals |
| `profiling/stream_profiler.py` | Multi-stream CUDA/HIP event timing and overlap calculation |
| `experiments/sdma_prototype.py` | SDMA copy-vs-compute overlap microbenchmark |

---

## Installation

### Developer install (full workspace)

```bash
cd /path/to/aorta
source .venv-test/bin/activate
uv sync
```

This installs all workspace packages (`aorta-core`, `aorta-training`, etc.) in editable mode.

### Shampoo optimizer (optional)

The Shampoo optimizer requires the `distributed_shampoo` package:

```bash
pip install distributed_shampoo
```

---

## Running Training

All training launches use `torchrun` with the root-level `train.py` entry point. Commands run from the repo root with the virtual environment active (`source .venv-test/bin/activate`).

### Single-Node

**Basic FSDP + AdamW (8 GPUs):**

```bash
source .venv-test/bin/activate
torchrun --standalone --nproc_per_node=8 train.py --config config/default.yaml
```

**2-GPU quick profiling run:**

```bash
torchrun --standalone --nproc_per_node=2 train.py --config config/profile_overlap_2gpu.yaml
```

**DDP mode:**

```bash
torchrun --standalone --nproc_per_node=8 train.py \
    --config config/default.yaml \
    --override distributed.mode=ddp
```

**Shampoo optimizer:**

```bash
torchrun --standalone --nproc_per_node=8 train.py \
    --config config/default.yaml \
    --override optimizer.name=shampoo
```

**Subset of GPUs:**

```bash
export ROCR_VISIBLE_DEVICES=0,1,2,3
torchrun --standalone --nproc_per_node=4 train.py --config config/default.yaml
```

**Custom overrides (no config edit needed):**

```bash
torchrun --standalone --nproc_per_node=8 train.py \
    --config config/default.yaml \
    --override training.batch_size=32 training.max_steps=50 \
              model.num_layers=12 fsdp.sharding_strategy=hybrid_shard
```

### Multi-Node

Multi-node training is orchestrated through bash scripts in `scripts/multi_node/`. These scripts handle Docker container management, SSH orchestration, NCCL/RCCL environment configuration, and `torchrun` launch across nodes.

**Quick start (after initial setup):**

```bash
./scripts/multi_node/master_launch.sh --channels 28 --threads 256 --nproc 8 \
    --config config/multi_node/distributed_multinode.yaml \
    --docker training-overlap-bugs-rocm70_9-1-shampoo
```

See [Multi-Node Setup](#multi-node-setup) for full instructions.

---

## Configuration Reference

All configuration is driven by YAML files in `config/`. Any value can be overridden on the CLI with `--override key.subkey=value`.

### Distributed Strategy (FSDP vs DDP)

Controlled by `distributed.mode` in the config:

```yaml
distributed:
  mode: fsdp     # "fsdp" or "ddp"
```

| Mode | When to use |
|------|-------------|
| `fsdp` | Default. Shards parameters across GPUs. Better memory efficiency for large models. Supports `full_shard`, `hybrid_shard`, `shard_grad_op`. |
| `ddp` | Replicates full model on each GPU. Simpler, sometimes faster for models that fit in memory. |

**FSDP-specific settings:**

```yaml
fsdp:
  sharding_strategy: full_shard    # full_shard, hybrid_shard, shard_grad_op, no_shard
  backward_prefetch: BACKWARD_PRE  # BACKWARD_PRE or BACKWARD_POST
  use_orig_params: true
  limit_all_gathers: true          # Rate-limit all-gathers (reduces memory spikes)
  forward_prefetch: true           # Prefetch next layer's params during forward
  sync_module_states: true
  param_init_device: cpu           # "cpu" or "meta" (meta = deferred init, less host memory)
```

**DDP-specific settings:**

```yaml
distributed:
  mode: ddp
  gradient_as_bucket_view: true
  static_graph: true               # Enables optimizations for fixed computation graphs
  bucket_cap_mb: 25                # Gradient bucket size for all-reduce
  find_unused_parameters: false
```

### Optimizer (AdamW vs Shampoo)

Controlled by `optimizer.name`:

```yaml
# AdamW (default)
optimizer:
  name: adamw
  lr: 0.0003
  weight_decay: 0.01
  betas: [0.9, 0.98]
  eps: 1.0e-8

# Shampoo (requires distributed_shampoo package)
optimizer:
  name: shampoo
  lr: 0.0002
  weight_decay: 0.01
  betas: [0.9, 0.985]
  eps: 1.0e-8
  precondition_frequency: 50
  max_preconditioner_dim: 8192
  start_preconditioning_step: 50
```

Shampoo uses `DDPDistributedConfig` internally for multi-GPU preconditioner communication. The Shampoo Docker image (`docker-compose.rocm70_9-1-shampoo.yaml`) comes with the package pre-installed.

**Shampoo-specific parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `precondition_frequency` | 50 | How often (in steps) Shampoo recomputes the preconditioner matrices. Lower values give more frequent updates but increase compute cost. |
| `max_preconditioner_dim` | 8192 | Maximum dimension of preconditioner matrices. Parameters with dimensions larger than this are block-diagonalized into chunks of this size. |
| `start_preconditioning_step` | 50 | Step at which preconditioning begins. Before this step, Shampoo behaves like a diagonal (Adam-like) optimizer. |

These can be overridden from the CLI:

```bash
torchrun --standalone --nproc_per_node=8 train.py \
    --config config/shampoo_opt.yaml \
    --override optimizer.precondition_frequency=100 \
              optimizer.max_preconditioner_dim=4096 \
              optimizer.start_preconditioning_step=200
```

### Mixed Precision

```yaml
training:
  mixed_precision: bf16    # "bf16", "fp16", or "none"
```

| Mode | Notes |
|------|-------|
| `bf16` | Default. Recommended for ROCm MI300/MI350. No loss scaling needed. |
| `fp16` | Uses `GradScaler`. More communication traffic than bf16. |
| `none` | Full FP32. Largest memory footprint. |

### FSDP Sharding Strategies

```yaml
fsdp:
  sharding_strategy: full_shard  # See table below
```

| Strategy | Behavior | Best for |
|----------|----------|----------|
| `full_shard` | Shards params + gradients + optimizer state across all GPUs | Single-node, memory-constrained |
| `hybrid_shard` | Shards within each node, replicates across nodes | Multi-node (reduces inter-node traffic) |
| `shard_grad_op` | Shards gradients + optimizer state only | Faster than full_shard if params fit |
| `no_shard` | No sharding (equivalent to DDP) | Debugging |

For `hybrid_shard`, the number of GPUs per shard group is auto-detected from `LOCAL_WORLD_SIZE` (set by `torchrun --nproc_per_node`). Override manually with:

```yaml
fsdp:
  sharding_strategy: hybrid_shard
  hybrid_shard_gpus_per_node: 8
```

### torch.compile

```yaml
compile:
  enabled: false          # Set true to enable
  backend: inductor       # "inductor" or other torch.compile backends
  mode: max-autotune      # "default", "reduce-overhead", "max-autotune"
  fullgraph: false
  dynamic: false
```

### Profiling

Two profiling systems are available:

**1. StreamProfiler** (always active) -- records per-iteration overlap metrics as JSONL:

```
experiments/rank_00_metrics.jsonl
experiments/rank_01_metrics.jsonl
...
```

**2. torch.profiler** (optional) -- generates Chrome traces and/or TensorBoard logs:

```yaml
profiling:
  enabled: true
  wait: 1            # Steps before warmup
  warmup: 2          # Warmup steps (not recorded)
  active: 4          # Steps to profile
  repeat: 1          # How many wait/warmup/active cycles
  record_shapes: true
  profile_memory: true
  with_stack: false
  with_flops: false
  tensorboard: true  # Write TensorBoard events
  chrome_trace: true # Write chrome://tracing JSON
  trace_filename: trace.json
```

Traces are saved to `<output_dir>/torch_profiler/rank<N>/`.

**rocprofv3** can be enabled for multi-node via `--rocprof` flag on `master_launch.sh`.

---

## Config Files

### Single-Node

| File | Strategy | Optimizer | Steps | Description |
|------|----------|-----------|-------|-------------|
| `config/default.yaml` | FSDP (`full_shard`) | AdamW | 200 | Baseline single-node benchmark |
| `config/shampoo_opt.yaml` | DDP | Shampoo | 2200 | Shampoo optimizer testing |
| `config/distributed.yaml` | DDP | AdamW | 500 | DDP stress test with allreduce injection |
| `config/profile_overlap_2gpu.yaml` | FSDP (`full_shard`) | AdamW | 12 | Quick 2-GPU profiling |
| `config/tf32_overlap_eval.yaml` | -- | -- | -- | TF32 overlap evaluation |
| `config/mi350_overlap_stress.yaml` | -- | -- | -- | MI350 overlap stress test |
| `config/reproduce_hang.yaml` | -- | -- | -- | Reproduce multi-stream hangs |

### Multi-Node

| File | Strategy | Optimizer | Description |
|------|----------|-----------|-------------|
| `config/multi_node/distributed_multinode.yaml` | FSDP (`hybrid_shard`) | AdamW | Multi-node baseline (48 layers) |
| `config/multi_node/shampoo_opt_multi_node.yaml` | FSDP (`hybrid_shard`) | Shampoo | Multi-node Shampoo (NaN investigation) |

---

## Multi-Node Setup

### Prerequisites

- 2+ nodes with ROCm GPUs and Docker
- Passwordless SSH between all nodes
- Shared or synced filesystem with the aorta repo
- Slurm (recommended) or manual node allocation

### Slurm Workflow

**1. Allocate nodes:**

```bash
salloc -N 3 -p gpu_partition -t 4:00:00
```

**2. Create node list (master node first):**

```bash
scontrol show hostnames $SLURM_NODELIST > scripts/multi_node/node_ip_list.txt
```

**3. Pull Docker image on all nodes:**

```bash
for HOST in $(cat scripts/multi_node/node_ip_list.txt); do
  ssh $HOST "docker pull rocm/pytorch-private:20251030_rocm_e2e_phantom_mi350_genai_nightly"
done
```

**4. Start Docker containers:**

```bash
./scripts/multi_node/start_docker_all_nodes.sh \
    docker/docker-compose.rocm70_9-1-shampoo.yaml \
    training-overlap-bugs-rocm70_9-1-shampoo
```

**5. Launch training:**

```bash
./scripts/multi_node/master_launch.sh --channels 28 --threads 256 --nproc 8 \
    --config config/multi_node/shampoo_opt_multi_node.yaml \
    --docker training-overlap-bugs-rocm70_9-1-shampoo \
    --label my_experiment
```

**Or use the interactive setup script (first time):**

```bash
./scripts/multi_node/setup_multi_node.sh
```

This walks you through node discovery, SSH verification, GPU detection, and network interface configuration.

### Docker Images

| Compose file | Container name | Includes |
|--------------|----------------|----------|
| `docker-compose.rocm70_9-1.yaml` | `training-overlap-bugs-rocm70_9-1` | Base ROCm + PyTorch |
| `docker-compose.rocm70_9-1-shampoo.yaml` | `training-overlap-bugs-rocm70_9-1-shampoo` | Base + `distributed_shampoo` |
| `docker-compose.build.yaml` | `training-overlap-bugs-default` | Development build |

Use the Shampoo image if you need both AdamW and Shampoo (it supports both).

### master_launch.sh Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `-c, --channels` | 28 | `NCCL_MAX_NCHANNELS` |
| `-t, --threads` | 256 | `RCCL_THREADS_PER_BLOCK` |
| `-p, --nproc` | 8 | GPUs per node |
| `-f, --config` | `config/multi_node/distributed_multinode.yaml` | Config file |
| `-d, --docker` | `training-overlap-bugs-rocm70_9-1` | Docker container name |
| `-l, --label` | (none) | Experiment label suffix |
| `-r, --rocprof` | false | Enable rocprofv3 tracing |
| `-m, --stats` | false | rocprof CU utilization stats |
| `--master-port` | auto | Override master port |

### NCCL/RCCL Tuning

All NCCL/RCCL environment variables are centralized in `scripts/multi_node/set_env_variables.sh`. Key settings:

```bash
# IB/RNIC for MI350X
export NCCL_IB_HCA=bnxt_re0,bnxt_re1,...
export NCCL_SOCKET_IFNAME=enp49s0f0np0,fenic0

# Protocol
export NCCL_PROTO=Simple
export NCCL_MAX_NCHANNELS=56

# Timeouts
export NCCL_TIMEOUT=150
export TORCH_DIST_INIT_TIMEOUT=150

# ROCm-specific
export HSA_ENABLE_SDMA=0
export RCCL_DIRECT_ALLGATHER_DISABLE=1
```

Edit this file to match your cluster's network topology. The `--channels` and `--threads` CLI flags override `NCCL_MAX_NCHANNELS` and `RCCL_THREADS_PER_BLOCK` respectively.

### Monitoring and Debugging

```bash
# Watch training progress (master node)
tail -f experiments/multinode_*/logs/node_0_*.txt

# Watch all nodes
tail -f experiments/multinode_*/logs/node_*.txt

# Check per-rank metrics
cat experiments/multinode_*/outputs/rank_00_metrics.jsonl | tail -5

# Stop training on all nodes
for HOST in $(cat scripts/multi_node/node_ip_list.txt); do
  ssh $HOST "docker exec <container> pkill -9 -f 'train.py|torchrun'"
done
```

Each experiment run is saved to `experiments/multinode_<channels>ch_<threads>th_<timestamp>[_label]/` with:
- `logs/` -- per-node stdout/stderr
- `config_used.yaml` -- snapshot of config
- `experiment_info.txt` -- metadata (git hash, parameters)
- Per-config output directories with rank metrics and traces

---

## Hyperparameter Sweeps

The `packages/aorta-training/scripts/` directory contains Optuna-based hyperparameter sweeps for overlap optimization.

**Quick test (10 trials, 20 steps):**

```bash
./packages/aorta-training/scripts/run_quick_sweep.sh
```

**FSDP parameter sweep (50 trials):**

```bash
./packages/aorta-training/scripts/run_fsdp_sweep.sh
```

**Full sweep (100 trials, all parameters):**

```bash
NUM_TRIALS=100 NUM_GPUS=4 ./packages/aorta-training/scripts/run_full_sweep.sh
```

Results are saved to `optuna_sweeps/` with:
- `optimization_results.json` -- best trial and parameters
- `best_config.yaml` -- ready-to-use config with optimal settings

Search spaces: `full` (all knobs), `fsdp_only`, `env_only` (RCCL env vars), `workload_only` (batch size, precision).

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: aorta.training` | Run `uv sync` or `pip install -e packages/aorta-training` from repo root |
| `ModuleNotFoundError: distributed_shampoo` | Use the Shampoo Docker image, or `pip install distributed_shampoo` |
| Training hangs at init | Check `NCCL_SOCKET_IFNAME` matches your network interface (`ip addr show`) |
| NCCL timeout | Increase `NCCL_TIMEOUT` in `set_env_variables.sh`, verify SSH connectivity |
| NaN loss with Shampoo | Known issue on multi-node ROCm. Try reducing `lr`, increasing `warmup_steps`, raising `optimizer.start_preconditioning_step`, or lowering `optimizer.precondition_frequency` |
| OOM on large configs | Reduce `training.batch_size`, use `fsdp.param_init_device: meta`, or use `full_shard` |
| `torch.compile` failures on ROCm | Set `compile.enabled: false` (default). Inductor on ROCm is experimental |
| Docker container not found | Run `start_docker_all_nodes.sh` first. Check with `docker ps` on each node |
| Branch mismatch across nodes | `master_launch.sh` auto-checks this. Fix with `ssh <node> 'cd /path/to/aorta && git checkout <branch>'` |
| GPU busy / shared cluster | Set `ROCR_VISIBLE_DEVICES` and reduce `--nproc` to use fewer GPUs |
