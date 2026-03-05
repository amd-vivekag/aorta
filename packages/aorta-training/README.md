# aorta-training

FSDP/DDP distributed training benchmarks with compute-communication overlap profiling for ROCm GPUs.

## Quick Start

### Install

```bash
# From the repo root, activate the virtual environment and install all workspace packages
source .venv-test/bin/activate
uv sync
```

### Single-Node Training (FSDP + AdamW)

All commands run from the repo root with the virtual environment active.

```bash
source .venv-test/bin/activate
torchrun --standalone --nproc_per_node=8 train.py --config config/default.yaml
```

### Single-Node Training (DDP + Shampoo)

```bash
torchrun --standalone --nproc_per_node=8 train.py \
    --config config/shampoo_opt.yaml \
    --override distributed.mode=ddp optimizer.name=shampoo
```

### Multi-Node Training

```bash
# 1. Set up node list
scontrol show hostnames $SLURM_NODELIST > scripts/multi_node/node_ip_list.txt

# 2. Start Docker on all nodes
./scripts/multi_node/start_docker_all_nodes.sh \
    docker/docker-compose.rocm70_9-1-shampoo.yaml \
    training-overlap-bugs-rocm70_9-1-shampoo

# 3. Launch training
./scripts/multi_node/master_launch.sh --channels 28 --threads 256 --nproc 8 \
    --config config/multi_node/shampoo_opt_multi_node.yaml \
    --docker training-overlap-bugs-rocm70_9-1-shampoo
```

## Switching Distributed Strategy and Optimizer

All configuration is in YAML. The two knobs that matter:

| What | YAML key | Options |
|------|----------|---------|
| Distributed strategy | `distributed.mode` | `fsdp` (default), `ddp` |
| Optimizer | `optimizer.name` | `adamw` (default), `shampoo` |

You can switch either via config file or `--override` on the CLI (all from repo root with `source .venv-test/bin/activate`):

```bash
# FSDP + AdamW (default)
torchrun --standalone --nproc_per_node=8 train.py --config config/default.yaml

# FSDP + Shampoo
torchrun --standalone --nproc_per_node=8 train.py --config config/default.yaml \
    --override optimizer.name=shampoo

# DDP + AdamW
torchrun --standalone --nproc_per_node=8 train.py --config config/default.yaml \
    --override distributed.mode=ddp

# DDP + Shampoo
torchrun --standalone --nproc_per_node=8 train.py --config config/default.yaml \
    --override distributed.mode=ddp optimizer.name=shampoo
```

## Pre-Built Configs

| Config | Strategy | Optimizer | Use Case |
|--------|----------|-----------|----------|
| `config/default.yaml` | FSDP | AdamW | Quick single-node test |
| `config/shampoo_opt.yaml` | DDP | Shampoo | Shampoo optimizer testing |
| `config/distributed.yaml` | DDP | AdamW | DDP stress testing |
| `config/profile_overlap_2gpu.yaml` | FSDP | AdamW | 2-GPU profiling run |
| `config/multi_node/distributed_multinode.yaml` | FSDP | AdamW | Multi-node baseline |
| `config/multi_node/shampoo_opt_multi_node.yaml` | FSDP | Shampoo | Multi-node Shampoo |

## Project Layout

```
packages/aorta-training/
├── README.md                  # This file (quick start)
├── docs/
│   └── training_user_guide.md # Detailed user guide
├── pyproject.toml
├── scripts/                   # Sweep & experiment scripts
│   ├── optuna_sweep.py
│   ├── run_fsdp_sweep.sh
│   ├── run_full_sweep.sh
│   ├── run_quick_sweep.sh
│   └── run_sdma_prototype.py
└── src/aorta/training/
    ├── __init__.py
    ├── fsdp_trainer.py        # Core trainer (FSDP & DDP)
    ├── data/                  # Synthetic dataset
    ├── models/                # RankingTransformer model
    ├── profiling/             # Multi-stream profiler
    └── experiments/           # SDMA overlap prototype

scripts/multi_node/            # Multi-node launch scripts (repo root)
├── master_launch.sh           # Main entry point
├── local_launch.sh            # Per-node launcher
├── start_docker_all_nodes.sh  # Docker orchestration
├── setup_multi_node.sh        # Interactive cluster setup
├── config_node.sh             # Per-node configuration
├── set_env_variables.sh       # NCCL/RCCL env vars
└── node_ip_list.txt           # Node hostnames
```

## Further Reading

- **[User Guide](docs/training_user_guide.md)** -- Full reference for all config options, multi-node setup, profiling, and troubleshooting.
- **[Multi-Node README](../../scripts/multi_node/README.md)** -- Detailed multi-node orchestration docs.
