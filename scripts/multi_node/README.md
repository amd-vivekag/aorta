# Multi-Node Training

Scripts for multi-node distributed training with custom NCCL channel and thread configurations.

## Table of Contents

- [Quick Start](#quick-start)
- [Slurm Setup](#slurm-setup)
- [Usage](#usage)
- [Experiment Tracking](#experiment-tracking)
- [Stopping Training](#stopping-training)
- [Troubleshooting](#troubleshooting)
- [NCCL Configuration](#nccl-configuration)
- [Conductor Setup](#conductor-setup)

## Prerequisites

- 2+ machines with ROCm GPUs, Docker, network connectivity (host mode)
- Passwordless SSH between nodes
- `scripts/multi_node/node_ip_list.txt` with node hostnames - master first
- All nodes on same git branch

## File Structure

```
aorta/
├── scripts/multi_node/
│   ├── master_launch.sh                # Main entrypoint
│   ├── start_docker_all_nodes.sh       # Start Docker on all nodes
│   ├── setup_multi_node.sh             # Automated setup (2+ nodes)
│   ├── config_node.sh                  # Per-node setup
│   ├── local_launch.sh                 # Per-node training (runs in Docker)
│   ├── set_env_variables.sh            # NCCL/RCCL config
│   ├── experiment_list.sh              # List experiments
│   ├── experiment_note.sh              # Add notes to experiments
│   └── node_ip_list.txt                # Node hostnames
├── docker/
│   ├── docker-compose.rocm70_9-1.yaml         # Base Docker config
│   └── docker-compose.rocm70_9-1-shampoo.yaml # Docker with Shampoo (supports both Adam and Shampoo)
├── config/
│   ├── multi_node/
│   │   └── distributed_multinode.yaml  # Default config
│   └── shampoo_opt.yaml                # Shampoo optimizer config
```

## Quick Start

```bash
# First time setup (once per allocation)
scontrol show hostnames $SLURM_NODELIST > scripts/multi_node/node_ip_list.txt
./scripts/multi_node/start_docker_all_nodes.sh \
    docker/docker-compose.rocm70_9-1-shampoo.yaml \
    training-overlap-bugs-rocm70_9-1-shampoo

# Run training (change optimizer via config file)
./scripts/multi_node/master_launch.sh --channels 28 --threads 256 --nproc 8 \
    --config ./config/multi_node/adam_opt_multi_node_seed42.yaml \
    --docker training-overlap-bugs-rocm70_9-1-shampoo \
    --label adam_debug_run13
```

Change optimizer by switching config: `adam_opt_multi_node_seed42.yaml` or `shampoo_opt_multi_node_seed42.yaml`.


---

## Slurm Setup

### Step 1: Pull Base Docker Image

```bash
docker pull rocm/pytorch-private:20251030_rocm_e2e_phantom_mi350_genai_nightly

# If authentication required
docker login
```

### Step 2: Allocate Nodes

```bash
# From head node
salloc -N 3 -p gpu_partition -t 4:00:00
squeue -u $USER
```

### Step 3: Create node_ip_list.txt

```bash
cd /path/to/aorta/scripts/multi_node
scontrol show hostnames $SLURM_NODELIST > node_ip_list.txt
cat node_ip_list.txt
```

### Step 4: SSH to Master and Test Connectivity

```bash
ssh node1-hostname
cd /path/to/aorta

# Test worker connectivity
ssh node2-hostname hostname
ssh node3-hostname hostname
```

### Step 5: Pull Image on All Nodes

```bash
for HOST in $(cat scripts/multi_node/node_ip_list.txt); do
  ssh $HOST "docker pull rocm/pytorch-private:20251030_rocm_e2e_phantom_mi350_genai_nightly"
done
```

### Step 6: Start Docker and Run Training

```bash
./scripts/multi_node/start_docker_all_nodes.sh \
    docker/docker-compose.rocm70_9-1-shampoo.yaml \
    training-overlap-bugs-rocm70_9-1-shampoo

./scripts/multi_node/master_launch.sh --channels 28 --threads 256 --nproc 8 \
    --config ./config/multi_node/adam_opt_multi_node_seed42.yaml \
    --docker training-overlap-bugs-rocm70_9-1-shampoo \
    --label my_experiment
```

---

## Usage

```bash
# Basic launch (defaults: 28 channels, 256 threads, 8 GPUs/node)
./scripts/multi_node/master_launch.sh

# Custom parameters
./scripts/multi_node/master_launch.sh -c 28 -t 256 -p 4 -f config/custom.yaml

# With experiment label
./scripts/multi_node/master_launch.sh --label baseline --amd-wait
```

### Parameters

| Flag | Option | Default | Description |
|------|--------|---------|-------------|
| -c | --channels | 28 | NCCL_MAX_NCHANNELS |
| -t | --threads | 256 | RCCL_THREADS_PER_BLOCK |
| -p | --nproc | 8 | GPUs per node |
| -f | --config | config/multi_node/distributed_multinode.yaml | Config file |
| -d | --docker | training-overlap-bugs-rocm70_9-1 | Docker container name |
| -l | --label | none | Experiment label |
| -w | --amd-wait | false | Enable AMD_OCL_WAIT_COMMAND=1 |
| -r | --rocprof | false | Enable rocprofv3 |
| -m | --stats | false | rocprof stats |
|  | --rocprof-input | none | rocprof yaml |
|  | --master-port | auto | Master port |

Environment variables: `CHANNELS=42 THREADS=512 ./scripts/multi_node/master_launch.sh`

GPU subset: Use `-p 4` or `export CUDA_VISIBLE_DEVICES=0,2,4,6`

### Monitoring

```bash
tail -f experiments/multinode_*/logs/node_*.txt                        # All nodes
tail -f experiments/multinode_*/logs/node_0_*.txt                      # Master only
cat experiments/multinode_*/outputs/rank_00_metrics.jsonl | tail -n 5  # Metrics
```

---

## Experiment Tracking

Each run auto-logs: config, git commit hash, NCCL settings, and full command.

```bash
./scripts/multi_node/master_launch.sh --label baseline [options]
./scripts/multi_node/experiment_list.sh
./scripts/multi_node/experiment_note.sh "your note"
cat experiments/multinode_XXX/experiment_info.txt
```

### Experiment Folder Structure

A complete experiment folder looks like:

```
experiments/multinode_28ch_256th_20251219_144717_adam_no_wait_aux/
├── experiment_info.txt                    # Experiment metadata and notes
├── logs/
│   ├── node_0_20251219_144717.txt         # Node 0 (master) console output
│   ├── node_1_20251219_144717.txt         # Node 1 console output
│   └── node_2_20251219_144717.txt         # Node 2 console output
└── 256thread_28channels/                  # Training outputs
    ├── rank0.log ... rank23.log           # Per-rank training logs
    ├── loss_rank0.log ... loss_rank23.log # Per-rank loss logs
    ├── rank_00_metrics.jsonl ... rank_23_metrics.jsonl  # Per-rank metrics
    └── nan_diagnostics/                   # NaN debugging (if enabled)
        ├── nan_gradients_step22_rank5.json
        └── param_evolution__*_rank*.jsonl
```

| File/Folder | Description |
|-------------|-------------|
| `experiment_info.txt` | Config, git hash, command, notes |
| `logs/node_*.txt` | Raw console output per node |
| `rank*.log` | Training logs (forward/backward, step times) |
| `loss_rank*.log` | Loss values per step |
| `rank_*_metrics.jsonl` | JSON metrics (loss, lr, throughput) |
| `nan_diagnostics/` | NaN gradient dumps and param evolution |

---

## Stopping Training

`Ctrl+C` stops monitoring but training continues in background.

To stop training:
```bash
for HOST in $(cat scripts/multi_node/node_ip_list.txt); do
  ssh $HOST "docker exec training-overlap-bugs-rocm70_9-1 pkill -9 -f 'train.py|torchrun'"
done
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Script hangs | Check last [STAGE] message |
| SSH fails | `ssh-copy-id $USER@<host>` |
| Docker version mismatch | `docker compose version` on each node |
| NCCL timeout | Update `NCCL_SOCKET_IFNAME` in `set_env_variables.sh` |
| World size mismatch | Check `rocm-smi --showid \| wc -l`, adjust `--nproc` |

### Missing Loss/Metrics Files

If experiment folder has no `loss_rank*.log` or `rank_*_metrics.jsonl`:

1. Check Docker is running on each node:
```bash
for HOST in $(cat scripts/multi_node/node_ip_list.txt); do
  ssh $HOST "docker ps | grep training-overlap"
done
```

2. If not running, start manually on each node:
```bash
ssh $HOST "cd ~/aorta && docker compose -f docker/docker-compose.rocm70_9-1.yaml up -d"
```

3. Check RoCE connectivity between nodes:
```bash
# From master, ping worker
ping <worker-ip>

# Check RDMA interfaces
ibstat
```

4. If RoCE fails, use TCP instead:
```bash
./scripts/multi_node/master_launch.sh --tcp [other options]
```

---

## NCCL Configuration

Edit `set_env_variables.sh`:

**InfiniBand:**
```bash
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_0  # Check: ibstat
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=ib0
```

**Ethernet:**
```bash
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
```

**Debug:** `export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ALL`

---

## Conductor Setup

For Conductor environments with SSH key management:

### SSH Key Setup

```bash
ssh-keygen -t rsa -b 4096 -C "conductor-multi-node" -f ~/.ssh/id_rsa_conductor -N ''
cat ~/.ssh/id_rsa_conductor.pub
```

Register public key with your cluster's SSH key management system.

```bash
cat >> ~/.ssh/config << 'EOF'
Host *.dcgpu smci350-* *.zts-gtu.dcgpu
    IdentityFile ~/.ssh/id_rsa_conductor
    StrictHostKeyChecking no
EOF
chmod 600 ~/.ssh/config
```

### Run Setup and Start Docker

```bash
./scripts/multi_node/setup_multi_node.sh
./scripts/multi_node/start_docker_all_nodes.sh
```

Creates `node_ip_list.txt` with hostnames, detects network interfaces, verifies SSH and git branches.
