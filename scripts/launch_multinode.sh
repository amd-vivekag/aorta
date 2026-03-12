#!/bin/bash
# Multi-node launcher for Shampoo NaN reproducers via SLURM.
#
# Usage:
#   sbatch --nodes=2 --gres=gpu:8 --ntasks-per-node=8 --partition=mi355x \
#     --time=01:00:00 scripts/launch_multinode.sh <script.py> [args...]
#
# Or inline with srun:
#   srun --partition=mi355x --nodes=2 --gres=gpu:8 --ntasks-per-node=1 \
#     --time=01:00:00 scripts/launch_multinode.sh <script.py> [args...]

set -euo pipefail

SCRIPT="$1"
shift

source /mnt/vast/huzhao/projects/aorta/.venv/bin/activate

export MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)
export MASTER_PORT=${MASTER_PORT:-29500}
export NNODES=$SLURM_NNODES
export NODE_RANK=$SLURM_NODEID
export GPUS_PER_NODE=8

echo "=== Multi-node launch ==="
echo "  Node: $(hostname) (rank $NODE_RANK / $NNODES)"
echo "  Master: $MASTER_ADDR:$MASTER_PORT"
echo "  GPUs/node: $GPUS_PER_NODE"
echo "  Script: $SCRIPT"
echo "  Args: $*"
echo "========================="

torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$GPUS_PER_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    "$SCRIPT" "$@"
