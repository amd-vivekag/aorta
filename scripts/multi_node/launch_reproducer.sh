#!/bin/bash
# Multi-node launch script for the minimal RCCL race condition reproducer
#
# Usage:
#   ./scripts/multi_node/launch_reproducer.sh \
#       --docker training-overlap-bugs-rocm70_9-1-shampoo \
#       --hw-queues 4 \
#       --warmup 100 \
#       --verify 10000
#
# This script launches the minimal reproducer on all nodes defined in
# scripts/multi_node/node_ip_list.txt

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Launch the minimal RCCL race condition reproducer on multi-node setup."
    echo ""
    echo "Options:"
    echo "  -d, --docker CONTAINER    Docker container name (required)"
    echo "  -p, --nproc NPROC         Processes per node (default: 8)"
    echo "  -q, --hw-queues N         GPU_MAX_HW_QUEUES value (default: 4)"
    echo "  -w, --warmup N            Warmup iterations (default: 100)"
    echo "  -v, --verify N            Verification iterations (default: 10000)"
    echo "  -s, --same-stream         Use same stream for H2D and datadist"
    echo "  -c, --no-compute          Disable compute simulation"
    echo "  -l, --label LABEL         Experiment label"
    echo "      --master-port PORT    Master port (default: auto-select)"
    echo "  -h, --help                Show this help"
    echo ""
    echo "Examples:"
    echo "  # Test with settings that EXPOSE the bug"
    echo "  $0 --docker training-overlap-bugs-rocm70_9-1-shampoo --hw-queues 4"
    echo ""
    echo "  # Test with settings that MASK the bug (comparison)"
    echo "  $0 --docker training-overlap-bugs-rocm70_9-1-shampoo --hw-queues 2"
    echo ""
    echo "  # Same-stream mode (definitive runtime bug test)"
    echo "  $0 --docker training-overlap-bugs-rocm70_9-1-shampoo --hw-queues 4 --same-stream"
    echo ""
    exit 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AORTA_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MACHINE_IP_FILE="$SCRIPT_DIR/node_ip_list.txt"

# Default values
DOCKER_CONTAINER=""
NPROC_PER_NODE=8
HW_QUEUES=4
WARMUP=100
VERIFY=10000
SAME_STREAM=""
NO_COMPUTE=""
LABEL=""
MASTER_PORT=""

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--docker)
            DOCKER_CONTAINER="$2"
            shift 2
            ;;
        -p|--nproc)
            NPROC_PER_NODE="$2"
            shift 2
            ;;
        -q|--hw-queues)
            HW_QUEUES="$2"
            shift 2
            ;;
        -w|--warmup)
            WARMUP="$2"
            shift 2
            ;;
        -v|--verify)
            VERIFY="$2"
            shift 2
            ;;
        -s|--same-stream)
            SAME_STREAM="--same-stream"
            shift
            ;;
        -c|--no-compute)
            NO_COMPUTE="--no-compute"
            shift
            ;;
        -l|--label)
            LABEL="$2"
            shift 2
            ;;
        --master-port)
            MASTER_PORT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate required arguments
if [[ -z "$DOCKER_CONTAINER" ]]; then
    echo "ERROR: --docker is required"
    usage
fi

# Auto-select master port if not specified
if [[ -z "$MASTER_PORT" ]]; then
    if ! MASTER_PORT=$(python3 - <<'PY'
import socket
s=socket.socket()
s.bind(('',0))
print(s.getsockname()[1])
s.close()
PY
    ); then
        echo "Error: Failed to auto-select master port. Set --master-port manually."
        exit 1
    fi
fi

# Count nodes and calculate world size
NUM_NODES=$(awk 'NF' "$MACHINE_IP_FILE" | wc -l)
WORLD_SIZE=$((NPROC_PER_NODE * NUM_NODES))

# Create experiment directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_DIR="$AORTA_ROOT/experiments/reproducer_hw${HW_QUEUES}_${TIMESTAMP}${LABEL:+_$LABEL}"
mkdir -p "$EXPERIMENT_DIR/logs"

# Save experiment info
cat > "$EXPERIMENT_DIR/experiment_info.txt" << EOF
Experiment: Minimal RCCL Race Condition Reproducer
Timestamp: $TIMESTAMP
Label: ${LABEL:-unlabeled}
Docker: $DOCKER_CONTAINER
GPU_MAX_HW_QUEUES: $HW_QUEUES
Warmup iterations: $WARMUP
Verify iterations: $VERIFY
Same stream mode: ${SAME_STREAM:-no}
Compute simulation: ${NO_COMPUTE:-yes}
Nodes: $NUM_NODES
World size: $WORLD_SIZE
Git: $(git rev-parse --short HEAD 2>/dev/null || echo "N/A")
EOF

echo "=========================================="
echo "Minimal RCCL Race Condition Reproducer"
echo "=========================================="
echo ""
echo "This reproducer uses PROPER SYNCHRONIZATION everywhere."
echo "If corruption occurs, it indicates a RUNTIME BUG in RCCL/HIP."
echo ""
echo "Configuration:"
echo "  Docker container: $DOCKER_CONTAINER"
echo "  GPU_MAX_HW_QUEUES: $HW_QUEUES"
echo "  Warmup iterations: $WARMUP"
echo "  Verify iterations: $VERIFY"
echo "  Same stream mode: ${SAME_STREAM:-no}"
echo "  Compute simulation: ${NO_COMPUTE:-enabled}"
echo "  Nodes: $NUM_NODES"
echo "  World size: $WORLD_SIZE GPUs"
echo "  Master port: $MASTER_PORT"
echo ""
echo "Experiment directory: $EXPERIMENT_DIR"
echo "=========================================="
echo ""

# Build the reproducer command
REPRODUCER_CMD="python -m aorta.race --warmup $WARMUP --verify $VERIFY --hw-queues $HW_QUEUES $SAME_STREAM $NO_COMPUTE"

# Launch on each node
node=0
while IFS= read -r HOST || [[ -n "$HOST" ]]; do
    if [[ -z "$HOST" ]]; then
        continue
    fi

    echo "Launching on Node $node: $HOST"

    LOG_FILE="$EXPERIMENT_DIR/logs/node_${node}.txt"

    if [[ "$node" -eq 0 ]]; then
        MASTER_ADDR="$HOST"
        echo "Master node: $MASTER_ADDR"
    fi

    # Build torchrun command
    TORCHRUN_CMD="torchrun \
        --nnodes $NUM_NODES \
        --node_rank $node \
        --nproc_per_node $NPROC_PER_NODE \
        --master_addr $MASTER_ADDR \
        --master_port $MASTER_PORT \
        -m aorta.race \
        --warmup $WARMUP \
        --verify $VERIFY \
        --hw-queues $HW_QUEUES \
        $SAME_STREAM $NO_COMPUTE"

    # Build docker exec command with environment variables
    DOCKER_CMD="docker exec \
        -e GPU_MAX_HW_QUEUES=$HW_QUEUES \
        -e NCCL_DEBUG=WARN \
        -e NCCL_IB_DISABLE=0 \
        $DOCKER_CONTAINER \
        bash -c 'cd /workspace/aorta && $TORCHRUN_CMD'"

    if [[ "$node" -eq 0 ]]; then
        # Master node - run locally
        eval "$DOCKER_CMD" > "$LOG_FILE" 2>&1 &
    else
        # Worker nodes - run via SSH
        ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
            "$USER@$HOST" "$DOCKER_CMD" > "$LOG_FILE" 2>&1 &
    fi

    ((node++))
done < "$MACHINE_IP_FILE"

echo ""
echo "=========================================="
echo "All nodes launched"
echo "=========================================="
echo ""
echo "Monitor logs:"
echo "  tail -f $EXPERIMENT_DIR/logs/node_0.txt"
echo ""
echo "Monitor all nodes:"
echo "  tail -f $EXPERIMENT_DIR/logs/node_*.txt"
echo ""
echo "Waiting for reproducer to complete..."
echo ""

wait

echo ""
echo "=========================================="
echo "Reproducer completed"
echo "=========================================="
echo ""
echo "Results saved to: $EXPERIMENT_DIR"
echo ""
echo "Check for corruption:"
echo "  grep -i 'CORRUPTION\|RUNTIME BUG' $EXPERIMENT_DIR/logs/*.txt"
echo ""
echo "Summary:"
echo "  grep -i 'VERDICT\|PASSED\|FAILED' $EXPERIMENT_DIR/logs/node_0.txt"
