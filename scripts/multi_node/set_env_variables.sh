#!/bin/bash
# Global NCCL/RCCL environment variables for multi-node training
# Configured for MI350X cluster

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NODE_LIST_FILE="${SCRIPT_DIR}/node_ip_list.txt"

# Check if USE_TCP is set (for non-interactive use)
if [[ -n "$USE_TCP" ]]; then
    use_tcp="$USE_TCP"
else
    read -p "Use TCP transport? [y/N]: " use_tcp
fi

if [[ "$use_tcp" =~ ^[Yy]$|^true$ ]]; then
    echo "[OK] Using TCP transport"
    export NCCL_IB_DISABLE=1
    export NCCL_NET=Socket
else
    echo "[OK] Using RDMA transport"
    export NCCL_IB_DISABLE=0
fi
export GPU_MAX_HW_QUEUES=2 
# NCCL Debug Settings (enabled to track NaN/Inf failures)
export NCCL_DEBUG=WARN
#export NCCL_DEBUG_SUBSYS=ALL

# IB/RNIC Configuration for MI350X (used when RoCE is available)
export NCCL_IB_HCA=bnxt_re0,bnxt_re1,bnxt_re2,bnxt_re3,bnxt_re4,bnxt_re5,bnxt_re6,bnxt_re7
export NCCL_IB_GID_INDEX=3
export NCCL_NCHANNELS_PER_NET_PEER=8

# HSA Settings for ROCm
export HSA_ENABLE_IPC_MODE_LEGACY=1

# NCCL Protocol
export NCCL_PROTO=Simple

# Channel Configuration (can be overridden by sweep parameters)
export NCCL_MIN_NCHANNELS=40
export NCCL_MAX_NCHANNELS=40

# Network Interface for MI350X cluster
export NCCL_SOCKET_IFNAME=enp49s0f0np0,fenic0

# Timeout settings
export TORCH_DIST_INIT_TIMEOUT=60
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# PyTorch ROCm Profiler
export PYTORCH_ROCM_PROFILER_ENABLE_TRACING=1

# Optional: Force non-overlap for debugging
# export GPU_MAX_HW_QUEUES=1
# unset TORCH_NCCL_HIGH_PRIORITY

# Optional: Disable SDMA for testing
# export HSA_ENABLE_SDMA=0

# Optional: Disable IB for Ethernet-only testing
# export NCCL_IB_DISABLE=1
