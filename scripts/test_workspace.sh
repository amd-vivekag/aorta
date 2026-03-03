#!/usr/bin/env bash
# Test workspace packages on a GPU node.
#
# Usage:
#   bash scripts/test_workspace.sh
#
# Options:
#   --skip-install    Skip uv sync + torch install (if already done)
#   --multi-node      Run multi-node tests (requires node_ip_list.txt)

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

SKIP_INSTALL=false
MULTI_NODE=false

for arg in "$@"; do
    case $arg in
        --skip-install) SKIP_INSTALL=true ;;
        --multi-node) MULTI_NODE=true ;;
    esac
done

export PATH="$HOME/.local/bin:$PATH"

echo "========================================"
echo "AORTA Workspace Test Suite"
echo "========================================"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Python: $(python3 --version 2>/dev/null || echo 'not found')"
echo ""

# --- Phase 1: Install ---
if [ "$SKIP_INSTALL" = false ]; then
    echo "[1/5] Installing uv..."
    if ! command -v uv &>/dev/null; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
    fi
    echo "  uv $(uv --version)"

    echo "[1/5] Running uv sync..."
    uv sync --all-packages 2>&1 | tail -3

    echo "[1/5] Installing PyTorch nightly (ROCm 7.2)..."
    uv pip install --pre torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/nightly/rocm7.2/ 2>&1 | tail -3
else
    echo "[1/5] Skipping install (--skip-install)"
fi

echo ""

# --- Phase 2: Import smoke tests ---
echo "[2/5] Import smoke tests..."

uv run python -c "from aorta.utils import detect_accelerator; print('  aorta-core: OK')"
uv run python -c "from aorta.report import cli; print('  aorta-report: OK')"
uv run python -c "from aorta.race.config import RaceConfig; print('  aorta-race: OK')"
uv run python -c "from aorta.hw_queue_eval.cli import main; print('  aorta-hw-queue: OK')"
uv run python -c "from aorta.training.data import SyntheticDatasetConfig; print('  aorta-training: OK')"

echo ""

# --- Phase 3: CLI entry points ---
echo "[3/5] CLI entry points..."

uv run aorta-report --version && echo "  aorta-report CLI: OK"
uv run python -m aorta.hw_queue_eval list 2>&1 | head -5 && echo "  hw_queue_eval list: OK"

echo ""

# --- Phase 4: GPU tests (single node) ---
GPU_COUNT=$(uv run python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
echo "[4/5] GPU tests (detected ${GPU_COUNT} GPUs)..."

if [ "${GPU_COUNT}" -gt 0 ]; then
    echo "  Running hw_queue_eval hetero_kernels..."
    uv run python -m aorta.hw_queue_eval run hetero_kernels --streams 4 --iterations 10 --warmup 2 2>&1 | tail -5

    if [ "${GPU_COUNT}" -ge 2 ]; then
        echo "  Running race detector (${GPU_COUNT} GPUs, quick)..."
        uv run torchrun --nproc_per_node="${GPU_COUNT}" -m aorta.race \
            --warmup 5 --verify 20 --no-compute 2>&1 | tail -10
    fi
else
    echo "  SKIPPED: No GPUs detected on $(hostname)"
    echo "  Run this script on a GPU node."
fi

echo ""

# --- Phase 5: pytest ---
echo "[5/5] Running pytest..."
uv run pytest tests/test_imports.py -v 2>&1 | tail -15

echo ""
echo "========================================"
echo "All tests complete."
echo "========================================"
