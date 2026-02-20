#!/bin/bash
# Run all 3 TF32 precision experiments sequentially with reduced steps,
# then compare results against the 0.1% max error tolerance.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AORTA_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$AORTA_ROOT"

CHANNELS="${CHANNELS:-56}"
THREADS="${THREADS:-256}"
NPROC="${NPROC:-8}"
DOCKER="training-overlap-bugs-rocm70_9-1-shampoo"

CONFIGS=(
    "./config/multi_node/shampoo_opt_multi_node_seed42_native_tf32.yaml"
    "./config/multi_node/shampoo_opt_multi_node_seed42_tf32x1.yaml"
    "./config/multi_node/shampoo_opt_multi_node_seed42_tf32x3.yaml"
)
LABELS=("precision_native_tf32" "precision_tf32x1" "precision_tf32x3")
MODES=("native" "x1" "x3")

echo "============================================"
echo "TF32 Precision Test"
echo "============================================"
echo "Max steps:  100 (set in config YAMLs)"
echo "Channels:   $CHANNELS"
echo "Threads:    $THREADS"
echo "Nproc:      $NPROC"
echo "Docker:     $DOCKER"
echo "Configs:    ${CONFIGS[*]}"
echo "============================================"
echo ""

for i in "${!MODES[@]}"; do
    MODE="${MODES[$i]}"
    LABEL="${LABELS[$i]}"
    CONFIG="${CONFIGS[$i]}"

    echo ""
    echo "============================================"
    echo "  Run $((i+1))/3: TF32 mode = $MODE"
    echo "  Config: $CONFIG"
    echo "============================================"
    echo ""

    ./scripts/multi_node/master_launch.sh \
        --channels "$CHANNELS" --threads "$THREADS" --nproc "$NPROC" \
        --config "$CONFIG" \
        --docker "$DOCKER" \
        --label "$LABEL"

    echo ""
    echo "  Run $((i+1))/3 ($MODE) completed."
    echo ""
done

echo ""
echo "============================================"
echo "All 3 runs completed. Running analysis..."
echo "============================================"
echo ""

python3 scripts/compare_precision_runs.py \
    --baseline-dir "experiments/multinode_*_precision_tf32x3" \
    --compare-dir "experiments/multinode_*_precision_tf32x1" \
                  "experiments/multinode_*_precision_native_tf32" \
    --tolerance 0.1 \
    --skip-warmup 10

echo ""
echo "Done."
