#!/bin/bash
# Verification script to check if batch training setup is ready

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AORTA_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "========================================="
echo "Batch Training Setup Verification"
echo "========================================="
echo ""

# Check required files
echo "Checking required files..."
echo ""

check_file() {
    local file="$1"
    local description="$2"
    if [[ -f "$file" ]]; then
        echo "✓ $description"
        echo "  Path: $file"
        return 0
    else
        echo "✗ $description"
        echo "  Missing: $file"
        return 1
    fi
}

check_executable() {
    local file="$1"
    local description="$2"
    if [[ -x "$file" ]]; then
        echo "✓ $description (executable)"
        return 0
    else
        echo "✗ $description (not executable)"
        return 1
    fi
}

errors=0

# Check scripts
check_executable "$SCRIPT_DIR/batch_train.sh" "Batch training script" || ((errors++))
check_file "$SCRIPT_DIR/master_launch.sh" "Master launch script" || ((errors++))
check_file "$SCRIPT_DIR/node_ip_list.txt" "Node IP list" || ((errors++))

echo ""

# Check config files
echo "Checking configuration files..."
echo ""
check_file "$AORTA_ROOT/config/multi_node/shampoo_opt_multi_node_seed42.yaml" "Shampoo config" || ((errors++))
check_file "$AORTA_ROOT/config/multi_node/adam_opt_multi_node_seed42.yaml" "AdamW config" || ((errors++))

echo ""

# Check node list
echo "Checking node configuration..."
echo ""
NUM_NODES=$(grep -c . "$SCRIPT_DIR/node_ip_list.txt" || echo "0")
echo "✓ Number of nodes: $NUM_NODES"
echo ""
echo "Node list:"
cat "$SCRIPT_DIR/node_ip_list.txt"

echo ""
echo "Checking Docker container..."
DOCKER_CONTAINER="training-overlap-bugs-rocm70_9-1-shampoo-vivekag"
if docker ps --format '{{.Names}}' | grep -q "^${DOCKER_CONTAINER}$"; then
    echo "✓ Docker container running: $DOCKER_CONTAINER"
else
    echo "⚠ Docker container not running: $DOCKER_CONTAINER"
    echo "  (This is OK if you plan to start it before running batch training)"
fi

echo ""
echo "========================================="
if [[ $errors -eq 0 ]]; then
    echo "✓ Setup verification PASSED"
    echo ""
    echo "You can now run:"
    echo "  cd $AORTA_ROOT"
    echo "  ./scripts/multi_node/batch_train.sh"
else
    echo "✗ Setup verification FAILED ($errors errors)"
    echo ""
    echo "Please fix the errors above before running batch training"
fi
echo "========================================="

exit $errors
