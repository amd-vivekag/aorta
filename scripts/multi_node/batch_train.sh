#!/bin/bash
# Batch training script for multi-node optimizer comparison
# Runs 12 combinations: 2 optimizers × 2 hw-queues × 3 streams

set -e -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AORTA_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
STATE_FILE="$AORTA_ROOT/.batch_train_state"
BATCH_LOG="$AORTA_ROOT/batch_train_$(date +%Y%m%d_%H%M%S).log"

# Configuration
CHANNELS=28
THREADS=256
NPROC=8
DOCKER_CONTAINER="training-overlap-bugs-rocm70_9-1-shampoo-vivekag"

# Define all combinations (grouped by optimizer)
declare -a COMBINATIONS=(
    # Shampoo combinations
    "shampoo:2:2:./config/multi_node/shampoo_opt_multi_node_seed42.yaml:shampoo_hwq2_str2:--tcp"
    "shampoo:2:4:./config/multi_node/shampoo_opt_multi_node_seed42.yaml:shampoo_hwq2_str4:--tcp"
    "shampoo:2:6:./config/multi_node/shampoo_opt_multi_node_seed42.yaml:shampoo_hwq2_str6:--tcp"
    "shampoo:4:2:./config/multi_node/shampoo_opt_multi_node_seed42.yaml:shampoo_hwq4_str2:--tcp"
    "shampoo:4:4:./config/multi_node/shampoo_opt_multi_node_seed42.yaml:shampoo_hwq4_str4:--tcp"
    "shampoo:4:6:./config/multi_node/shampoo_opt_multi_node_seed42.yaml:shampoo_hwq4_str6:--tcp"
    # Adam combinations
    "adam:2:2:./config/multi_node/adam_opt_multi_node_seed42.yaml:adam_hwq2_str2:--tcp"
    "adam:2:4:./config/multi_node/adam_opt_multi_node_seed42.yaml:adam_hwq2_str4:--tcp"
    "adam:2:6:./config/multi_node/adam_opt_multi_node_seed42.yaml:adam_hwq2_str6:--tcp"
    "adam:4:2:./config/multi_node/adam_opt_multi_node_seed42.yaml:adam_hwq4_str2:--tcp"
    "adam:4:4:./config/multi_node/adam_opt_multi_node_seed42.yaml:adam_hwq4_str4:--tcp"
    "adam:4:6:./config/multi_node/adam_opt_multi_node_seed42.yaml:adam_hwq4_str6:--tcp"
)

# Get number of nodes for verification
MACHINE_IP_FILE="$SCRIPT_DIR/node_ip_list.txt"
if [[ ! -f "$MACHINE_IP_FILE" ]]; then
    echo "Error: $MACHINE_IP_FILE not found"
    exit 1
fi
NUM_NODES=$(grep -c . "$MACHINE_IP_FILE" || echo "0")

# Functions
log_message() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$msg"
    echo "$msg" >> "$BATCH_LOG"
}

mark_completed() {
    local combo_id="$1"
    echo "$combo_id" >> "$STATE_FILE"
    log_message "Marked completed: $combo_id"
}

is_completed() {
    local combo_id="$1"
    if [[ -f "$STATE_FILE" ]]; then
        grep -qx "$combo_id" "$STATE_FILE" && return 0
    fi
    return 1
}

wait_for_training_completion() {
    local exp_dir="$1"
    local label="$2"
    local check_interval=30  # Check every 30 seconds

    log_message "Monitoring training for: $label"
    log_message "Experiment directory: $exp_dir"

    while true; do
        sleep $check_interval

        # Check if experiment directory exists
        if [[ ! -d "$exp_dir/logs" ]]; then
            log_message "Warning: Experiment logs directory not found yet: $exp_dir/logs"
            continue
        fi

        # Find all node log files
        local log_files=()
        while IFS= read -r -d '' file; do
            log_files+=("$file")
        done < <(find "$exp_dir/logs" -name "node_*.txt" -print0 2>/dev/null)

        if [[ ${#log_files[@]} -eq 0 ]]; then
            log_message "Warning: No log files found yet in $exp_dir/logs"
            continue
        fi

        # Check if we have logs for all expected nodes
        local expected_logs=$NUM_NODES
        if [[ ${#log_files[@]} -lt $expected_logs ]]; then
            log_message "Found ${#log_files[@]}/$expected_logs log files, waiting for all nodes..."
            continue
        fi

        # Check each node log for completion message
        local completed_nodes=0
        local failed_nodes=0

        for log_file in "${log_files[@]}"; do
            if tail -20 "$log_file" 2>/dev/null | grep -q "Node [0-9]* training completed"; then
                completed_nodes=$((completed_nodes + 1))
            fi
        done

        log_message "Progress: $completed_nodes/$NUM_NODES nodes completed"

        # All nodes completed
        if [[ $completed_nodes -eq $NUM_NODES ]]; then
            log_message "SUCCESS: All $NUM_NODES nodes completed training"

            # Check for failures
            for log_file in "${log_files[@]}"; do
                if tail -50 "$log_file" 2>/dev/null | grep -qi "failed\|error"; then
                    log_message "Warning: Node $(basename "$log_file") may have encountered errors (check logs)"
                    failed_nodes=$((failed_nodes + 1))
                fi
            done

            if [[ $failed_nodes -gt 0 ]]; then
                log_message "Warning: $failed_nodes node(s) may have failed (continuing anyway)"
            fi

            return 0
        fi
    done
}

run_training() {
    local optimizer="$1"
    local hw_queues="$2"
    local streams="$3"
    local config="$4"
    local label="$5"
    local extra_flags="$6"

    log_message "========================================="
    log_message "Starting training: $label"
    log_message "  Optimizer: $optimizer"
    log_message "  HW Queues: $hw_queues"
    log_message "  Streams: $streams"
    log_message "  Config: $config"
    log_message "========================================="

    # Build command
    local cmd="$SCRIPT_DIR/master_launch.sh \
        --channels $CHANNELS \
        --threads $THREADS \
        --nproc $NPROC \
        --config $config \
        --docker $DOCKER_CONTAINER \
        --hw-queues $hw_queues \
        --streams $streams \
        --label $label \
        $extra_flags"

    log_message "Command: $cmd"

    # Execute training (run in background and capture experiment directory)
    cd "$AORTA_ROOT"

    # Start training - use bash -c to ensure we get a single trackable PID
    bash -c "$cmd" &
    local training_pid=$!

    log_message "Training launched with PID: $training_pid"

    # Verify the process is running
    if ! ps -p $training_pid > /dev/null 2>&1; then
        log_message "ERROR: Training process $training_pid not found immediately after launch"
        return 1
    fi

    # Wait a bit for experiment directory to be created
    sleep 5

    # Find the newest experiment directory matching the pattern
    local exp_dir=""
    local max_attempts=10
    local attempt=0

    while [[ $attempt -lt $max_attempts ]]; do
        exp_dir=$(find "$AORTA_ROOT/experiments" -maxdepth 1 -type d -name "*${label}" 2>/dev/null | sort -r | head -n 1)
        if [[ -n "$exp_dir" ]]; then
            break
        fi
        sleep 2
        attempt=$((attempt + 1))
    done

    if [[ -z "$exp_dir" ]]; then
        log_message "ERROR: Could not find experiment directory for $label"
        # Kill the background process
        kill $training_pid 2>/dev/null || true
        return 1
    fi

    # Wait for training process to finish spawning all nodes
    sleep 10

    # Monitor logs for completion
    wait_for_training_completion "$exp_dir" "$label"

    # Wait for the master_launch.sh process to actually finish
    log_message "Waiting for master_launch.sh (PID: $training_pid) to finish..."
    wait $training_pid 2>/dev/null || true
    log_message "Master launch process finished"

    # Give some time for cleanup
    sleep 5

    log_message "Training completed: $label"
    log_message "Results in: $exp_dir"
    log_message ""

    return 0
}

# Main execution
main() {
    log_message "========================================="
    log_message "Batch Training Script Started"
    log_message "========================================="
    log_message "Total combinations: ${#COMBINATIONS[@]}"
    log_message "State file: $STATE_FILE"
    log_message "Batch log: $BATCH_LOG"
    log_message "Number of nodes: $NUM_NODES"
    log_message ""

    local total=${#COMBINATIONS[@]}
    local completed=0
    local skipped=0
    local failed=0

    for combo in "${COMBINATIONS[@]}"; do
        # Parse combination
        IFS=':' read -r optimizer hw_queues streams config label extra_flags <<< "$combo"

        local combo_id="${optimizer}_hwq${hw_queues}_str${streams}"

        # Check if already completed
        if is_completed "$combo_id"; then
            log_message "SKIPPED (already completed): $combo_id"
            skipped=$((skipped + 1))
            continue
        fi

        # Run training (disable exit-on-error for this section)
        set +e
        run_training "$optimizer" "$hw_queues" "$streams" "$config" "$label" "$extra_flags"
        local run_result=$?
        set -e

        if [[ $run_result -eq 0 ]]; then
            mark_completed "$combo_id"
            completed=$((completed + 1))
        else
            log_message "FAILED: $combo_id (continuing to next combination)"
            echo "$combo_id FAILED" >> "${STATE_FILE}.failed"
            failed=$((failed + 1))
        fi

        log_message "Progress: $((completed + skipped))/$total completed, $failed failed"
        log_message ""
    done

    log_message "========================================="
    log_message "Batch Training Completed"
    log_message "========================================="
    log_message "Total combinations: $total"
    log_message "Completed: $completed"
    log_message "Skipped (resume): $skipped"
    log_message "Failed: $failed"
    log_message ""
    log_message "Full log: $BATCH_LOG"

    if [[ $failed -gt 0 ]]; then
        log_message "Failed combinations logged in: ${STATE_FILE}.failed"
    fi
}

# Run main
main
