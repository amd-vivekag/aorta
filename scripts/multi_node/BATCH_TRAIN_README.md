# Batch Training Script

Automated script for running 12 training combinations (2 optimizers × 2 hw-queues × 3 streams).

## Features

- **12 Combinations**: Runs all combinations of:
  - Optimizers: Shampoo, AdamW
  - Hardware Queues: 2, 4
  - Streams: 2, 4, 6

- **Resume Support**: Automatically tracks completed runs and skips them on restart
- **Intelligent Monitoring**: Waits for all node logs to show completion before proceeding
- **Failure Handling**: Logs failures but continues with remaining combinations
- **Grouped Execution**: Runs all Shampoo combinations first, then all AdamW combinations

## Usage

### Basic Usage

```bash
cd /home/vivekag/scratch/apps/aorta_work/aorta
./scripts/multi_node/batch_train.sh
```

The script will:
1. Start running combinations in order
2. Monitor logs for completion (checks all node logs)
3. Move to the next combination automatically
4. Log all progress to a timestamped batch log file

### Resuming After Interruption

If the script is interrupted (Ctrl+C, system restart, etc.), simply run it again:

```bash
./scripts/multi_node/batch_train.sh
```

It will automatically skip already-completed combinations using the state file `.batch_train_state`.

### Monitoring Progress

During execution, you can:

**View the batch log:**
```bash
tail -f batch_train_*.log
```

**Check current training:**
```bash
# Find the most recent experiment directory
ls -lt experiments/ | head -5

# Monitor specific experiment
tail -f experiments/multinode_28ch_256th_<timestamp>_<label>/logs/node_*.txt
```

**Check state file:**
```bash
cat .batch_train_state  # Shows completed combinations
```

## Output Files

- **`batch_train_<timestamp>.log`**: Complete log of all batch operations
- **`.batch_train_state`**: Tracks completed combinations (for resume)
- **`.batch_train_state.failed`**: Lists failed combinations
- **`experiments/`**: Individual experiment directories for each run

## Training Combinations

The script runs these combinations in order:

### Shampoo Optimizer (6 combinations)
1. shampoo_hwq2_str2 (hw-queues=2, streams=2, --tcp)
2. shampoo_hwq2_str4 (hw-queues=2, streams=4, --tcp)
3. shampoo_hwq2_str6 (hw-queues=2, streams=6, --tcp)
4. shampoo_hwq4_str2 (hw-queues=4, streams=2, --tcp)
5. shampoo_hwq4_str4 (hw-queues=4, streams=4, --tcp)
6. shampoo_hwq4_str6 (hw-queues=4, streams=6, --tcp)

### AdamW Optimizer (6 combinations)
7. adam_hwq2_str2 (hw-queues=2, streams=2, --tcp)
8. adam_hwq2_str4 (hw-queues=2, streams=4, --tcp)
9. adam_hwq2_str6 (hw-queues=2, streams=6, --tcp)
10. adam_hwq4_str2 (hw-queues=4, streams=2, --tcp)
11. adam_hwq4_str4 (hw-queues=4, streams=4, --tcp)
12. adam_hwq4_str6 (hw-queues=4, streams=6, --tcp)

## Completion Detection

The script monitors all node log files and waits for the completion message:
```
============================================
Node X training completed
============================================
```

Only when **ALL** nodes show this message does the script proceed to the next combination.

## Troubleshooting

### Script stuck waiting

If the script appears stuck:
1. Check the most recent experiment logs
2. Verify training is actually running: `ps aux | grep torchrun`
3. Check for errors in node logs

### Want to reset and start over

```bash
rm .batch_train_state
rm .batch_train_state.failed
./scripts/multi_node/batch_train.sh
```

### Want to skip a specific combination

Manually add the combination ID to `.batch_train_state`:
```bash
echo "adam_hwq2_str4" >> .batch_train_state
```

### Check which combinations are pending

```bash
# Compare total combinations (12) with completed
wc -l .batch_train_state
```

## Configuration

To modify the script behavior, edit `batch_train.sh`:

- **CHANNELS**: NCCL channels (default: 28)
- **THREADS**: RCCL threads per block (default: 256)
- **NPROC**: Processes per node (default: 8)
- **DOCKER_CONTAINER**: Docker container name
- **check_interval**: Seconds between log checks (default: 30)

## Example Output

```
[2026-01-12 13:00:00] =========================================
[2026-01-12 13:00:00] Batch Training Script Started
[2026-01-12 13:00:00] =========================================
[2026-01-12 13:00:00] Total combinations: 12
[2026-01-12 13:00:00] Number of nodes: 3
[2026-01-12 13:00:00]
[2026-01-12 13:00:00] =========================================
[2026-01-12 13:00:00] Starting training: shampoo_hwq2_str2
[2026-01-12 13:00:00]   Optimizer: shampoo
[2026-01-12 13:00:00]   HW Queues: 2
[2026-01-12 13:00:00]   Streams: 2
[2026-01-12 13:00:00] =========================================
[2026-01-12 13:00:05] Training launched with PID: 12345
[2026-01-12 13:00:10] Monitoring training for: shampoo_hwq2_str2
[2026-01-12 13:00:10] Progress: 0/3 nodes completed
[2026-01-12 13:05:40] Progress: 3/3 nodes completed
[2026-01-12 13:05:40] SUCCESS: All 3 nodes completed training
[2026-01-12 13:05:40] Training completed: shampoo_hwq2_str2
...
```

## Notes

- Each training run can take significant time (varies by workload)
- The script will run continuously until all 12 combinations complete
- You can safely stop and resume using Ctrl+C
- All experiment artifacts are preserved in individual directories
