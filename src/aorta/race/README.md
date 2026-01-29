# Race Condition Injection Module

This module provides tools to inject controlled race conditions for testing distributed training robustness. It simulates scenarios where H2D memcpy and RCCL collectives race on different GPU streams.

## Overview

Race conditions in distributed training can cause:
- Silent data corruption
- NaN values in loss/gradients
- Training hangs
- Non-deterministic behavior

This module enables controlled injection of race conditions to:
- Reproduce issues seen in production
- Test robustness of synchronization patterns
- Validate fixes for race-related bugs

## Race Categories

### 1. H2D Race (Host-to-Device Memory Copy)

Simulates race conditions between H2D memory transfers and forward pass computation.

**How it works:**
- Batch data is copied to GPU on a separate `memcpy_stream`
- If `h2d_skip_sync_before_forward=True`, the forward pass starts before H2D completes
- This creates a race window where the model reads uninitialized/partial data

**Configuration:**
```yaml
race_experiment:
  h2d_memcpy_racing: true           # Enable separate memcpy stream
  h2d_skip_sync_before_forward: true # Skip synchronization (causes race!)
  h2d_racing_start_step: 3          # Start racing after warmup
```

### 2. Datadist Race (TorchRec-style all_to_all)

Simulates race conditions in sparse data distribution patterns like TorchRec's `SparseDataDistributedAllToAll`.

**How it works:**
- `all_to_all` operations run on a separate `datadist_stream`
- If `datadist_skip_sync_before_collective=True`, FSDP collectives start before `all_to_all` completes
- This creates a race between data distribution and model parameter synchronization

**Configuration:**
```yaml
race_experiment:
  datadist_racing: true                      # Enable separate datadist stream
  datadist_skip_sync_before_collective: true # Skip synchronization (causes race!)
  datadist_racing_start_step: 3              # Start racing after warmup
```

### 3. Timing Skew Experiment

Introduces controlled timing delays to demonstrate how timing variations affect training.

**Modes:**
- `none`: No artificial skew
- `fixed`: Fixed delay in microseconds
- `progressive`: Delay increases each step (`skew_us * step`)
- `random`: Random delay within range

**Configuration:**
```yaml
race_experiment:
  timing_skew_enabled: true
  timing_skew_mode: "fixed"    # none, fixed, progressive, random
  timing_skew_us: 500          # Delay in microseconds
  timing_skew_ranks: [0]       # Which ranks get delayed (empty = all)
  timing_skew_start_step: 3    # Start after warmup
```

## GPU Hardware Queue Settings

The `GPU_MAX_HW_QUEUES` environment variable controls hardware queue parallelism and is critical for race exposure:

| Value | Behavior | Race Exposure |
|-------|----------|---------------|
| 1-2 | Streams share HW queues | Implicit serialization masks races |
| 4+ | Each stream gets own HW queue | True parallelism exposes races |

**Configuration:**
```yaml
race_experiment:
  gpu_max_hw_queues: 4  # Set before GPU initialization
```

## Supporting Options

### Warmup Control

```yaml
race_experiment:
  skip_training_warmup: false  # Skip training warmup for timing variability
  training_warmup_steps: 1     # Number of warmup steps
  skip_rccl_warmup: false      # Skip RCCL communicator warmup
  rccl_warmup_iterations: 10   # RCCL warmup iterations
```

### NaN Checking

```yaml
race_experiment:
  nan_check_collectives: true  # Check for NaN before/after RCCL collectives
```

## Configuration Reference

All options are set under the `race_experiment:` section in YAML config files.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `h2d_memcpy_racing` | bool | false | Use separate memcpy stream for H2D |
| `h2d_skip_sync_before_forward` | bool | false | Skip sync before forward (causes race) |
| `h2d_racing_start_step` | int | 0 | Step to start H2D racing |
| `datadist_racing` | bool | false | Use separate stream for all_to_all |
| `datadist_skip_sync_before_collective` | bool | false | Skip sync before collective (causes race) |
| `datadist_racing_start_step` | int | 0 | Step to start datadist racing |
| `timing_skew_enabled` | bool | false | Enable timing skew experiment |
| `timing_skew_mode` | str | "none" | Skew mode (none/fixed/progressive/random) |
| `timing_skew_us` | int | 0 | Delay in microseconds |
| `timing_skew_ranks` | list | [] | Ranks to delay (empty = all) |
| `timing_skew_start_step` | int | 3 | Step to start timing skew |
| `skip_training_warmup` | bool | false | Skip training warmup |
| `training_warmup_steps` | int | 1 | Training warmup steps |
| `skip_rccl_warmup` | bool | false | Skip RCCL warmup |
| `rccl_warmup_iterations` | int | 10 | RCCL warmup iterations |
| `nan_check_collectives` | bool | false | Enable NaN checking |
| `gpu_max_hw_queues` | int | null | Set GPU_MAX_HW_QUEUES env var |

## Example Configurations

### Aggressive H2D Race Testing

```yaml
race_experiment:
  gpu_max_hw_queues: 4
  h2d_memcpy_racing: true
  h2d_skip_sync_before_forward: true
  h2d_racing_start_step: 0
  nan_check_collectives: true
```

### TorchRec-style Datadist Race

```yaml
race_experiment:
  gpu_max_hw_queues: 4
  datadist_racing: true
  datadist_skip_sync_before_collective: true
  datadist_racing_start_step: 3
  nan_check_collectives: true
```

### Combined Race Testing

```yaml
race_experiment:
  gpu_max_hw_queues: 4
  h2d_memcpy_racing: true
  h2d_skip_sync_before_forward: true
  datadist_racing: true
  datadist_skip_sync_before_collective: true
  nan_check_collectives: true
```

## Module Structure

- `config.py` - `RaceConfig` dataclass with all configuration options
- `h2d_racing.py` - H2D memory copy racing implementation
- `datadist_racing.py` - Datadist/all_to_all racing implementation
- `timing_skew_experiment.py` - Controlled timing skew injection
- `injectors.py` - High-level injection interface for the trainer
- `inflight_checks.py` - Repeated in-flight reads to detect torn reads
- `correctness_verification.py` - Manual verification to detect silent corruption
