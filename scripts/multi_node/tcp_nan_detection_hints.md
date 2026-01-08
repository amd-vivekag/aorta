# Stream Race NaN Detection Setup

## Goal
Reproduce NaN caused by stream race condition (aux reads gradients before compute finishes backward).

## Config: `config/multi_node/shampoo_opt_multi_node_seed42.yaml`

### Required Settings
```yaml
aux_wait_compute_after_backward: false   # Disable sync to allow race
debug_stream_race_report: true           # Log when race occurs
debug_stream_race_report_wait: false     # Don't fix, just report
```

### Extend Race Window (optional)
```yaml
extend_backward_compute_ms: 50   # Add 50ms compute after backward
```

## Code Changes

### 1. Detect race NaN in `nan_debugger.py`
- `check_gradients()` now detects when `isfinite()` returns False but `isnan().sum()` returns 0
- This proves stream race: data changed between reads
- Logs: `STREAM RACE DETECTED` and stops training

### 2. Check before clip in `fsdp_trainer.py`
- Moved `check_gradients()` into same block as `track_parameter_evolution()`
- Runs BEFORE `clip_grad_norm_()` to catch race window
- Stops immediately on detection

### 3. Extend race window in `fsdp_trainer.py`
- New config: `extend_backward_compute_ms`
- Adds matmul compute on compute stream after backward
- Keeps compute busy while aux tries to read gradients

## Expected Log Output
```
[StreamRaceReport] aux reached param tracking before compute backward finished
[NaNDebugger] STREAM RACE DETECTED: isfinite().all() returned False but num_nan=0
[NaNDebugger] NaN/Inf detected in gradients (pre-clip race window) - stopping training
```

## Fix
Set `aux_wait_compute_after_backward: true` to enforce proper stream ordering.
