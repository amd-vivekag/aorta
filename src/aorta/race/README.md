# RCCL Runtime Race Condition Reproducer

A standalone tool to detect **runtime-level bugs** in RCCL/HIP that can manifest in multi-node distributed training with overlapping streams.

## Purpose

Distributed training workloads use multiple concurrent streams for overlapping compute, communication, and data movement. This module:

- Tests for RCCL/HIP runtime ordering violations
- Uses known-pattern data to detect ANY data corruption
- Simulates realistic training timing profiles
- Provides minimal reproducers for runtime bug reports

## Quick Start

```bash
# Single-node validation (8 GPUs)
GPU_MAX_HW_QUEUES=4 torchrun --nproc_per_node=8 -m aorta.race \
    --warmup 10 --verify 100

# Multi-node test (via launch script)
./scripts/multi_node/launch_reproducer.sh \
    --docker <container-name> \
    --hw-queues 4 \
    --warmup 100 \
    --verify 10000

# Same-stream mode (definitive runtime bug test)
./scripts/multi_node/launch_reproducer.sh \
    --docker <container-name> \
    --hw-queues 4 \
    --same-stream
```

## Test Configurations

| Test | Command | Purpose |
|------|---------|---------|
| **Baseline** | `--hw-queues 4` | True stream parallelism |
| **Serialized** | `--hw-queues 2` | Reduced parallelism (comparison) |
| **Same-Stream** | `--same-stream` | Definitive runtime bug test |
| **No Compute** | `--no-compute` | Fast iteration (~5ms/step) |
| **NCCL Implicit** | `--nccl-implicit` | Serialized NCCL ordering |

## Command-Line Options

### Basic Options

| Option | Default | Description |
|--------|---------|-------------|
| `--warmup N` | 100 | Warmup iterations (no verification) |
| `--verify N` | 10000 | Verification iterations |
| `--no-compute` | - | Skip compute simulation |
| `--same-stream` | - | H2D + datadist on same stream |
| `--no-stop-on-first` | - | Continue after first corruption |
| `--gemm-size N` | 5120 | GEMM matrix size |
| `--gemm-layers N` | 26 | Number of GEMM layers |

### Environment Variable Flags

| Flag | Env Variable | Effect |
|------|--------------|--------|
| `--hw-queues N` | `GPU_MAX_HW_QUEUES=N` | Control HW queue count |
| `--nccl-implicit-order` | `NCCL_LAUNCH_ORDER_IMPLICIT=1` | Serialize NCCL ops |
| `--disable-sdma` | `HSA_ENABLE_SDMA=0` | Disable SDMA engine |
| `--signal-pool-size N` | `ROC_SIGNAL_POOL_SIZE=N` | HSA signal pool size |
| `--disable-cheap-fence` | `RCCL_GFX9_CHEAP_FENCE_OFF=1` | Disable fence optimization |

## Output

### Pass
```
PASSED: No corruption in 10100 iterations with proper synchronization
VERDICT: No runtime bug detected with current settings.
```

### Fail (Runtime Bug Detected)
```
RUNTIME BUG DETECTED: 15 corruptions in 5432 iterations
Corruption occurred DESPITE proper synchronization - this is a bug in RCCL/HIP runtime
VERDICT: RUNTIME BUG DETECTED!
```

### Experiment Directory

Each run saves results to `experiments/reproducer_hw<N>_<timestamp>_<label>/`:

```
experiments/reproducer_hw4_20260129_211127_test/
├── logs/
│   ├── node_0.txt
│   └── node_1.txt
├── config/
│   ├── run_config.yaml      # Actual config used
│   ├── minimal_reproducer.yaml
│   └── set_env_variables.sh
└── experiment_info.txt
```

## Interpreting Results

| Baseline (HW=4) | Serialized (HW=2) | Same-Stream | Conclusion |
|-----------------|-------------------|-------------|------------|
| Fail | Pass | Pass | Runtime bug triggered by parallelism |
| Fail | Pass | Fail | Runtime bug in stream ordering |
| Pass | Pass | Pass | No runtime bug detected |
| Fail | Fail | Fail | Possible hardware issue |

## Architecture

```
src/aorta/race/
├── __init__.py
├── __main__.py              # CLI entry point
├── minimal_reproducer.py    # Core reproducer logic
├── config.py                # Configuration dataclass
└── README.md
```

### Data Flow

```
memcpy_stream:  [H2D] → batch_gpu
                          ↓ (Forward READS batch_gpu)
default_stream:          [Forward] → [Backward] → [all_reduce]

datadist_stream:         [all_to_all]
                          (overlaps with backward)
```

## References

- **Config Reference:** `config/race/minimal_reproducer.yaml`
- **Multi-node Scripts:** `scripts/multi_node/`
- **NCCL/RCCL Settings:** `scripts/multi_node/set_env_variables.sh`
