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

# DDP mode (gradient all_reduce pattern)
torchrun --nproc_per_node=8 -m aorta.race \
    --warmup 100 --verify 10000 \
    -o mode=ddp
```

## Test Configurations

| Test | Command | Purpose |
|------|---------|---------|
| **Baseline** | `--hw-queues 4` | True stream parallelism |
| **Serialized** | `--hw-queues 2` | Reduced parallelism (comparison) |
| **Same-Stream** | `--same-stream` | Definitive runtime bug test |
| **No Compute** | `--no-compute` | Fast iteration (~5ms/step) |
| **NCCL Implicit** | `--nccl-implicit` | Serialized NCCL ordering |
| **DDP Mode** | `-o mode=ddp` | Test gradient all_reduce pattern |

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
| `-o key=value` | - | Override config options (e.g., `-o mode=ddp`) |

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

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        __main__.py                              │
│                    (CLI entry point)                            │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     modes/__init__.py                           │
│                  create_reproducer(config)                      │
└─────────────────────────────┬───────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│   modes/default.py      │     │     modes/ddp.py        │
│ DefaultModeReproducer   │     │   DDPModeReproducer     │
│ (all_to_all + all_reduce)│    │ (gradient all_reduce)   │
└────────────┬────────────┘     └────────────┬────────────┘
             │                               │
             └───────────────┬───────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                         base.py                                 │
│                     BaseReproducer                              │
│  (streams, compute, optimizer, run loop, verification)         │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        compute.py                               │
│              BaseCompute → GEMMCompute                          │
│         (forward/backward simulation)                           │
└─────────────────────────────────────────────────────────────────┘
```

### File Structure

```
src/aorta/race/
├── __init__.py              # Public API exports
├── __main__.py              # CLI entry point
├── config.py                # ReproducerConfig, ReproducerResult
├── minimal_reproducer.py    # Legacy reproducer (backward compat)
├── base.py                  # BaseReproducer abstract class
├── compute.py               # Pluggable compute simulation
├── modes/                   # Mode implementations
│   ├── __init__.py          # Factory function
│   ├── default.py           # TorchRec-like mode (all_to_all + all_reduce)
│   └── ddp.py               # DDP mode (gradient all_reduce)
└── README.md                # This file
```

### Available Modes

| Mode | Description | Data Flow |
|------|-------------|-----------|
| `default` | TorchRec-like pattern | H2D → Forward → Backward + all_to_all → all_reduce |
| `ddp` | DDP gradient sync | H2D (double-buffered) → Forward → Backward → gradient all_reduce |

### Data Flow: Default Mode (TorchRec-like)

```
memcpy_stream:  [H2D] → batch_gpu
                          ↓ (Forward READS batch_gpu)
default_stream:          [Forward] → [Backward] → [all_reduce]

datadist_stream:         [all_to_all]
                          (overlaps with backward)
```

### Data Flow: DDP Mode

```
Iteration N:
    memcpy_stream:  [H2D batch_N+1] ────────────────────────┐
                                                            │ (prefetch overlaps)
    default_stream: [Forward(batch_N)] → [Backward] → [all_reduce grads]
                                                            │
                    ← swap buffers ─────────────────────────┘
```

---

## Developer Guide

### Adding a New Mode

To add a new reproducer mode (e.g., FSDP, Pipeline Parallel):

#### 1. Create the mode file

Create `src/aorta/race/modes/your_mode.py`:

```python
"""Your mode description."""

import torch
import torch.distributed as dist

from aorta.race.base import BaseReproducer


class YourModeReproducer(BaseReproducer):
    """Your mode reproducer implementation."""

    def setup_buffers(self) -> None:
        """Allocate mode-specific buffers."""
        # Required: Create your GPU buffers
        self.my_buffer = torch.zeros(
            self.config.tensor_size,
            dtype=self.dtype,
            device=self.device,
        )
        # ... more buffers as needed

    def run_iteration(self, iteration: int) -> bool:
        """
        Run one iteration of the mode-specific data flow.
        
        Args:
            iteration: Current iteration number
            
        Returns:
            True if verification passed, False if corruption detected
        """
        # 1. Prepare data with known pattern
        expected_value = float(iteration % 1000)
        self.batch_cpu.fill_(expected_value)
        
        # 2. H2D transfer on memcpy_stream
        with torch.cuda.stream(self.memcpy_stream):
            self.batch_gpu.copy_(self.batch_cpu, non_blocking=True)
        
        # 3. Record event for synchronization
        self.h2d_complete.record(self.memcpy_stream)
        
        # 4. Wait for H2D before compute
        self.default_stream.wait_event(self.h2d_complete)
        
        # 5. Run your compute/communication pattern
        # ... your mode-specific logic here
        
        # 6. Verify data integrity (during verification phase)
        if iteration >= self.config.warmup_iterations:
            return self._verify_my_data(expected_value)
        
        return True

    def _verify_my_data(self, expected: float) -> bool:
        """Verify your mode-specific data."""
        # Check for corruption
        if not torch.allclose(self.my_buffer, expected_tensor):
            self.corruption_count += 1
            return False
        return True


__all__ = ["YourModeReproducer"]
```

#### 2. Register the mode

Edit `src/aorta/race/modes/__init__.py`:

```python
from aorta.race.modes.your_mode import YourModeReproducer

MODE_REGISTRY = {
    "default": DefaultModeReproducer,
    "ddp": DDPModeReproducer,
    "your_mode": YourModeReproducer,  # Add your mode
}
```

#### 3. Use your mode

```bash
torchrun --nproc_per_node=8 -m aorta.race -o mode=your_mode
```

### Adding a New Compute Type

To add a new compute pattern (e.g., Attention, Embedding Lookup):

#### 1. Create the compute class

Edit `src/aorta/race/compute.py`:

```python
class AttentionCompute(BaseCompute):
    """Attention-based compute simulation."""

    def setup(self, requires_grad: bool = False) -> None:
        """Initialize attention layers."""
        self.query = torch.nn.Linear(
            self.config.gemm_size, self.config.gemm_size
        ).to(self.device, dtype=self.dtype)
        self.key = torch.nn.Linear(
            self.config.gemm_size, self.config.gemm_size
        ).to(self.device, dtype=self.dtype)
        self.value = torch.nn.Linear(
            self.config.gemm_size, self.config.gemm_size
        ).to(self.device, dtype=self.dtype)
        
        if requires_grad:
            for p in self.parameters():
                p.requires_grad_(True)

    def forward(self, batch_gpu: torch.Tensor) -> torch.Tensor:
        """Run attention forward pass."""
        # Reshape input for attention
        x = batch_gpu.view(-1, self.config.gemm_size)
        
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.config.gemm_size ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)
        
        return output.sum()

    def backward(self, output: torch.Tensor, use_autograd: bool = False) -> None:
        """Run attention backward pass."""
        if use_autograd:
            output.backward()
        else:
            # Manual backward simulation
            for _ in range(self.config.gemm_layers):
                dummy = torch.matmul(
                    torch.randn(self.config.gemm_size, self.config.gemm_size, 
                               device=self.device, dtype=self.dtype),
                    torch.randn(self.config.gemm_size, self.config.gemm_size,
                               device=self.device, dtype=self.dtype)
                )

    def parameters(self):
        """Return trainable parameters."""
        return list(self.query.parameters()) + \
               list(self.key.parameters()) + \
               list(self.value.parameters())
```

#### 2. Register the compute type

```python
# At the bottom of compute.py
register_compute("attention", AttentionCompute)
```

#### 3. Use your compute type

Update `config/race/minimal_reproducer.yaml`:

```yaml
compute_type: attention  # instead of "gemm"
```

### Key Design Patterns

1. **Template Method Pattern**: `BaseReproducer.run()` defines the algorithm structure, subclasses implement `setup_buffers()` and `run_iteration()`.

2. **Pluggable Compute**: Compute is separate from the reproducer so different compute patterns can be tested without modifying mode logic.

3. **Registry Pattern**: `MODE_REGISTRY` and `COMPUTE_REGISTRY` allow adding new types without modifying existing code.

4. **Verification in Base Class**: `_verify_h2d()` is shared since all modes need H2D verification. Mode-specific verification is in subclasses.

### BaseReproducer Provided Utilities

The `BaseReproducer` class provides these utilities for subclasses:

| Attribute/Method | Description |
|------------------|-------------|
| `self.memcpy_stream` | CUDA stream for H2D transfers |
| `self.default_stream` | Default CUDA stream for compute |
| `self.h2d_complete` | CUDA event for H2D synchronization |
| `self.batch_cpu` | Pinned CPU tensor for H2D source |
| `self.batch_gpu` | GPU tensor for H2D destination |
| `self.compute` | Compute simulator instance |
| `self.optimizer` | Optimizer instance (if configured) |
| `self._verify_h2d(expected)` | Verify H2D data integrity |
| `self._log_progress(iteration)` | Log iteration progress |

---

## References

- **Config Reference:** `config/race/minimal_reproducer.yaml`
- **Environment Variables:** `config/race/customer_env_vars.yaml`
- **Multi-node Scripts:** `scripts/multi_node/`
- **NCCL/RCCL Settings:** `scripts/multi_node/set_env_variables.sh`
