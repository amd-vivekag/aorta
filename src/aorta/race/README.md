# RCCL Runtime Race Condition Reproducer

## What This Tool Does

A standalone test that checks whether **RCCL/HIP has a runtime-level bug** that silently corrupts data during multi-GPU distributed training. The test uses **correct synchronization everywhere** -- if data corruption still occurs, the bug is **in the runtime itself**, not in application code.

## Eval Workload (Start Here)

For the **pipelined eval NaN investigation** (Experiments A, B, CCA), see **[EXPERIMENTS.md](EXPERIMENTS.md)** -- that document contains the full test matrix, commands, decision trees, and YAML configs for reproducing and diagnosing NaN in pipelined eval loops.

**Quick smoke test** (run inside the docker container):

```bash
# Experiment A: queue depth race (expect NaN ~350 iters)
torchrun --nproc_per_node=2 -m aorta.race \
    --config config/race/eval_exp_a_reproduce.yaml

# CCA cross-stream race (expect integrity violation)
GPU_MAX_HW_QUEUES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --nproc_per_node=2 -m aorta.race \
    --config config/race/eval_exp_cca_calibrated.yaml

# Control: mitigated with AQL=1024 (expect clean)
torchrun --nproc_per_node=2 -m aorta.race \
    --config config/race/eval_exp_a_mitigated.yaml
```

## Quick Start (Other Modes)

```bash
# Default mode (TorchRec-like) — most common test
GPU_MAX_HW_QUEUES=4 torchrun --nproc_per_node=8 -m aorta.race \
    --warmup 10 --verify 100

# DDP mode (gradient all_reduce pattern)
torchrun --nproc_per_node=8 -m aorta.race --mode ddp \
    --warmup 100 --verify 10000 --deterministic

# FSDP mode (per-layer all_gather + reduce_scatter)
torchrun --nproc_per_node=8 -m aorta.race --mode fsdp \
    --warmup 10 --verify 100

# Same-stream mode (strongest proof of runtime bug)
torchrun --nproc_per_node=8 -m aorta.race --same-stream

# Multi-node (via launch script)
./scripts/multi_node/launch_reproducer.sh \
    --docker <container-name> --hw-queues 4 --warmup 100 --verify 10000
```

## Modes

| Mode | Pattern | What It Tests |
|------|---------|---------------|
| `default` | H2D + `all_to_all` + `all_reduce` | TorchRec-like recommendation model |
| `ddp` | H2D + gradient `all_reduce` | Distributed Data Parallel training |
| `fsdp` | Per-layer `all_gather` + `reduce_scatter` | Fully Sharded Data Parallel training |
| `eval_pipelined` | Pipelined eval with `torch.compile`, metrics, datadist | NaN investigation (Experiments A, B, CCA) |

### default (TorchRec-like)

```
 memcpy_stream:   [fill batch_cpu → copy to batch_gpu]
                                                       │ wait_stream()
                                                       ▼
  default_stream:                                     [Forward(batch_gpu)] → [Backward] ──────→ [all_reduce]
                                                                                           ▲
  datadist_stream:                                                         [all_to_all] ──┘ wait_stream()
```

Simulates a TorchRec recommendation model with H2D on `memcpy_stream`, `all_to_all` on `datadist_stream`, and compute + `all_reduce` on `default_stream`. Verifies H2D, all_to_all, and all_reduce correctness.

### ddp (Distributed Data Parallel)

Simulates DDP training with H2D (single or double-buffered with `--prefetch`), forward/backward with autograd, and gradient `all_reduce`. Supports non-bucketed (default) and bucketed per-layer all_reduce (`--bucketed`).

```bash
torchrun --nproc_per_node=8 -m aorta.race --mode ddp \
    --warmup 100 --verify 10000 --deterministic

# Bucketed (per-layer backward + all_reduce overlap)
torchrun --nproc_per_node=8 -m aorta.race --mode ddp --bucketed \
    --warmup 100 --verify 10000 --deterministic
```

### fsdp (Fully Sharded Data Parallel)

Simulates FSDP training with per-layer `all_gather` (reconstruct parameters) and `reduce_scatter` (shard gradients). Many small collectives interleaved with compute creates a different timing profile from bulk collectives.

```bash
torchrun --nproc_per_node=8 -m aorta.race --mode fsdp \
    --warmup 10 --verify 100
```

### eval_pipelined (Pipelined Eval NaN Investigation)

Replicates a pipelined eval loop to investigate NaN from three distinct root causes. Uses `torch.compile`, accumulated metrics (NE/MAE/calibration), DDP wrapper, and double-buffered pipeline with datadist.

Full documentation: **[EXPERIMENTS.md](EXPERIMENTS.md)**

## Test Matrix

| Test | Command | What It Proves |
|------|---------|----------------|
| **Baseline** | `--hw-queues 4` | Full HW queue parallelism -- most likely to trigger |
| **Serialized** | `--hw-queues 2` | Reduced parallelism -- if clean, parallelism-related |
| **Same-Stream** | `--same-stream` | Single stream. Corruption = definitive runtime bug |
| **No Compute** | `--no-compute` | Fast iteration, may not hit timing window |
| **H2D Prefetch** | `--prefetch` | Double-buffered H2D (any mode) |
| **DDP Bucketed** | `--mode ddp --bucketed` | Per-layer backward + all_reduce overlap |
| **FSDP** | `--mode fsdp` | Many small all_gather + reduce_scatter |
| **NCCL Implicit** | `--nccl-implicit-order` | Serialize RCCL ops via implicit ordering |
| **Eval Exp A** | `--config config/race/eval_exp_a_reproduce.yaml` | Queue depth race (NaN ~350 iters) |
| **Eval Exp B** | `--config config/race/eval_exp_b_reproduce.yaml` | Large batch + pipeline NaN |
| **Eval CCA** | `--config config/race/eval_exp_cca_calibrated.yaml` | CCA cross-stream recycling race |

## CLI Options

### Core

| Option | Default | Description |
|--------|---------|-------------|
| `--mode MODE` | `default` | Workload: `default`, `ddp`, `fsdp`, `eval_pipelined` |
| `--config PATH` | - | YAML config file (CLI overrides YAML) |
| `--warmup N` | 100 | Warmup iterations (no verification) |
| `--verify N` | 10000 | Verification iterations |
| `--no-compute` | - | Skip GEMM simulation |
| `--same-stream` | - | H2D and datadist on same stream |
| `--prefetch` | - | Double-buffered H2D prefetch |
| `--no-stop-on-first` | - | Continue after first corruption |
| `--deterministic` | - | Fixed seeds for DDP gradient verification |
| `--bucketed` | - | Per-layer gradient all_reduce (DDP) |
| `--fsdp-shard-size N` | 100000 | FSDP shard size per rank |
| `--optimizer OPT` | `none` | Optimizer: `none`, `adamw`, `sgd`, `shampoo` |

### Environment Variable Flags

| Flag | Env Variable | Effect |
|------|--------------|--------|
| `--hw-queues N` | `GPU_MAX_HW_QUEUES=N` | HW queue count (4 = exposes bug) |
| `--nccl-implicit-order` | `NCCL_LAUNCH_ORDER_IMPLICIT=1` + `RCCL_ENABLE_CONTEXT_TRACKING=1` | Serialize RCCL |
| `--disable-sdma` | `HSA_ENABLE_SDMA=0` | Disable SDMA engine |
| `--disable-cheap-fence` | `RCCL_GFX9_CHEAP_FENCE_OFF=1` | Disable fence optimization |
| `--aql-queue-size N` | `ROC_AQL_QUEUE_SIZE=N` | AQL queue depth (1024 mitigates Exp A) |
| `--signal-pool-size N` | `ROC_SIGNAL_POOL_SIZE=N` | HSA signal pool size |

## Output

### Pass
```
PASSED: No corruption in 10100 iterations with proper synchronization
VERDICT: No runtime bug detected with current settings.
```

### Fail
```
RUNTIME BUG DETECTED: 15 corruptions in 5432 iterations
Corruption occurred DESPITE proper synchronization - this is a bug in RCCL/HIP runtime
VERDICT: RUNTIME BUG DETECTED!
```

## Interpreting Results

| Baseline (HW=4) | Serialized (HW=2) | Same-Stream | Conclusion |
|-----------------|-------------------|-------------|------------|
| Fail | Pass | Pass | Runtime bug triggered by HW queue parallelism |
| Fail | Pass | Fail | Runtime bug in stream ordering itself |
| Pass | Pass | Pass | No runtime bug detected |
| Fail | Fail | Fail | Possible hardware issue |

## Adding a New Mode

1. Create `modes/your_mode.py` inheriting from `BaseReproducer`
2. Implement `setup_buffers()` and `run_iteration()`
3. Register in `modes/__init__.py`
4. Add `--mode your_mode` to CLI choices in `__main__.py`

See `developer_guide.md` for the full walkthrough.

## References

- **Eval Workload & NaN Experiments:** [EXPERIMENTS.md](EXPERIMENTS.md) -- **Start here** for pipelined eval testing (Experiments A, B, CCA with full decision trees, commands, and configs)
- **RCCL Fence Stress Test:** `scripts/rccl_fence_stress.py` -- Concurrent collectives from multiple process groups
- **Background & Concepts:** `background.md` -- Distributed training patterns, streams, and the race condition
- **YAML Configs:** `config/race/eval_exp_*.yaml`
- **Multi-node Scripts:** `scripts/multi_node/launch_reproducer.sh`
- **Developer Guide:** `developer_guide.md`
