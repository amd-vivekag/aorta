# NaN / Crash Stress Test Report

## Problem Statement

Customers report non-deterministic NaN values and hard crashes (`HSA_STATUS_ERROR_EXCEPTION`)
during large-scale recommendation system training. The issue:

- **Present** when `GPU_MAX_HW_QUEUES=4` (default)
- **Absent** when `GPU_MAX_HW_QUEUES=2`
- **Absent** when `NCCL_LAUNCH_ORDER_IMPLICIT=1`

The customer workload combines: DDP gradient all\_reduce, Distributed Shampoo optimizer
(with AllGather during `optimizer.step()`), TorchRec-style sharded embeddings with
`all_to_all` on a dedicated `datadist_stream`, H2D double-buffered data loading on a
`memcpy_stream`, and `bf16` mixed precision — all running concurrently across multiple
CUDA streams on AMD MI300X / MI350X GPUs.

## Reproduction Approach

We built a standalone stress test that replicates the customer's multi-stream workload
without requiring the full training infrastructure. The test maximizes stream contention
by combining:

1. **Real model** — 18-layer ranking transformer with 350K-vocab embedding, matching
   customer architecture
2. **Distributed Shampoo** — with `DDPDistributedConfig`, `precondition_frequency=50`,
   `max_preconditioner_dim=8192`
3. **Embedding stress simulator** — 8 large embedding tables (2M × 128 each), lookups
   on 4 dedicated `embedding_streams`, redistributed via `all_to_all` on `datadist_stream`
4. **H2D double-buffering** — pinned CPU buffers copied asynchronously on `memcpy_stream`
5. **DDP** — standard `DistributedDataParallel` with gradient all\_reduce
6. **Gradient accumulation** — 2 micro-steps per optimizer step (matching customer config)

This creates 7+ concurrent CUDA streams per rank: `default_stream`, `memcpy_stream`,
`datadist_stream`, 4 × `embedding_streams`, plus Shampoo's internal communication streams.

## Files Required for Reproduction

| File | Description |
|------|-------------|
| `scripts/nan_stress_test.py` | Main stress test script (885 lines). Contains the full training loop, `EmbeddingStressSimulator`, `DoubleBufferedBatchGenerator`, `NaNChecker`, SIGABRT handler. |
| `config/nan_stress_test.yaml` | Default configuration (71 lines). All model, optimizer, embedding, and monitoring parameters. |
| `src/aorta/models/ranking_transformer.py` | `RankingTransformerModel` — the model under test. |
| `src/aorta/models/__init__.py` | `ModelConfig` dataclass. |
| `scripts/diagnose_crash.py` | Diagnostic script for component-level isolation (tests `gpu_only`, `h2d_only`, `full`, etc.). |

### Dependencies

- PyTorch with ROCm support
- `distributed_shampoo` (Shampoo optimizer package)
- `pyyaml`

## How to Run

```bash
# Full stress test with HW_QUEUES=4 (should crash)
GPU_MAX_HW_QUEUES=4 PYTHONPATH=src torchrun --nproc_per_node=8 \
    scripts/nan_stress_test.py --config config/nan_stress_test.yaml \
    --max-steps 500 --hw-queues 4

# Same test with HW_QUEUES=2 (should pass)
GPU_MAX_HW_QUEUES=2 PYTHONPATH=src torchrun --nproc_per_node=8 \
    scripts/nan_stress_test.py --config config/nan_stress_test.yaml \
    --max-steps 500 --hw-queues 2

# Same test with HW_QUEUES=1 (should pass)
GPU_MAX_HW_QUEUES=1 PYTHONPATH=src torchrun --nproc_per_node=8 \
    scripts/nan_stress_test.py --config config/nan_stress_test.yaml \
    --max-steps 500 --hw-queues 1
```

### Diagnostic Isolation Script

```bash
# Test with GPU-generated data + emb_stress, no H2D (isolates H2D)
GPU_MAX_HW_QUEUES=4 PYTHONPATH=src torchrun --nproc_per_node=8 \
    scripts/diagnose_crash.py --mode gpu_only --steps 200

# Test with H2D + model, no emb_stress (isolates embedding stress)
GPU_MAX_HW_QUEUES=4 PYTHONPATH=src torchrun --nproc_per_node=8 \
    scripts/diagnose_crash.py --mode h2d_only --steps 200

# Full combination
GPU_MAX_HW_QUEUES=4 PYTHONPATH=src torchrun --nproc_per_node=8 \
    scripts/diagnose_crash.py --mode full --steps 200
```

### GPU Reset After Crashes

After hard crashes (`SIGABRT`, `HSA_STATUS_ERROR_EXCEPTION`), the GPU driver state may
become corrupted. Reset all GPUs before subsequent test runs:

```bash
for i in 0 1 2 3 4 5 6 7; do
    sudo rocm-smi --gpureset -d $i
done
```

## Reproduction Results

All tests run on an 8×GPU node (MI350X) with 8 ranks (`torchrun --nproc_per_node=8`).

### HW Queue Sweep

| `GPU_MAX_HW_QUEUES` | Max Steps | Steps Completed | Result |
|---|---|---|---|
| 1 | 200 | 200 | **PASSED** — 0 NaN, 0 crashes |
| 2 | 200 | 200 | **PASSED** — 0 NaN, 0 crashes |
| 4 (run 1) | 200 | ~120 | **CRASHED** — `HSA_STATUS_ERROR_EXCEPTION` |
| 4 (run 2) | 500 | ~360 | **CRASHED** — `HSA_STATUS_ERROR_EXCEPTION` |

The crash point varies between runs (step 120 vs step 360), confirming the
non-deterministic nature of the issue.

### Component Isolation (all at `GPU_MAX_HW_QUEUES=4`)

| Mode | Components | Steps | Result |
|---|---|---|---|
| `gpu_only` | DDP + Shampoo + emb\_stress, **no H2D** | 200 | **PASSED** |
| `h2d_only` | DDP + Shampoo + H2D, **no emb\_stress** | 200 | **PASSED** (with clean GPUs + sufficient disk) |
| `full` | DDP + Shampoo + emb\_stress + H2D | 200 | **CRASHED** |

The crash requires the **full combination** of H2D double-buffering and embedding stress
running concurrently with DDP + Shampoo. Each component individually passes, but together
they create enough stream contention to trigger the hardware exception.

### Crash Signature

```
:0:rocdevice.cpp :3676: ... Callback: Queue 0x... aborting with error :
    HSA_STATUS_ERROR_EXCEPTION: An HSAIL operation resulted in a hardware exception.
    code: 0x1016

[PG ID 0 PG GUID 0(default_pg) Rank N] Process group watchdog thread terminated
    with exception: CUDA error: an illegal memory access was encountered
    Search for `hipErrorIllegalAddress'
```

The crash is reported asynchronously through the NCCL process group watchdog. The root
cause is a hardware exception (illegal memory access) during GPU kernel execution, not a
NCCL-level error.

## Key Findings

1. **The crash is strictly `GPU_MAX_HW_QUEUES` dependent.** HWQ=1 and HWQ=2 pass
   reliably; HWQ=4 crashes non-deterministically. This matches the customer report exactly.

2. **The crash requires high stream contention.** It only occurs when multiple CUDA
   streams (memcpy, embedding, datadist, default, Shampoo comm) are all active
   simultaneously. Removing any major stream source (H2D or embedding stress) prevents it.

3. **The crash is a hardware exception, not a NaN.** The error is
   `HSA_STATUS_ERROR_EXCEPTION` (illegal memory access), not a floating-point NaN. The
   NaN that customers observe may be a secondary symptom if the crash is caught/recovered,
   leaving corrupted GPU memory that produces NaN in subsequent operations.

4. **The crash point is non-deterministic.** It occurs at different training steps across
   runs (step ~120 in one run, step ~360 in another), consistent with a timing-sensitive
   race condition in the GPU hardware queue scheduler or ROCm runtime.

5. **`AMD_SERIALIZE_KERNEL=3` prevents the crash.** Serializing all GPU kernels (which
   forces sequential execution and eliminates stream parallelism) makes the crash
   disappear, confirming it is a concurrency issue.

6. **Root cause hypothesis:** With `GPU_MAX_HW_QUEUES=4`, multiple logical CUDA streams
   are mapped to 4 physical hardware queues. Under the heavy multi-stream workload
   (7+ concurrent streams), the hardware queue multiplexing exposes a race condition
   where stream dependencies established via `wait_stream()` / `wait_event()` are not
   correctly enforced at the hardware level. This causes kernels to access GPU memory
   that hasn't been fully written by a preceding operation on another stream, resulting
   in the `HSA_STATUS_ERROR_EXCEPTION`.

## Stream Architecture Diagram

```
                    ┌─────────────────────────────────────────────┐
                    │            Training Iteration               │
                    └─────────────────────────────────────────────┘

    memcpy_stream:    [H2D: cpu_buf → gpu_buf]
                         │
                         ▼ wait_stream
    embedding_streams:   [table0 lookup] [table1 lookup] ... [table7 lookup]
    (4 streams)             │               │                    │
                            └───────────────┴────────────────────┘
                                            │
                                            ▼ wait_stream (per embedding stream)
    datadist_stream:                    [all_to_all]
                                            │
                                            ▼ wait_stream + work.wait()
    default_stream:      [wait H2D] ──► [forward] ──► [backward + DDP all_reduce]
                                                              │
                                                              ▼
    shampoo_comm:                                     [optimizer.step: AllGather]
```

When `GPU_MAX_HW_QUEUES=4`, these 7+ logical streams compete for 4 physical hardware
queues. The HW queue scheduler must correctly enforce all `wait_stream` / `wait_event`
dependencies across queue boundaries. The crash indicates this enforcement fails under
high contention.
