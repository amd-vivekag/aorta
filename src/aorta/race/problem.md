# Meta NaN Issue Debug Report

## Context

We are AMD engineers helping our customer Meta debug a NaN issue on their recommendation system workload. Meta is not able to share the proprietary codebase for us to reproduce. The NaN issue does not happen on NVIDIA hardware.

## System Setup

PyTorch version: 2.11.0
Gcn arch name: gfx950:sramecc+:xnack-
ROCm version: 7.0.2.0-17-9428210
Rocblas version: 5.0.2-20250912-42-1205-g554bb20204
Hipblaslt version: 100200-7e32d53eb1
Model precision: fp32
Driver version: 6.16.6 ( rocm-smi --showdriverversion)

---

## March 6, 2026 -- Facts and Hypotheses after Meta Debug Session

---

## Issue A: Default Stream vs Side Stream Race

### Workload

Meta's eval workload runs a pipelined forward-pass loop with zero CPU-GPU synchronization. Each iteration has:
1. Prefetch (H2D + data distribution on side streams)
2. Compiled forward pass
3. Two metrics updated on default stream

### Evidence from Meta

- NaN disappears with `ROC_AQL_QUEUE_SIZE=1024` at batch sizes <= 512
- Disabling either side stream (memcpy or datadist) independently eliminates NaN
- AMD traces show 3-4 iteration CPU-GPU lag; NVIDIA traces show 1-2 iterations
- NaN appears at ~350 iters at bs=512 without mitigation; disappears entirely with AQL=1024

### Facts

- The CPU submits dispatch packets to the AQL queue. The GPU consumes them. The gap between the write pointer (CPU) and read pointer (GPU) is how far the CPU is ahead.
- On **NVIDIA**, the queue holds ~1K packets. When full, the CPU blocks (backpressure). The CPU can never get more than ~1K dispatches ahead of the GPU.
- On **AMD**, the queue holds up to 16K packets. The CPU can submit 10K+ dispatches before any backpressure. The GPU can fall thousands of dispatches behind.

### Hypothesis

When the CPU is far ahead, two memory recycling mechanisms cause corruption:

- **Kernarg recycling:** The HIP runtime reuses kernel argument buffers before the GPU reads them. The GPU executes a dispatch but finds arguments for a different kernel.
- **Tensor recycling:** PyTorch's caching allocator recycles GPU memory blocks before the GPU finishes reading them. The GPU reads overwritten data from a later iteration.

The corrupted data is still valid GPU memory (just wrong values), so the kernel runs without error but computes garbage, producing NaN. No crash, no error -- silent corruption.

Using multiple side streams (memcpy, datadist) alongside the default stream allows the CPU to submit packets in parallel across streams, filling the queue faster than with a single stream.

### Mitigations That Work for Issue A

- `ROC_AQL_QUEUE_SIZE=1024` -- Matches NVIDIA's queue depth, provides backpressure at 1K dispatches
- Moving side stream work to the default stream -- Serializes submission, CPU fills queue slower
- `GPU_MAX_HW_QUEUES=2` -- Reduces hardware parallelism, GPU keeps up better
- Any form of CPU-GPU sync (`.item()`, `synchronize()`) -- Drains the queue periodically

---

## Issue B: Large Batch + Pipelining NaN

### Workload

Same eval pipeline as Issue A: pipelined forward-pass loop with prefetch on side streams, compiled forward pass, and two metric updates on default stream.

### Evidence from Meta

- At bs >= 1024 with `ROC_AQL_QUEUE_SIZE=1024` (Issue A fully mitigated), NaN still appears
- Local run, 2 GPUs, bs=1024, AQL=1024 + gc=0: NaN ~340 iters
- MAST (2x8, bs=4096, AQL=1024): NaN from beginning
- MAST (2x8, bs=4096, AQL=1024 + gc=0): NaN (2/2 runs)
- `torch.cuda.synchronize()` at ALL pipeline points still produces NaN at bs=4096 -- queue depth is literally zero and NaN persists
- `EVAL_DISABLE_PIPELINING=1` (disabling prefetch) eliminates NaN at any batch size, including bs=4096
- `gc_collect_interval=0` (GC disabled) does NOT prevent Issue B

### Facts

- `torch.cuda.synchronize()` drains the AQL queue to zero (rptr catches up to wptr completely).
- Disabling pipelining means each iteration is independent: load data, compute, metrics, done, next iteration. No cross-iteration buffer sharing, no prefetch overlap.
- With pipelining enabled, the pipeline object manages buffers across iterations -- it prefetches iteration N+1's data while iteration N's compute is still running, and it reuses buffer objects between iterations.
- Larger batch sizes change kernel launch parameters (tile sizes, grid dimensions, working set size).

### Hypotheses

**Hypothesis A: Torch.compile / Triton codegen bug on ROCm at larger tensor dimensions**

Torch.compile in the forward generates Triton kernels. Triton's ROCm backend is less mature than the CUDA backend. At large batch sizes (bs >= 1024), Triton selects different tile sizes, grid dimensions, and memory access patterns. The pipelined graph structure (where prefetch tensors are inputs) produces a different compiled graph than the non-pipeline version. A codegen bug in a specific kernel configuration would:

- Be AMD-specific
- Not be fixed by sync (it's wrong code, not wrong timing)
- Only trigger at large batch sizes (different kernel parameters)
- Only trigger with pipelining (different compiled graph)
- Explain NaN from the beginning at bs=4096 (deterministic wrong code)

**Hypothesis B: HIP memory coherence / cache visibility bug at large working sets**

AMD GPUs have a different cache hierarchy and coherence model than NVIDIA. `synchronize()` ensures kernel completion but may not guarantee full memory writeback. At bs=4096, the working set may exceed certain cache thresholds, and stale data in caches could be read by subsequent kernels. With pipelining, buffers are reused across iterations (same virtual addresses), making cache staleness visible. Without pipelining, fresh allocations get different addresses, avoiding stale cache lines.

**Hypothesis C: Pipeline buffer management has an AMD-specific code path or interacts differently with HIP**

The pipeline object manages buffer reuse across iterations. If it has any HIP-specific behavior (or if HIP's memory mapping / pointer semantics differ subtly from CUDA), the pipeline could hand the wrong buffer contents to the forward pass. This would be a correctness bug in the buffer management logic specific to the ROCm path.

### Mitigations That Work for Issue B

- `EVAL_DISABLE_PIPELINING=1` -- Each iteration is fully independent
- Reducing batch size to <= 512

### Mitigations That Do NOT Work for Issue B

- `ROC_AQL_QUEUE_SIZE=1024` -- Fixes Issue A but not Issue B
- `torch.cuda.synchronize()` at all pipeline points (bs=4096) -- Queue depth is zero, NaN persists
- Sync at all pipeline points (bs=4096) -- Even with serialized stream execution, NaN persists
- `gc_collect_interval=0` (disable Python GC) at bs >= 1024 -- NaN persists
- `AQL=1024 + report_interval=10` at bs=4096 -- Reducing metric reporting frequency doesn't help

---

Jan 1, 2026

## Shampoo NaN Issue

Meta engineering team conducted systematic experiments to isolate the NaN source:

| # | Experiment | Configuration | NaN Observed? |
|---|-----------|---------------|---------------|
| 1 | Baseline | `GPU_MAX_HW_QUEUES=4` | YES |
| 2 | Serialized HW queues (workaround) | `GPU_MAX_HW_QUEUES=2` | YES |
| 3 | cudaStreamDefault flag | Changed `kDefaultFlags` in PyTorch `CUDAStream.cpp` | NO (but much slower) |
| 4 | Memcpy on default stream | H2D stream = `torch.cuda.default_stream()` | NO (5% faster than workaround) |
| 5 | Datadist on default stream | Datadist stream = `torch.cuda.default_stream()` | NO |
| 6 | H2D + datadist on same (non-default) stream | Both on 1 new stream, `GPU_MAX_HW_QUEUES=2` | YES |
| 7 | TorchRec base class instead of datadist | No datadist class | NO |
| 8 | RCCL implicit launch order | `NCCL_LAUNCH_ORDER_IMPLICIT=1` | NO (lower QPS) |
| 9 | RCCL cheap fence off | `RCCL_GFX942_CHEAP_FENCE_OFF=1` | YES |
| 10 | Implicit order + cheap fence off | `NCCL_LAUNCH_ORDER_IMPLICIT=1` + `RCCL_GFX942_CHEAP_FENCE_OFF=1` | NO (very slow) |
| 11 | AMD logging enabled | `AMD_LOG_LEVEL` set | NO |
| 12 | Disable SDMA | `HSA_ENABLE_SDMA=0` | YES |
| 13 | Forced blit copy | `GPU_FORCE_BLIT_COPY_SIZE=128` | YES |
| 14 | Increased signal pool | `ROC_SIGNAL_POOL_SIZE=16384` | YES |
| 15 | Disabled batch CPU sync | `DEBUG_CLR_BATCH_CPU_SYNC_SIZE=0` | YES |

### Key Observations from Experiments

- **HW queue parallelism is central:** NaN disappears when `GPU_MAX_HW_QUEUES=2` (serialized) but appears with `=4` (parallel).
- **Stream identity matters, not just count:** Row 6 shows NaN persists even with `GPU_MAX_HW_QUEUES=2` when H2D and datadist share a non-default stream. This means serializing HW queues alone is not the full fix -- what matters is *which stream* the work is placed on.
- **Default stream eliminates NaN:** Moving either H2D (row 4) or datadist (row 5) to the default stream eliminates NaN. Moving H2D to the default stream is the fastest workaround (~5% faster than the `GPU_MAX_HW_QUEUES=2` workaround).
- **AMD logging masks the bug:** Enabling AMD logging introduces enough serialization to hide the race condition (row 11).
- **RCCL fence alone doesn't help:** `RCCL_GFX942_CHEAP_FENCE_OFF=1` alone still produces NaN (row 9), but combined with `NCCL_LAUNCH_ORDER_IMPLICIT=1` it eliminates NaN at significant performance cost (row 10).
- **Not SDMA, signal-pool, or blit-copy related:** Rows 12-15 all still produce NaN, ruling out these mechanisms.
