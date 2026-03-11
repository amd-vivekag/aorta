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

## Stack

- **Model:** DLRMv3 / HSTU (Hierarchical Sequential Transduction Unit) -- [MLCommons reference](https://github.com/mlcommons/inference/tree/master/recommendation/dlrm_v3)
- **Framework:** PyTorch 2.11.0 + ROCm 7.0.2, `torch.compile` with Triton backend
- **Embeddings:** TorchRec `EmbeddingCollection` -- 1B item table (dim=64, bf16), user + category tables, sharded across GPUs via `KeyedJaggedTensor`
- **Training pipeline:** TorchRec `TrainPipelineSparseDist` -- 3-stage pipeline with triple-buffered device tensors:
  - `memcpy_stream`: H2D copy from pinned host memory (iteration N+2)
  - `data_dist_stream`: RCCL `all_to_all` embedding redistribution (iteration N+1)
  - `default_stream`: compiled forward pass + metrics (iteration N)
- **Collectives:** RCCL `all_to_all_single` / `reduce_scatter` via `ProcessGroupNCCL` with `async_op=True`
- **Dense forward:** 5-layer multi-head self-attention + FFN + LayerNorm, output MLP (256 -> 1), bf16 mixed precision
- **Hardware:** AMD Instinct MI350X (gfx950), 252 GB HBM per GPU

---

## CUDA Stream Sanitizer (CSAN) Results on Meta's NaN Workload

**Date:** March 2026
**Configuration:** `ROC_AQL_QUEUE_SIZE=1024`, `GPU_MAX_HW_QUEUES=1`, `CSAN=1`

Meta ran the PyTorch CUDA Stream Sanitizer against their eval workload to detect stream-level data races.

### What CSAN Detected

CSAN found a **data race on a tensor** (data pointer `139991783932688`) between two streams:

| | Stream | Operation | Pipeline Stage |
|---|--------|-----------|----------------|
| **Access 1** | `data_dist_stream` (id `140011564741504`) | `aten::empty.memory_format` -- allocating a new tensor | `wait_sparse_data_dist` |
| **Access 2** | `default_stream` (id `0`) | `c10d::alltoall_base_` -- reading input tensor | `model_fwd` |

### Code Paths from Stack Traces

**data_dist_stream side:**
`trainer.py` -> `_train_loop` -> `progress()` -> `wait_sparse_data_dist()` -> `KJTAllToAllTensorsAwaitable.__init__()` at `dist_data.py:402` -> `torch.empty(...)`.
The CachingAllocator (CCA) returned a memory block at the raced address.

**default_stream side:**
`alltoall_base_(input=..., async_op=True)` -- the NCCL/RCCL all-to-all collective was still reading from a tensor at that same address as its input argument. The collective was launched asynchronously.

### The Race Scenario

```
Timeline:
  default_stream:     alltoall_base_(input=0x7F4B...0A10, async_op=True) --> still reading input
  data_dist_stream:   torch.empty() --> CCA returns block at 0x7F4B...0A10 --> OVERWRITES
```

The CachingAllocator reused the memory block while the async `alltoall_base_` on the default stream was still reading from it.

### Meta's Interpretation

> "CSAN detected race between the data_dist_stream and the default_stream. The former is requesting a buffer during wait_sparse_data_dist stage of pipelining, while the latter is doing some NCCL ops (all2all here, but I've also seen reduce scatter etc.) during model_fwd. My interpretation: when the data_dist_stream requests the buffer, CachingAllocator handed out a block that was still in use by the default stream."

### Three Hypotheses Discussed

**Hypothesis 1: NCCL/RCCL dropped a tensor reference (RULED OUT)**

Meta initially suspected that NCCL internal code failed to keep a reference to the all-to-all input tensor, allowing the CCA to recycle it. They examined a specific PR that introduced a new tensor to hold the A2A input but failed to keep refs. However, `ProcessGroupNCCL.cpp` correctly stashes both inputs and outputs of async collective ops. They could not reproduce the race by manually queuing async comms + allocating on another stream.

**Hypothesis 2: HIP event completion reports success prematurely (STRONGEST LEAD)**

Proposed by Jeremy Hadidjojo at Meta. The mechanism:

1. When a cross-stream buffer is deleted, the CachingAllocator inserts `hipEvent` on every stream using it
2. The block doesn't get freed until all events report completion
3. If `hipEventQuery()` returns `hipSuccess` **before the GPU actually finishes** using the memory, the CCA would recycle the block too early

This would be an AMD-specific bug in HIP event management -- the event signals completion to the CPU before the GPU has actually finished the associated kernel/collective.

This hypothesis is consistent with the Shampoo experiment matrix:

- **Row 2** (`GPU_MAX_HW_QUEUES=2` fixes): Fewer HW queues = fewer concurrent operations = smaller window for premature event completion
- **Row 4** (H2D on default stream fixes): No cross-stream event needed -- same-stream ordering guaranteed by queue semantics
- **Row 6** (same non-default stream still races): Even on the same user-stream, if RCCL internally uses sub-streams or the HW queue mapping differs from default stream, event behavior changes
- **Row 8** (`NCCL_LAUNCH_ORDER_IMPLICIT=1` fixes): Forces RCCL to serialize launches, making events complete in order
- **Row 11** (logging masks bug): Extra serialization from logging gives events time to "catch up"

**Hypothesis 3: CSAN false positive due to async c10d::Work**

Raised by Jeff Daily. CSAN may not understand the semantics of `async_op=True` in NCCL collectives. When `alltoall_base_` returns a `c10d.Work` handle, the GPU operation continues after the Python call returns. CSAN hooks at the dispatcher level and records accesses at dispatch time -- it may not model the extended GPU-side lifetime of async collective ops. If CSAN thinks the `alltoall_base_` is "done" when the Python call returns, it would incorrectly flag a subsequent allocation as a race.

However, even if the CSAN report is a false positive in its detection mechanism, the NaN is real. The report may still be pointing at the right tensor and the right streams.

### Assessment

**Hypothesis 2 (premature HIP event completion) is the strongest lead** because:

1. It's AMD-specific -- CUDA's event implementation is battle-tested; HIP's maps to HSA signals which have different completion semantics
2. It explains the full Shampoo experiment matrix (serialization, default stream, implicit launch order all reduce the window for premature completion)
3. It's mechanistically sound -- the CCA relies on `hipEventQuery()` to decide when to recycle blocks, and if HSA signals report completion before memory writes are globally visible, the CCA would hand out still-in-use memory

### Recommended Next Steps

1. **Test the HIP event hypothesis:** Patch PyTorch's CachingAllocator to call `hipEventSynchronize()` instead of `hipEventQuery()` in `process_events()`. If NaN disappears, this confirms premature HIP event completion.

2. **Check HSA signal behavior:** On the AMD side, verify that `hsa_signal_wait_*()` used by HIP events correctly waits for memory write visibility (not just kernel completion). There may be a missing cache flush / memory fence between the RCCL kernel's last memory write and the HSA signal being decremented.

3. **Validate CSAN on CUDA:** Run the same workload on NVIDIA with CSAN enabled. If the same race is flagged on CUDA but doesn't produce NaN, it confirms a CSAN false positive for async ops. If it's NOT flagged on CUDA, the race is AMD-specific.

4. **Instrument CCA event polling:** Add logging to `process_events()` to record when events are queried, when they return complete, and when blocks are recycled. Compare the recycled block addresses against active RCCL collective inputs to catch premature reuse in the act.

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

## Jan 1, 2026

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
