# HIP Event Premature Completion -- Investigation and Corrected Findings

**Date:** March 10, 2026
**Hardware:** AMD Instinct MI355X (gfx950)
**Software:** PyTorch 2.11.0.dev20260215+rocm7.1, ROCm 7.1
**Reproducer:** `scripts/meta_nan_hip_event_race.py`

---

## 1. Background

Meta reported NaN corruption in their pipelined RecSys eval workload on AMD GPUs. The CUDA Stream Sanitizer (CSAN) detected a data race between `data_dist_stream` and `default_stream`: PyTorch's CachingAllocator (CCA) recycled a memory block while an async RCCL `alltoall_base_` on another stream was still reading from it.

One hypothesis proposed: **HIP's `hipEventQuery()` returns success before the GPU has actually finished using the memory**, causing the CCA to recycle blocks too early. This document describes our investigation of this hypothesis, including a critical bug found in our initial test, and the corrected results.

---

## 2. The CachingAllocator Event-Based Block Recycling Mechanism

When a tensor is used across multiple CUDA/HIP streams, PyTorch's CachingAllocator must ensure the underlying memory block is not freed until all streams are done with it. The mechanism:

### Step 1: Cross-stream usage triggers `recordStream`

When tensor `T` (allocated on stream A) is passed to an operator on stream B, the code must call `T.record_stream(stream_B)` (or PyTorch calls it internally for certain ops). This marks the block as "in use by stream B."

**Critical detail:** PyTorch does NOT automatically call `record_stream` when you use a tensor inside a `with torch.cuda.stream(stream_B):` context. The `with` block only changes the current stream for new operations -- it does NOT register existing tensors with the CCA for that stream. You must explicitly call `tensor.record_stream(stream)`, or use an op that does so internally (e.g., NCCL collectives call it on their input/output tensors).

### Step 2: Python reference is dropped

When the Python reference to `T` is deleted (`del t`), the CCA does NOT immediately free the block. Instead, for each stream that was registered via `record_stream` (besides the owning stream), it:

1. Creates a `hipEvent` and records it on that stream: `hipEventRecord(event, stream_B)`
2. Moves the block to a **pending-free list** with its associated events

### Step 3: `process_events()` polls for completion

Later, when a new allocation request comes in, the CCA calls `process_events()` to check if any pending blocks can be recycled:

```
for each (block, event) in pending_free_list:
    if hipEventQuery(event) == hipSuccess:
        move block back to free pool
    else:
        keep in pending list
```

### Step 4: Block is recycled

A subsequent `torch.empty()` or `torch.full()` can now receive the recycled block from the free pool.

### Without `record_stream`

If `record_stream` is never called, **no event is recorded on stream B**. The CCA has no knowledge that stream B is using the block. When the Python reference is dropped, the block goes directly to the free pool -- no pending list, no event polling. Any subsequent allocation can immediately reuse the block, even if stream B is still reading from it.

---

## 3. Bug in Initial Test (v1)

### The problem

Our initial reproducer (`meta_nan_hip_event_race.py` v1) never called `tensor.record_stream()`. The code pattern was:

```python
with torch.cuda.stream(producer):
    t = torch.full((numel,), 42.0, device=device)

consumer.wait_stream(producer)
with torch.cuda.stream(consumer):
    result = make_heavy_workload(t, num_ops=50)

del t  # BUG: CCA has no event for consumer stream
       # Block goes directly to free pool

with torch.cuda.stream(producer):
    new_t = torch.full((numel,), -999.0, device=device)  # Gets same block immediately
```

The CCA never recorded an event on the consumer stream, so the block was immediately recyclable. The "corruption" we observed was the expected behavior of overwriting a block that the CCA correctly (from its perspective) considered free.

### Proof

Direct test confirming the block is immediately recycled without `record_stream`:

```
Without record_stream: 20/20 allocations got the same block (immediate reuse)
With record_stream:     0/20 allocations got the same block (CCA held it)
```

### Why we mistakenly attributed it to hipEventQuery

The corruption was systematic (always pairs 2-3 in saturate mode) and correlated with `GPU_MAX_HW_QUEUES`:
- HWQ=4: corruption on every iteration
- HWQ=1: no corruption

This made it look like a HW queue visibility issue. In reality, HWQ=1 serializes all streams to a single hardware queue, so the GPU naturally finishes the consumer's work before the producer's overwrite arrives -- masking the missing `record_stream` bug. With HWQ=4, the hardware queues run more independently, and the producer's overwrite reaches memory before the consumer finishes reading.

The `torch.cuda.synchronize()` between free and realloc also "fixed" it because it drained all queues, not because it was simulating `hipEventSynchronize`.

---

## 4. Corrected Test (v2)

### Changes

Added `tensor.record_stream(consumer_stream)` before cross-stream usage in all modes. Added `--skip-record-stream` flag as a control to reproduce the v1 bug for comparison.

### Results

All tests on a single MI355X GPU, node `chi2881`, `GPU_MAX_HW_QUEUES=4`, `ROC_AQL_QUEUE_SIZE=1024`.

#### Saturate Mode (4 stream pairs, 50 chain ops, 64MB tensors, 16 alloc pressure)

| Configuration | Corruptions | Ptr Reuses | Iterations |
|---------------|-------------|------------|------------|
| **v2 with record_stream** | **0** | 3 | 5,000 |
| **v2 --skip-record-stream (v1 bug)** | **9,998** | 20,000 | 5,000 |

#### Alloc-Free Mode (2 streams, 50 chain ops, mixed sizes 1-256MB, 16 alloc pressure, "high" pressure)

| Configuration | Corruptions | Ptr Reuses | Iterations |
|---------------|-------------|------------|------------|
| **v2 with record_stream** | **0** | 4,004 | 10,000 |

#### Interpretation

With proper `record_stream` usage:
- The CCA correctly holds blocks until `hipEventQuery()` confirms the consumer stream has finished
- 4,004 ptr reuses occurred across 10,000 iterations, meaning the CCA did poll and recycle blocks -- but only **after** the GPU was truly done
- **Zero corruption** in 15,000 total iterations across both modes
- `hipEventQuery()` is working correctly on MI355X

### Baseline Validation

| Test | Result | Proves |
|------|--------|--------|
| Single stream, same math | rel_err = 1.7e-6 | Expected value calculation is correct |
| Cross-stream, tensor kept alive (no CCA free) | rel_err = 1.7e-6 | Cross-stream handoff is correct |
| Cross-stream, freed, no alloc pressure | rel_err = 1.7e-6 | CCA pending-free path is correct |
| v2 with record_stream, HWQ=4, heavy load | 0/5,000 corrupt | **hipEventQuery is correct** |
| v2 --skip-record-stream, HWQ=4 | 9,998/10,000 corrupt | Missing record_stream causes corruption |

---

## 5. Implications for Meta's NaN Issue

### hipEventQuery is NOT the root cause

Our corrected test shows that `hipEventQuery()` correctly reports event completion on MI355X with `GPU_MAX_HW_QUEUES=4`. The CCA's event-based recycling mechanism works as designed when `record_stream` is properly used.

### What CSAN actually detected

The CSAN race report showed `data_dist_stream` accessing memory that `default_stream` was still using via `alltoall_base_`. This is consistent with a **missing `record_stream` call** somewhere in the PyTorch/TorchRec pipeline -- not a HIP event bug.

Possible locations where `record_stream` might be missing:
1. **TorchRec's `TrainPipelineSparseDist`** may not call `record_stream` when passing tensors between its pipeline stages
2. **RCCL/NCCL `all_to_all_single`** may not internally call `record_stream` on its input buffer for the calling stream when `async_op=True`
3. **`torch.compile` / Triton codegen** may produce code that doesn't preserve `record_stream` annotations from eager mode

### Why GPU_MAX_HW_QUEUES and other mitigations work

| Mitigation | Why it works |
|------------|-------------|
| `GPU_MAX_HW_QUEUES=1` or `=2` | Serializes streams to fewer HW queues, so the GPU naturally finishes reading before the overwrite arrives. Masks the missing `record_stream`. |
| Default stream only | The default stream has implicit synchronization semantics: all operations on non-default streams implicitly synchronize with the default stream. This makes `record_stream` unnecessary. |
| `NCCL_LAUNCH_ORDER_IMPLICIT=1` | Serializes RCCL operations, reducing the window where a missing `record_stream` can cause corruption. |
| `AMD_LOG_LEVEL` logging | Introduces CPU-side delay, giving the GPU time to finish before the CCA recycles blocks. |
| `torch.cuda.synchronize()` | Drains all queues, making `record_stream` unnecessary. |
| `PYTORCH_NO_CUDA_MEMORY_CACHING=1` | Disables CCA entirely. Every alloc/free goes through `hipMalloc`/`hipFree`, which synchronizes implicitly. Bypasses the `record_stream`/`hipEventQuery` path completely. **Best diagnostic test** -- if NaN disappears, the CCA recycling path is confirmed as the cause. |

All of these are **timing mitigations** that hide a missing `record_stream`, not fixes for a HIP bug (except `PYTORCH_NO_CUDA_MEMORY_CACHING=1`, which eliminates the CCA path entirely).

---

## 6. Recommended Next Steps

### Find the missing `record_stream` -- RCCL internal stream is the prime suspect

**New finding (v4 stress test):** We confirmed that RCCL `all_to_all_single(async_op=True)` reads `send_buf` on an **RCCL-internal stream** that is invisible to the user. If an event is recorded on the user-visible `default_stream`, it only covers the user's compute ops -- not the RCCL kernel. When the event reports success and the CCA recycles the block, RCCL is still reading it.

Reproducer (`scripts/meta_nan_hip_event_stress.py --mode rccl_event_race`): corruption on every iteration starting from iter 2. `max_diff=1083.0` on recv buffer, confirming the RCCL kernel read overwritten data (`-999.0` instead of `84.0`). The event polls ~24,000 times before reporting success, confirming the compute chain on `default_stream` took real GPU time -- but RCCL's read on its internal stream was still in progress.

This matches Meta's CSAN report exactly: the CCA hands out `send_buf`'s block because the event on `default_stream` (or `data_dist_stream`) says "done", but RCCL's internal kernel is still reading it.

**Key question:** Does `ProcessGroupNCCL` call `record_stream(nccl_internal_stream)` on the input tensor when `async_op=True`? If not, this is the root cause. The fix is to add `input_tensor.record_stream(nccl_work_stream)` in `ProcessGroupNCCL::alltoall_base_` before returning the `Work` handle.

1. **Audit `ProcessGroupNCCL.cpp`**: Check whether async collective inputs get `record_stream` called for the NCCL/RCCL work stream.

2. **Audit TorchRec's pipeline**: When tensors are passed between `memcpy_stream`, `default_stream`, and `data_dist_stream`, are they properly `record_stream`'d?

3. **Check `torch.compile` interaction**: Does compilation strip `record_stream` calls that exist in eager mode?

### Ask Meta to test `PYTORCH_NO_CUDA_MEMORY_CACHING=1`

This is the simplest diagnostic: set the environment variable `PYTORCH_NO_CUDA_MEMORY_CACHING=1` and re-run the NaN workload. This disables the CachingAllocator entirely -- every allocation goes through `hipMalloc()` and every free goes through `hipFree()`, which is a synchronizing operation that waits for all streams to finish before releasing memory.

- **If NaN disappears**: the bug is in the CCA's block recycling path (missing `record_stream` or premature `hipEventQuery`). No code changes required, just an env var.
- **If NaN persists**: the corruption source is elsewhere (kernel bug, Triton codegen, RCCL data corruption), and CCA recycling is not the cause.

This is much simpler than patching `hipEventSynchronize` into the CCA, and requires zero code changes on Meta's side. It will be significantly slower due to `hipMalloc`/`hipFree` overhead on every allocation, but for a diagnostic run that's acceptable.

### If CCA is confirmed: find the missing `record_stream`

Have Meta insert explicit `record_stream` calls at the pipeline stage boundaries in their TorchRec eval workload. If this fixes the NaN without needing `GPU_MAX_HW_QUEUES=1` or disabling the CCA, it pinpoints exactly where the annotation is missing.

### Fix the CCA's stream tracking for RCCL ops

If RCCL's async collectives do not call `record_stream` on their input buffers, this should be added in PyTorch's RCCL process group implementation (`ProcessGroupNCCL.cpp`).

---

## 7. How to Reproduce

```bash
# hipEventQuery correctness: should pass (0 corruption)
GPU_MAX_HW_QUEUES=4 .venv/bin/python3 scripts/meta_nan_hip_event_race.py \
    --mode saturate --iterations 5000 --chain-ops 50 --size-mb 64 --alloc-pressure 16

# RCCL internal stream race: CORRUPTS on every iteration
# Demonstrates that RCCL reads send_buf on an internal stream invisible to user events
GPU_MAX_HW_QUEUES=4 torchrun --nproc_per_node=2 scripts/meta_nan_hip_event_stress.py \
    --mode rccl_event_race --iterations 100 --chain-ops 50 --size-mb 64

# Single-GPU stress tests: all pass (hipEventQuery is correct for compute kernels)
GPU_MAX_HW_QUEUES=4 .venv/bin/python3 scripts/meta_nan_hip_event_stress.py \
    --mode stream_fanout --iterations 2000 --chain-ops 50 --size-mb 64
GPU_MAX_HW_QUEUES=4 .venv/bin/python3 scripts/meta_nan_hip_event_stress.py \
    --mode h2d_compute_race --iterations 5000 --chain-ops 50 --size-mb 64
GPU_MAX_HW_QUEUES=4 .venv/bin/python3 scripts/meta_nan_hip_event_stress.py \
    --mode threaded_poll --iterations 5000 --chain-ops 50 --size-mb 64
```
