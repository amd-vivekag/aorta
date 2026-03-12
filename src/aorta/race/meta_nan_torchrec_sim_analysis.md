# Fixed-Buffer Pipeline Crash — Root Cause Analysis

**Date:** March 12, 2026
**Script:** `scripts/meta_nan_torchrec_sim.py --pipelined`
**Hardware:** AMD Instinct MI355X, ROCm 7.1, PyTorch 2.11.0.dev

---

## Root Cause

**Missing `memcpy_stream.wait_stream(default_stream)` before H2D writes.**

The 3-slot pipeline has a one-directional synchronization bug: `default_stream` waits for `memcpy_stream` (so the forward pass sees completed H2D data), but `memcpy_stream` never waits for `default_stream` (so H2D can overwrite a buffer the GPU is still reading).

With 3 rotating slots, each slot is reused every 3 iterations:

| Iter | H2D writes (memcpy_stream) | Forward reads (default_stream) |
|------|----------------------------|-------------------------------|
| 0 | slot2 | **slot0** |
| 1 | **slot0** | slot1 |
| 2 | slot1 | **slot2** |
| 3 | **slot2** | slot0 |

At iteration 1, `memcpy_stream` writes slot0 while iteration 0's forward pass may still be reading slot0 on `default_stream`. On AMD, the CPU races hundreds of dispatches ahead (AQL queue gap ~400-700), so the GPU hasn't finished iteration 0 when iteration 1's H2D arrives. The `EmbeddingBag` kernel reads corrupted `offsets`, computes out-of-bounds ranges, and crashes.

## The Fix

One line before each H2D:

```python
memcpy_stream.wait_stream(default_stream)  # wait for forward to finish reading
pipeline.h2d_into_slot(2, memcpy_stream)
```

## Test Results

| Configuration | Iters | Result |
|---|---|---|
| Original (no reverse wait) | 1000 | **CRASH** `0x1016` (rptr=295, wptr=785) |
| Original, system default AQL (~16K) | 1000 | **CRASH** `0x1016` (rptr=279, wptr=702) |
| Original + `sync_per_iter` | 1000 | PASS |
| **+ `memcpy_stream.wait_stream(default_stream)`** | 3000 | **PASS** (249 it/s) |
| **+ reverse wait, system default AQL** | 3000 | **PASS** (249 it/s) |

All crashes hit `EmbeddingBag_updateOutputKernel_sum_mean<BFloat16, long>` with `HSA_STATUS_ERROR_EXCEPTION: 0x1016`.

## Why This Only Crashes on AMD

On NVIDIA/CUDA, the command queue depth is ~1K and the GPU typically keeps pace with the CPU, so the forward pass finishes before the next H2D arrives at the same slot. On AMD/ROCm, the 16K AQL queue lets the CPU race hundreds of iterations ahead, making the WAR (write-after-read) hazard reliably exploitable.

## Distinction from record_stream Issue

This is a **separate mechanism** from the caching-allocator race fixed by `record_stream`:

| | Caching allocator race | Fixed-buffer race (this bug) |
|---|---|---|
| Buffer management | `.to()` allocates new tensors | `copy_()` into pre-allocated slots |
| What recycles | Allocator reuses freed memory | Same physical buffer via slot rotation |
| Fix | `tensor.record_stream(stream)` | `memcpy_stream.wait_stream(default_stream)` |
| `record_stream` relevant? | Yes | No |

## Commands

```bash
# Reproduce the crash
PYTHONPATH=scripts torchrun --nproc_per_node=2 \
    scripts/meta_nan_torchrec_sim.py --pipelined --batch-size 4096 --no-compile

# Verify sync mitigation
PYTHONPATH=scripts torchrun --nproc_per_node=2 \
    scripts/meta_nan_torchrec_sim.py --pipelined --batch-size 4096 --no-compile --sync-per-iter
```
