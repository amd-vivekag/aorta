# HIP Issues That Can Trigger RCCL Racing and Data Corruption

**Date:** March 13, 2026
**Hardware:** AMD Instinct MI355X (gfx950), chi2880
**Software:** ROCm 7.1, PyTorch 2.11.0.dev, RCCL 2.27.7

---

## Key Finding

**There is no HIP or RCCL bug that causes RCCL data corruption.** The RCCL internal stream race that we reproduce at 200/200 corruption rate is caused by **ProcessGroupNCCL** (PyTorch's C++ wrapper), not by HIP or RCCL themselves.

We proved this by calling the RCCL C API directly (bypassing ProcessGroupNCCL) and showing that `hipEventQuery` works correctly — 0/200 corruption, ~3856 polls per event before reporting done.

---

## Proof: Pure HIP + RCCL C API — No Corruption

```bash
GPU_MAX_HW_QUEUES=4 torchrun --nproc_per_node=2 scripts/hip_event_cross_stream.py
```

This test uses raw `hipMalloc`, `hipStreamCreate`, `hipEventRecord/Query`, and the RCCL C API `ncclAllToAll` — **no PyTorch tensors, no CCA, no ProcessGroupNCCL**.

| Step | What happens |
|------|-------------|
| 1 | `hipMalloc` send_buf, fill with 42.0 on `user_stream` |
| 2 | `ncclAllToAll(send_buf, recv_buf, ..., user_stream)` |
| 3 | `hipEventRecord(ev, user_stream)` |
| 4 | Poll `hipEventQuery(ev)` → blocks for **~3856 polls** |
| 5 | `hipMemsetAsync(send_buf, 0xFF, side_stream)` — overwrite |
| 6 | Sync all streams, check recv_buf |

**Result: 0/200 corrupted.** The event correctly tracks the collective because **RCCL launches the kernel directly on `user_stream`**. The event polls ~3856 times, confirming the collective genuinely ran on `user_stream` and the event waited for it.

---

## Comparison: ProcessGroupNCCL — 200/200 Corruption

```bash
GPU_MAX_HW_QUEUES=4 torchrun --nproc_per_node=2 scripts/rccl_stream_race_mini.py
```

This test uses `torch.distributed.all_to_all_single(async_op=True)`, which goes through ProcessGroupNCCL.

**Result: 199-200/200 corrupted.** The event fires immediately (0 polls) because ProcessGroupNCCL creates its own internal `ncclStream` and passes that to RCCL instead of the user's stream.

---

## Root Cause: ProcessGroupNCCL's Internal Stream Design

```cpp
// ProcessGroupNCCL::collective() — PyTorch c10d
ncclStream = ncclStreams_.at(key);          // ← PGNCCL creates this, not RCCL
syncStream(device, event, ncclStream);     //   ncclStream waits for user_stream
ncclAllToAll(send, recv, ..., ncclStream); // ← RCCL runs on ncclStream
ncclEndEvent.record(ncclStream);           //   end event on ncclStream
```

ProcessGroupNCCL intercepts the user's stream and substitutes its own `ncclStream`. RCCL itself faithfully runs the collective on whatever stream it's given. The problem is that `ncclStream` is invisible to the user — they can only record events on their own stream, which has no collective work on it.

This is a **ProcessGroupNCCL design choice**, not a HIP or RCCL bug.

---

## Full Test Matrix (MI355X, chi2880)

### Pure HIP + RCCL C API (`hip_event_cross_stream.py`)

| GPU_MAX_HW_QUEUES | Corruption | Avg Polls | Conclusion |
|---|---|---|---|
| 4 | **0/200** | 3856 | hipEventQuery works correctly |

### ProcessGroupNCCL (`rccl_stream_race_mini.py`)

| GPU_MAX_HW_QUEUES | Corruption | Why |
|---|---|---|
| (default) | **200/200** | PGNCCL internal stream ≠ user stream |
| 4 | **199/200** | same |
| 2 | **199/200** | same |
| 1 | **0/200** | all streams serialize to one HW queue |

The HWQ=1 pass in the PGNCCL test is a **timing coincidence** — with one HW queue, all streams serialize, so the collective finishes before the overwrite reaches memory. It does not indicate a HIP bug is involved; it just masks the PGNCCL design issue.

---

## Implications

### What IS working correctly in HIP

- `hipEventRecord` on stream A correctly tracks all work on stream A
- `hipEventQuery` correctly reports completion only after stream A's work finishes
- RCCL C API faithfully runs collectives on the user-provided stream
- `hipStreamWaitEvent` correctly establishes cross-stream dependencies

### What IS the issue

- ProcessGroupNCCL creates internal streams and passes those to RCCL
- The user has no access to these internal streams
- `record_stream(user_stream)` is ineffective because the collective runs elsewhere
- PyTorch 2.11 mitigated this with CPU-side tensor stashing (PR [#148590](https://github.com/pytorch/pytorch/pull/148590))

### Related RCCL issue (deadlock, not corruption)

RCCL had a real deadlock bug on MI250X (Frontier) due to AMD's 64-thread wavefront vs NCCL's assumption of 32-thread warps in its pipeline synchronization (`NCCL_STEPS=8`). This was fixed upstream in RCCL. See [ROCm/rccl#1600](https://github.com/ROCm/rccl/pull/1600).

---

## Reproducer Scripts

| Script | What It Proves |
|--------|---------------|
| `scripts/hip_event_cross_stream.py` | HIP + RCCL C API: **0/200 corrupt** — hipEventQuery is correct |
| `scripts/rccl_stream_race_mini.py` | ProcessGroupNCCL: **200/200 corrupt** — internal stream design issue |
