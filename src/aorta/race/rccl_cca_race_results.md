# RCCL Internal Stream Race

## Bug

`record_stream(user_stream)` does not protect tensors used by async NCCL collectives, because the collective runs on an **internal `ncclStream`**, not on `user_stream`.

## Root Cause

When you call `dist.all_to_all_single(recv, send, async_op=True)` on `user_stream`, ProcessGroupNCCL does this internally (`ProcessGroupNCCL.cpp`):

```
ncclStream = ncclStreams_.at(key)        // ← internal stream, NOT user_stream
syncStream(device, ev, ncclStream)       //   ncclStream waits for user_stream
ncclAllToAll(send, recv, ..., ncclStream) // ← collective reads send on ncclStream
ncclEndEvent.record(ncclStream)          //   end event on ncclStream
```

After the call returns, `user_stream` has **zero** collective kernels on it. All collective work is on `ncclStream`.

`record_stream(user_stream)` tells the CCA: "record an event on `user_stream`; don't recycle until that event completes." But `user_stream` is idle — so the event fires immediately. CCA recycles the block while `ncclStream` is still reading it → **corruption**.

## Mini Reproducer

```bash
# 200/200 corrupted on MI355X
GPU_MAX_HW_QUEUES=4 torchrun --nproc_per_node=2 scripts/rccl_stream_race_mini.py
```

Each iteration simulates what `record_stream(user_stream)` + CCA recycling would do:

1. Fill `send_buf = 42.0`
2. `all_to_all(async_op=True)` on `user_s` → RCCL runs on `ncclStream`
3. `hipEventRecord` on `user_s`, poll → fires immediately (no collective on `user_s`)
4. Overwrite `send_buf = -999` on `side_s` (simulates CCA handing out the same block)
5. `work.wait()` + sync → `recv_buf` = -999 (RCCL read the overwritten data)

## Why PyTorch 2.11 is Not Affected

PyTorch 2.11 replaced `record_stream` with **C++ tensor stashing** for async collectives:

```cpp
// ProcessGroupNCCL::collective()
work->stashed_for_allocator_safety_->stash(inputs);  // holds shared_ptr
```

The stash holds a C++ reference that the CCA cannot bypass. Only `work.wait()` releases it — after blocking on `ncclEndEvent` (recorded on `ncclStream`). So the stash is not released until the collective actually finishes.

## Test Matrix (MI355X)

| HWQ | CCA | Mode | Result | Why |
|-----|-----|------|--------|-----|
| 4 | ON | raw | **200/200 CORRUPT** | `user_stream` event fires before `ncclStream` done |
| 1 | ON | raw | PASS | HWQ=1 serializes all streams |
| 4 | OFF | raw | PASS | `hipMalloc`/`hipFree` serializes |
| 4 | ON | pipeline | PASS | C++ stashing holds refs |

## Implication

Any code that relies on `record_stream(user_stream)` for tensor lifetime across async NCCL ops is vulnerable. This includes older PyTorch versions before the stashing mechanism was introduced. The fix is `record_stream(ncclStream)`, but `ncclStream` is internal to ProcessGroupNCCL and not exposed to users.
