"""
Pure HIP-level test: does hipEventQuery have cross-stream visibility issues
when RCCL is called directly (bypassing ProcessGroupNCCL)?

FINDING:
  When ncclAllToAll is called directly via the RCCL C API with user_stream,
  RCCL launches the collective kernel ON user_stream. The hipEvent recorded
  on user_stream correctly tracks the collective's progress (~3700 polls
  before completion). No corruption occurs.

  The "internal stream" that causes the RCCL racing issue in our other
  reproducer (rccl_stream_race_mini.py) is NOT created by RCCL itself —
  it is created by ProcessGroupNCCL (PyTorch's C++ wrapper). PGNCCL
  creates its own ncclStreams_ and passes those to RCCL instead of the
  user's stream.

  Therefore: the cross-stream visibility issue is a ProcessGroupNCCL
  design issue, not a HIP or RCCL bug. hipEventQuery correctly tracks
  progress on the stream it was recorded on.

THIS REPRODUCER:
  Uses raw HIP C API (ctypes) + RCCL C API (ctypes) for all GPU work.
  (torch.distributed gloo is used only to broadcast the RCCL unique ID.)

  1. hipMalloc send_buf, fill with 42.0 on user_stream
  2. ncclAllToAll(send_buf, recv_buf, ..., user_stream)
     → RCCL runs the collective ON user_stream (verified by event polls)
  3. hipEventRecord(ev, user_stream)
  4. Poll hipEventQuery(ev) → blocks for ~3700 polls (collective is running)
  5. hipMemsetAsync(send_buf, 0xFF, ..., side_stream) — overwrite
  6. hipStreamSynchronize on all streams
  7. Check recv_buf: contains correct data (RCCL finished before overwrite)

  Result: 0/200 corrupted. hipEventQuery works correctly.

Usage:
  GPU_MAX_HW_QUEUES=4 torchrun --nproc_per_node=2 scripts/hip_event_cross_stream.py
"""

import ctypes
import ctypes.util
import os
import struct
import sys
import time

hip = ctypes.CDLL("libamdhip64.so")
rccl = ctypes.CDLL("librccl.so")

FILL_VAL = 42.0
POISON_BYTE = 0xFF
ITERS = 200
COUNT = 4 * 1024 * 1024


def hip_check(err, msg=""):
    if err != 0:
        raise RuntimeError(f"HIP error {err}: {msg}")


def rccl_check(err, msg=""):
    if err != 0:
        raise RuntimeError(f"RCCL error {err}: {msg}")


class NcclUniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_char * 128)]


def main():
    rank = int(os.environ.get("RANK", os.environ.get("OMPI_COMM_WORLD_RANK", "0")))
    world = int(os.environ.get("WORLD_SIZE", os.environ.get("OMPI_COMM_WORLD_SIZE", "2")))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", str(rank))))

    hip_check(hip.hipSetDevice(local_rank), "hipSetDevice")

    if rank == 0:
        print(f"Pure HIP + RCCL test (no PyTorch tensors)")
        print(f"GPU_MAX_HW_QUEUES={os.environ.get('GPU_MAX_HW_QUEUES', '(not set)')}")
        print(f"world={world}, count={COUNT} float32 ({COUNT*4/(1024*1024):.0f} MB/rank)")
        print()

    import torch
    import torch.distributed as dist
    dist.init_process_group(backend="gloo")

    uid = NcclUniqueId()
    if rank == 0:
        rccl_check(rccl.ncclGetUniqueId(ctypes.byref(uid)), "ncclGetUniqueId")
    uid_bytes = bytes(ctypes.string_at(ctypes.byref(uid), 128))
    uid_tensor = torch.tensor(list(uid_bytes), dtype=torch.uint8)
    dist.broadcast(uid_tensor, src=0)
    ctypes.memmove(ctypes.byref(uid), bytes(uid_tensor.tolist()), 128)

    comm = ctypes.c_void_p()
    rccl_check(rccl.ncclCommInitRank(ctypes.byref(comm), world, uid, rank), "ncclCommInitRank")

    buf_bytes = COUNT * 4
    total_bytes = buf_bytes * world

    d_send = ctypes.c_void_p()
    d_recv = ctypes.c_void_p()
    hip_check(hip.hipMalloc(ctypes.byref(d_send), total_bytes), "hipMalloc send")
    hip_check(hip.hipMalloc(ctypes.byref(d_recv), total_bytes), "hipMalloc recv")

    h_recv = (ctypes.c_float * (COUNT * world))()
    h_fill = (ctypes.c_float * (COUNT * world))()
    for i in range(COUNT * world):
        h_fill[i] = FILL_VAL

    user_stream = ctypes.c_void_p()
    side_stream = ctypes.c_void_p()
    hip_check(hip.hipStreamCreate(ctypes.byref(user_stream)), "create user_stream")
    hip_check(hip.hipStreamCreate(ctypes.byref(side_stream)), "create side_stream")

    # Warmup
    hip_check(hip.hipMemcpy(d_send, h_fill, total_bytes, 1), "warmup H2D")
    rccl_check(rccl.ncclAllToAll(d_send, d_recv, COUNT, 7, comm, user_stream), "warmup alltoall")
    hip_check(hip.hipStreamSynchronize(user_stream), "warmup sync")

    corrupt = 0
    event_immediate = 0
    total_polls = 0

    for it in range(ITERS):
        hip_check(hip.hipMemcpyAsync(d_send, h_fill, total_bytes, 1, user_stream), "fill H2D")
        hip_check(hip.hipStreamSynchronize(user_stream), "sync fill")

        rccl_check(
            rccl.ncclAllToAll(d_send, d_recv, COUNT, 7, comm, user_stream),
            f"alltoall iter {it}"
        )

        ev = ctypes.c_void_p()
        hip_check(hip.hipEventCreateWithFlags(ctypes.byref(ev), 0x02), "event create")
        hip_check(hip.hipEventRecord(ev, user_stream), "event record")

        poll_count = 0
        while hip.hipEventQuery(ev) != 0:
            poll_count += 1
        if poll_count == 0:
            event_immediate += 1
        total_polls += poll_count

        hip_check(hip.hipMemsetAsync(d_send, POISON_BYTE, total_bytes, side_stream), "poison send")

        hip_check(hip.hipStreamSynchronize(user_stream), "sync user")
        hip_check(hip.hipStreamSynchronize(side_stream), "sync side")

        hip_check(hip.hipMemcpy(h_recv, d_recv, total_bytes, 2), "D2H recv")

        bad = False
        for i in range(min(COUNT * world, 1000)):
            if abs(h_recv[i] - FILL_VAL) > 0.01:
                bad = True
                break

        if bad:
            corrupt += 1
            if rank == 0 and corrupt <= 10:
                sample_val = h_recv[0]
                print(f"iter {it}: CORRUPT  got={sample_val:.6g}  expected={FILL_VAL}")

        hip_check(hip.hipEventDestroy(ev), "event destroy")

    if rank == 0:
        avg_polls = total_polls / ITERS if ITERS > 0 else 0
        print(f"\n{'='*60}")
        print(f"Result: {corrupt}/{ITERS} corrupted")
        print(f"Event fired immediately (0 polls): {event_immediate}/{ITERS}")
        print(f"Average polls before event done: {avg_polls:.0f}")
        print()
        if corrupt > 0:
            print("CORRUPTION DETECTED")
        else:
            print("PASS: no corruption.")
            print("  RCCL C API launches the collective ON user_stream.")
            print("  hipEventQuery correctly waits for the collective.")
            print("  The internal stream is a ProcessGroupNCCL (PyTorch)")
            print("  design, not a HIP or RCCL issue.")
        print(f"{'='*60}")

    rccl.ncclCommDestroy(comm)
    hip.hipFree(d_send)
    hip.hipFree(d_recv)
    hip.hipStreamDestroy(user_stream)
    hip.hipStreamDestroy(side_stream)
    dist.destroy_process_group()

    sys.exit(1 if corrupt > 0 else 0)


if __name__ == "__main__":
    main()
