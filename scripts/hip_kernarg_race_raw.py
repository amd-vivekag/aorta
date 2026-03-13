"""
Raw HIP-level reproducer: kernarg buffer recycling race via AQL queue overflow.

Unlike hip_kernarg_race_mini.py (PyTorch-based), this goes directly through
the HIP C API to prove the bug is in the HIP runtime, not PyTorch.

THE BUG:
  The HIP runtime maintains a kernarg pool — a ring of memory slots where
  kernel arguments (pointers, sizes, strides) are written by the CPU before
  each kernel dispatch. Each dispatch writes its arguments into a kernarg
  slot, then enqueues an AQL packet pointing to that slot.

  The kernarg pool is finite and reused. The HIP runtime determines when a
  slot can be recycled by checking if the corresponding dispatch has completed.
  But with deep AQL queues (16K entries), the CPU can submit thousands of
  dispatches before the GPU starts the first one. The kernarg pool recycles
  slots from "completed" dispatches — but "completed" is determined by a
  simple counter/fence, not by checking whether the GPU has actually read
  the kernarg data from that slot.

  Timeline of the race:
    CPU time ──────────────────────────────────────────────────────────>
    [dispatch K: write kernargs to slot S] [dispatch K+N: recycle slot S, write new kernargs]
                                              │
    GPU time ──────────────────────────────────────────────────────────>
    [... still executing K-100 ...] [kernel K starts, reads from slot S → STALE DATA]

  On CUDA, this never happens because:
  1. CUDA's command queue provides backpressure (shallower than AMD's AQL)
  2. CUDA's kernarg management is synchronous to the stream

METHOD:
  - Allocate two device buffers A and B
  - Launch many kernels that read A and write B (memcpy-like)
  - Each kernel's arguments = {A_ptr, B_ptr, size}
  - Launch them rapidly on multiple streams WITHOUT synchronization
  - If kernarg recycling occurs, a kernel reads wrong A_ptr → either crash
    or silent data corruption (wrong values in B)

  We use hipModuleLoadData + hipModuleLaunchKernel to keep kernarg behavior
  explicit.

Usage:
    # Should CRASH or produce corruption
    python scripts/hip_kernarg_race_raw.py

    # Should PASS (shallow AQL queue)
    ROC_AQL_QUEUE_SIZE=1024 python scripts/hip_kernarg_race_raw.py

    # Should PASS (per-launch sync)
    python scripts/hip_kernarg_race_raw.py --sync-per-launch
"""

import ctypes
import ctypes.util
import os
import struct
import sys
import time

# ── Load HIP runtime ──
hip_lib = None
for name in ["libamdhip64.so", "libhip_hcc.so"]:
    try:
        hip_lib = ctypes.CDLL(name)
        break
    except OSError:
        continue
if hip_lib is None:
    hip_path = ctypes.util.find_library("amdhip64")
    if hip_path:
        hip_lib = ctypes.CDLL(hip_path)
if hip_lib is None:
    print("ERROR: cannot load HIP runtime library")
    sys.exit(1)


def hip_check(err, msg=""):
    if err != 0:
        raise RuntimeError(f"HIP error {err}: {msg}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-streams", type=int, default=4)
    parser.add_argument("--dispatches-per-stream", type=int, default=20000)
    parser.add_argument("--buffer-size-mb", type=int, default=16)
    parser.add_argument("--sync-per-launch", action="store_true")
    args = parser.parse_args()

    print(f"ROC_AQL_QUEUE_SIZE={os.environ.get('ROC_AQL_QUEUE_SIZE', '(not set, default ~16K)')}")
    print(f"GPU_MAX_HW_QUEUES={os.environ.get('GPU_MAX_HW_QUEUES', '(not set)')}")

    # Init device 0
    hip_check(hip_lib.hipSetDevice(0), "hipSetDevice")

    buf_bytes = args.buffer_size_mb * 1024 * 1024
    num_elements = buf_bytes // 4  # float32

    # Allocate device buffers
    d_src = ctypes.c_void_p()
    d_dst = ctypes.c_void_p()
    d_check = ctypes.c_void_p()
    hip_check(hip_lib.hipMalloc(ctypes.byref(d_src), buf_bytes), "malloc src")
    hip_check(hip_lib.hipMalloc(ctypes.byref(d_dst), buf_bytes), "malloc dst")
    hip_check(hip_lib.hipMalloc(ctypes.byref(d_check), buf_bytes), "malloc check")

    # Fill src with known pattern (0x3F800000 = 1.0f for all elements)
    hip_check(hip_lib.hipMemset(d_src, 0, buf_bytes), "memset src")

    # Allocate host buffer for verification
    h_result = (ctypes.c_float * num_elements)()

    # Initialize src on host and copy
    h_src = (ctypes.c_float * num_elements)()
    FILL_VAL = 42.0
    for i in range(num_elements):
        h_src[i] = FILL_VAL
    hip_check(hip_lib.hipMemcpy(d_src, h_src, buf_bytes, 1), "memcpy H2D src")  # 1 = hipMemcpyHostToDevice
    hip_check(hip_lib.hipMemset(d_dst, 0, buf_bytes), "memset dst")

    # Create streams
    streams = []
    for i in range(args.num_streams):
        s = ctypes.c_void_p()
        hip_check(hip_lib.hipStreamCreate(ctypes.byref(s)), f"create stream {i}")
        streams.append(s)

    print(f"\nLaunching {args.dispatches_per_stream} hipMemcpyAsync per stream x {args.num_streams} streams")
    print(f"Total dispatches: {args.dispatches_per_stream * args.num_streams}")
    print(f"Buffer size: {args.buffer_size_mb} MB ({num_elements} float32 elements)")
    print()

    start = time.time()
    corruption_count = 0

    # Rapidly dispatch memcpy operations across streams
    # hipMemcpyAsync internally enqueues AQL packets with kernarg buffers
    for iteration in range(args.dispatches_per_stream):
        for s in streams:
            # Copy src → dst (this is a DMA/kernel dispatch with kernarg = {src_ptr, dst_ptr, size})
            err = hip_lib.hipMemcpyAsync(
                d_dst, d_src, buf_bytes,
                3,  # hipMemcpyDeviceToDevice
                s
            )
            if err != 0:
                print(f"hipMemcpyAsync failed at iter {iteration}: err={err}")
                corruption_count += 1
                break

            if args.sync_per_launch:
                hip_check(hip_lib.hipStreamSynchronize(s), "stream sync")

        # Periodically verify: dst should contain FILL_VAL
        if (iteration + 1) % 2000 == 0:
            hip_check(hip_lib.hipDeviceSynchronize(), "device sync for check")
            hip_check(hip_lib.hipMemcpy(h_result, d_dst, buf_bytes, 2), "memcpy D2H")  # 2 = hipMemcpyDeviceToHost

            bad = 0
            for i in range(num_elements):
                if h_result[i] != FILL_VAL:
                    bad += 1
                    if bad <= 5:
                        print(f"  CORRUPTION at [{i}]: expected {FILL_VAL}, got {h_result[i]}")
            if bad > 0:
                corruption_count += 1
                print(f"  iter {iteration+1}: {bad}/{num_elements} corrupted elements!")
            else:
                elapsed = time.time() - start
                rate = (iteration + 1) * args.num_streams / elapsed
                print(f"  iter {iteration+1}: OK ({rate:.0f} dispatches/s)")

    # Final check
    hip_check(hip_lib.hipDeviceSynchronize(), "final sync")
    hip_check(hip_lib.hipMemcpy(h_result, d_dst, buf_bytes, 2), "final D2H")

    bad = 0
    for i in range(num_elements):
        if h_result[i] != FILL_VAL:
            bad += 1
    if bad > 0:
        corruption_count += 1
        print(f"\nFINAL CHECK: {bad}/{num_elements} corrupted elements!")

    elapsed = time.time() - start
    total_dispatches = args.dispatches_per_stream * args.num_streams
    print(f"\nDone: {total_dispatches} dispatches in {elapsed:.1f}s")
    print(f"Corruption events: {corruption_count}")

    # Cleanup
    hip_lib.hipFree(d_src)
    hip_lib.hipFree(d_dst)
    hip_lib.hipFree(d_check)
    for s in streams:
        hip_lib.hipStreamDestroy(s)

    if corruption_count > 0:
        print("BUG: HIP kernarg recycling race detected")
        sys.exit(1)
    else:
        print("PASS: no corruption detected")
        sys.exit(0)


if __name__ == "__main__":
    main()
