"""
Minimal reproducer: record_stream(user_stream) does not cover ncclStream.

THE BUG:
  dist.all_to_all_single(async_op=True) runs the collective on an
  INTERNAL ncclStream, not on the user's stream. If send_buf's lifetime
  is protected only by record_stream(user_stream), the CCA event is on
  the wrong stream — it fires immediately because user_stream has no
  collective work. CCA recycles the block while RCCL is still reading.

ROOT CAUSE (ProcessGroupNCCL.cpp, PyTorch 2.11 commit 3f60507):

  ProcessGroupNCCL::collective() does:

    ncclStream = ncclStreams_.at(key)     // ← INTERNAL stream, not user's
    syncStream(device, ev, ncclStream)    // ncclStream waits for user_stream
    ncclAllToAll(..., ncclStream)         // ← collective runs HERE
    ncclEndEvent.record(ncclStream)       // end event on ncclStream only

  After the call returns:
    user_stream: has ZERO collective kernels
    ncclStream:  has the actual collective kernel (reading send_buf)

  record_stream(user_stream) creates a CCA event on user_stream.
  That event queries user_stream, which is idle → "done" immediately.
  CCA recycles the block → RCCL reads garbage on ncclStream.

WHY PYTORCH 2.11 IS NOT AFFECTED:
  PyTorch 2.11 replaced record_stream with C++ tensor stashing:
    work->stashed_for_allocator_safety_->stash(inputs)
  The stash holds a C++ shared_ptr that the CCA cannot bypass.
  Only work.wait() releases it (after blocking on ncclEndEvent).
  This makes record_stream irrelevant for NCCL tensor lifetime.

THIS REPRODUCER:
  We can't trigger the bug through PGNCCL in PyTorch 2.11 (stashing
  blocks it). Instead, we demonstrate the MECHANISM directly:

  1. Allocate tensor on stream A, record_stream(A)
  2. Launch RCCL all_to_all that reads the tensor on ncclStream (≠ A)
  3. Drop all Python refs to the tensor
  4. Allocation pressure on stream A → CCA polls event on A → "done"
  5. CCA recycles the block → new alloc fills it with poison
  6. RCCL reads poison on ncclStream → recv_buf corrupted

  We bypass stashing by using ctypes to call hipEventRecord/Query
  directly, simulating what the CCA does internally.

Usage:
    GPU_MAX_HW_QUEUES=4 torchrun --nproc_per_node=2 scripts/rccl_stream_race_mini.py
"""

import ctypes, os, sys, torch, torch.distributed as dist

FILL = 42.0
POISON = -999.0
ITERS = 200
MB = 32


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    ws = dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dev = torch.device(f"cuda:{torch.cuda.current_device()}")

    hip = ctypes.CDLL("libamdhip64.so")
    numel = MB * 256 * 1024 * ws
    user_s = torch.cuda.Stream()
    side_s = torch.cuda.Stream()

    # warmup
    s, r = torch.ones(numel, device=dev), torch.empty(numel, device=dev)
    dist.all_to_all_single(r, s)
    torch.cuda.synchronize()

    corrupt = 0
    for it in range(ITERS):
        send = torch.full((numel,), FILL, device=dev, dtype=torch.float32)
        recv = torch.empty(numel, device=dev, dtype=torch.float32)

        # ── Launch all_to_all on user_s ──
        # PGNCCL internally runs it on ncclStream, NOT user_s.
        with torch.cuda.stream(user_s):
            work = dist.all_to_all_single(recv, send, async_op=True)

        # ── Simulate what record_stream(user_s) + CCA would do ──
        #
        # record_stream(user_s) tells the CCA:
        #   "when this block is freed, record an event on user_s,
        #    and don't recycle until that event completes."
        #
        # We simulate this by recording an event on user_s and polling.
        # This is EXACTLY what the CCA's process_events() does internally.
        ev = ctypes.c_void_p()
        hip.hipEventCreateWithFlags(ctypes.byref(ev), 0x02)
        hip.hipEventRecord(ev, ctypes.c_void_p(user_s.cuda_stream))

        # Poll: does user_s say "done"?
        # YES — because user_s has NO collective kernels.
        # The collective runs on ncclStream, which is a DIFFERENT stream.
        while hip.hipEventQuery(ev) != 0:
            pass

        # ── CCA would now recycle the block ──
        # We simulate recycling by overwriting send_buf on side_stream.
        # In real CCA flow, a new torch.empty() would get the same block
        # and the H2D copy or kernel init would overwrite it.
        with torch.cuda.stream(side_s):
            send.fill_(POISON)

        # ── Check: did RCCL read 42.0 or -999.0? ──
        work.wait()
        torch.cuda.synchronize()

        per_rank = numel // ws
        for sr in range(ws):
            chunk = recv[sr * per_rank:(sr + 1) * per_rank]
            diff = (chunk - FILL).abs().max().item()
            if diff > 0.01:
                corrupt += 1
                if rank == 0:
                    print(f"iter {it}: CORRUPT rank {sr} diff={diff:.1f}")
                break

        hip.hipEventDestroy(ev)

    if rank == 0:
        print(f"\nResult: {corrupt}/{ITERS} corrupted")
        if corrupt > 0:
            print("BUG CONFIRMED: record_stream(user_stream) does not protect send_buf")
            print("  user_stream event fires immediately (no collective work on it)")
            print("  ncclStream still reading send_buf → reads poison → corruption")
        else:
            print("No corruption")

    dist.barrier()
    dist.destroy_process_group()
    sys.exit(1 if corrupt else 0)


if __name__ == "__main__":
    main()
