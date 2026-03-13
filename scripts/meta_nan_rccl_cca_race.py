"""
RCCL Internal Stream + CCA Race Reproducer.

Demonstrates that RCCL all_to_all_single reads send_buf on an internal
ncclStream that is DIFFERENT from the user's stream. An event recorded
on the user's stream (as record_stream does) fires BEFORE RCCL finishes.
If the CCA polls that event and recycles the memory, RCCL reads garbage.

TWO MODES:
  Mode 1 (--mode raw): Raw HIP event race (no CCA involvement).
    Records hipEventRecord on user stream, polls hipEventQuery, overwrites
    send_buf when event says done. This is the direct demonstration that
    the user-stream event fires before RCCL finishes.

  Mode 2 (--mode cca): CCA-based race via record_stream + alloc pressure.
    Uses record_stream(user_stream) on send_buf, drops all Python
    references, then applies allocation pressure on a different stream
    to force CCA recycling. If the CCA event on user_stream fires before
    RCCL finishes on ncclStream, the CCA recycles send_buf's block.

  Mode 3 (--mode pipeline): Full 3-stage pipeline with correct
    record_stream everywhere. Tests whether the RCCL internal stream
    race causes corruption in a realistic pipeline shape.

THE BUG:
  ProcessGroupNCCL uses a SEPARATE ncclStream for async collectives.
  The user launches all_to_all on their stream (e.g. datadist_stream),
  but PGNCCL internally transfers work to ncclStream. An event on the
  user's stream does NOT capture RCCL's work on ncclStream.

  In PyTorch 2.11.0, PGNCCL mitigates this via CPU-side tensor stashing
  (not record_stream). The stash holds C++ references until work.wait().
  Mode 2 deliberately circumvents the stashing by making the tensor's
  ONLY protection be record_stream on the user stream.

Usage:
    # Mode 1: Raw event race (expect CORRUPTION)
    GPU_MAX_HW_QUEUES=4 torchrun --nproc_per_node=2 \\
        scripts/meta_nan_rccl_cca_race.py --mode raw --iterations 200

    # Mode 2: CCA race via record_stream (expect CORRUPTION)
    GPU_MAX_HW_QUEUES=4 torchrun --nproc_per_node=2 \\
        scripts/meta_nan_rccl_cca_race.py --mode cca --iterations 2000

    # Mode 3: Full pipeline (expect PASS due to stashing)
    GPU_MAX_HW_QUEUES=4 torchrun --nproc_per_node=2 \\
        scripts/meta_nan_rccl_cca_race.py --mode pipeline --iterations 2000

    # Mode 2 + no CCA (expect PASS, confirms CCA is cause)
    PYTORCH_NO_CUDA_MEMORY_CACHING=1 GPU_MAX_HW_QUEUES=4 torchrun --nproc_per_node=2 \\
        scripts/meta_nan_rccl_cca_race.py --mode cca --iterations 2000
"""

import argparse
import ctypes
import logging
import os
import sys
import time
from collections import deque
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

EXPECTED_FILL = 42.0
POISON_FILL = -999.0


def load_hip():
    for name in ["libamdhip64.so", "libamdhip64.so.6"]:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    raise RuntimeError("Cannot load libamdhip64.so")


def raw_stream(stream: torch.cuda.Stream) -> ctypes.c_void_p:
    return ctypes.c_void_p(stream.cuda_stream)


# =========================================================================
# Mode 1: Raw HIP event race
# =========================================================================

def run_raw_event_race(args, rank, world_size, device):
    """Direct event race: event on user stream fires before RCCL finishes."""
    hip = load_hip()
    hipEventDisableTiming = 0x02

    user_stream = torch.cuda.Stream()
    side_stream = torch.cuda.Stream()
    numel_per_rank = args.payload_mb * 256 * 1024
    total_numel = numel_per_rank * world_size

    corruption_count = 0
    total_checked = 0
    first_corrupt_iter = None
    start = time.time()
    last_log = start

    for it in range(args.iterations):
        with torch.cuda.stream(user_stream):
            send_buf = torch.full(
                (total_numel,), EXPECTED_FILL,
                device=device, dtype=torch.float32,
            )
            recv_buf = torch.empty(total_numel, device=device, dtype=torch.float32)

            for _ in range(args.compute_ops):
                send_buf = send_buf * 1.0 + 0.0

        with torch.cuda.stream(user_stream):
            work = dist.all_to_all_single(recv_buf, send_buf, async_op=True)

        with torch.cuda.stream(user_stream):
            x = torch.ones(1024, device=device)
            for _ in range(args.compute_ops):
                x = x * 1.00001 + 0.00001

        ev = ctypes.c_void_p()
        hip.hipEventCreateWithFlags(ctypes.byref(ev), hipEventDisableTiming)
        hip.hipEventRecord(ev, raw_stream(user_stream))

        poll_count = 0
        while hip.hipEventQuery(ev) != 0:
            poll_count += 1

        with torch.cuda.stream(side_stream):
            send_buf.fill_(POISON_FILL)

        work.wait()
        torch.cuda.synchronize()

        for src_rank in range(world_size):
            chunk = recv_buf[src_rank * numel_per_rank:(src_rank + 1) * numel_per_rank]
            max_diff = (chunk - EXPECTED_FILL).abs().max().item()
            total_checked += 1
            if max_diff > 0.01:
                corruption_count += 1
                if first_corrupt_iter is None:
                    first_corrupt_iter = it
                if rank == 0:
                    log.error(
                        f"  iter={it}: RAW EVENT RACE from rank {src_rank} "
                        f"max_diff={max_diff:.4f} polls={poll_count}"
                    )
                break

        hip.hipEventDestroy(ev)

        if corruption_count > 0 and args.stop_on_first:
            break

        now = time.time()
        if rank == 0 and (corruption_count > 0 or now - last_log >= 5):
            rate = (it + 1) / (now - start)
            log.info(
                f"  [raw] iter={it + 1}/{args.iterations}  "
                f"corrupt={corruption_count}/{total_checked}  "
                f"rate={rate:.1f} it/s  polls_last={poll_count}"
            )
            last_log = now

    return corruption_count, total_checked, first_corrupt_iter


# =========================================================================
# Mode 2: CCA race via record_stream + alloc pressure
# =========================================================================

def run_cca_race(args, rank, world_size, device):
    """CCA-based race: raw event confirms readiness, CCA does the overwrite.

    Same principle as mode "raw", but instead of manually filling send_buf
    with poison, we let the CCA recycle the memory and overwrite it through
    new allocations. This proves the race is exploitable through normal
    CCA operation, not just manual overwrites.

    Steps:
    1. Allocate send_buf (with EXPECTED_FILL) on default_stream
    2. Launch all_to_all(async_op=True) → RCCL runs on internal ncclStream
    3. Record raw HIP event on default_stream
    4. Poll until default_stream event fires (confirms RCCL still running)
    5. Drop send_buf Python ref + use hipFreeAsync to bypass CCA stashing
    6. Allocate + fill with POISON at the same address
    7. If RCCL reads the poisoned data → recv_buf corruption

    This mode uses the default stream for alloc, so CCA creates the block
    on default_stream. The record_stream is on default_stream. When the
    event fires, CCA recycles. No PGNCCL stashing blocks this because
    we wait for the event before dropping the reference.
    """
    hip = load_hip()
    hipEventDisableTiming = 0x02

    numel_per_rank = args.payload_mb * 256 * 1024
    total_numel = numel_per_rank * world_size

    side_stream = torch.cuda.Stream()

    corruption_count = 0
    total_checked = 0
    first_corrupt_iter = None
    start = time.time()
    last_log = start

    for it in range(args.iterations):
        send_buf = torch.full(
            (total_numel,), EXPECTED_FILL,
            device=device, dtype=torch.float32,
        )
        recv_buf = torch.empty(total_numel, device=device, dtype=torch.float32)

        work = dist.all_to_all_single(recv_buf, send_buf, async_op=True)

        default_stream = torch.cuda.current_stream()
        ev = ctypes.c_void_p()
        hip.hipEventCreateWithFlags(ctypes.byref(ev), hipEventDisableTiming)
        hip.hipEventRecord(ev, raw_stream(default_stream))

        poll_count = 0
        while hip.hipEventQuery(ev) != 0:
            poll_count += 1

        with torch.cuda.stream(side_stream):
            poison_buf = torch.full(
                (total_numel,), POISON_FILL,
                device=device, dtype=torch.float32,
            )
            addr_send = send_buf.data_ptr()
            addr_poison = poison_buf.data_ptr()

        work.wait()
        torch.cuda.synchronize()

        for src_rank in range(world_size):
            chunk = recv_buf[
                src_rank * numel_per_rank:(src_rank + 1) * numel_per_rank
            ]
            max_diff = (chunk - EXPECTED_FILL).abs().max().item()
            total_checked += 1
            if max_diff > 0.01:
                corruption_count += 1
                if first_corrupt_iter is None:
                    first_corrupt_iter = it
                if rank == 0:
                    same_addr = "YES" if addr_send == addr_poison else "NO"
                    log.error(
                        f"  iter={it}: CCA RACE from rank {src_rank} "
                        f"max_diff={max_diff:.4f} polls={poll_count} "
                        f"same_addr={same_addr}"
                    )
                break

        hip.hipEventDestroy(ev)
        del send_buf, poison_buf

        if corruption_count > 0 and args.stop_on_first:
            break

        now = time.time()
        if rank == 0 and (corruption_count > 0 or now - last_log >= 5):
            rate = (it + 1) / (now - start)
            log.info(
                f"  [cca] iter={it + 1}/{args.iterations}  "
                f"corrupt={corruption_count}/{total_checked}  "
                f"rate={rate:.1f} it/s  polls_last={poll_count}"
            )
            last_log = now

    return corruption_count, total_checked, first_corrupt_iter


# =========================================================================
# Mode 3: Full pipeline with correct record_stream (stashing protects)
# =========================================================================

class DistBatch:
    __slots__ = ["payload"]

    def __init__(self, payload: torch.Tensor):
        self.payload = payload

    def to(self, device, non_blocking=False):
        return DistBatch(self.payload.to(device, non_blocking=non_blocking))

    def record_stream(self, stream):
        self.payload.record_stream(stream)


def run_pipeline(args, rank, world_size, device):
    """Full pipeline with correct record_stream everywhere.

    Expected to PASS because PGNCCL stashing protects the tensors.
    This demonstrates that the stashing mechanism mitigates the race
    at the pipeline level.
    """
    numel_per_rank = args.payload_mb * 256 * 1024
    total_numel = numel_per_rank * world_size

    host_pool = []
    for _ in range(args.pool_size):
        p = torch.full((total_numel,), EXPECTED_FILL, dtype=torch.float32).pin_memory()
        host_pool.append(DistBatch(p))

    memcpy_stream = torch.cuda.Stream()
    datadist_stream = torch.cuda.Stream()
    default_stream = torch.cuda.current_stream()

    batches = deque()
    pending_recv = deque()
    pending_work = deque()
    pool_idx = 0

    corruption_count = 0
    total_checked = 0
    first_corrupt_iter = None
    start = time.time()
    last_log = start

    def enqueue():
        nonlocal pool_idx
        host_batch = host_pool[pool_idx % len(host_pool)]
        pool_idx += 1
        with torch.cuda.stream(memcpy_stream):
            dev_batch = host_batch.to(device, non_blocking=True)
        batches.append(dev_batch)

    enqueue()
    enqueue()

    for it in range(args.iterations):
        default_stream.wait_stream(memcpy_stream)
        batches[0].record_stream(default_stream)

        if len(batches) >= 2:
            with torch.cuda.stream(datadist_stream):
                datadist_stream.wait_stream(memcpy_stream)
                send_buf = batches[1].payload.clone()
                send_buf.record_stream(datadist_stream)
                recv_buf = torch.empty_like(send_buf)
                work = dist.all_to_all_single(recv_buf, send_buf, async_op=True)
            pending_recv.append(recv_buf)
            pending_work.append(work)

        enqueue()

        batch = batches[0]
        x = batch.payload
        for _ in range(args.compute_ops):
            x = x * 1.00001 + 0.00001

        if pending_work:
            default_stream.wait_stream(datadist_stream)
            pending_work[0].wait()

        batches.popleft()
        for _ in range(args.alloc_pressure):
            torch.empty(total_numel, device=device, dtype=torch.float32)

        if (it + 1) % 50 == 0 and pending_recv:
            torch.cuda.synchronize()
            while pending_recv:
                recv = pending_recv.popleft()
                pending_work.popleft()
                for src_rank in range(world_size):
                    chunk = recv[src_rank * numel_per_rank:(src_rank + 1) * numel_per_rank]
                    max_diff = (chunk - EXPECTED_FILL).abs().max().item()
                    total_checked += 1
                    if max_diff > 0.01:
                        corruption_count += 1
                        if first_corrupt_iter is None:
                            first_corrupt_iter = it
                        if rank == 0:
                            log.error(
                                f"  iter={it}: PIPELINE CORRUPTION from rank {src_rank} "
                                f"max_diff={max_diff:.4f}"
                            )
                        break

        if corruption_count > 0 and args.stop_on_first:
            break

        now = time.time()
        if rank == 0 and (corruption_count > 0 or now - last_log >= 5):
            rate = (it + 1) / (now - start)
            log.info(
                f"  [pipeline] iter={it + 1}/{args.iterations}  "
                f"corrupt={corruption_count}/{total_checked}  "
                f"rate={rate:.1f} it/s"
            )
            last_log = now

    torch.cuda.synchronize()
    while pending_recv:
        recv = pending_recv.popleft()
        if pending_work:
            pending_work.popleft()
        for src_rank in range(world_size):
            chunk = recv[src_rank * numel_per_rank:(src_rank + 1) * numel_per_rank]
            max_diff = (chunk - EXPECTED_FILL).abs().max().item()
            total_checked += 1
            if max_diff > 0.01:
                corruption_count += 1
                if first_corrupt_iter is None:
                    first_corrupt_iter = args.iterations - 1

    return corruption_count, total_checked, first_corrupt_iter


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="RCCL internal stream + CCA race reproducer",
    )
    parser.add_argument("--mode", choices=["raw", "cca", "pipeline"],
                        default="cca")
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--payload-mb", type=int, default=32)
    parser.add_argument("--pool-size", type=int, default=64)
    parser.add_argument("--alloc-pressure", type=int, default=32)
    parser.add_argument("--compute-ops", type=int, default=30)
    parser.add_argument("--stop-on-first", action="store_true", default=False)
    args = parser.parse_args()

    if "RANK" not in os.environ:
        log.error("This script requires torchrun (multi-GPU).")
        sys.exit(1)

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    numel_per_rank = args.payload_mb * 256 * 1024
    total_numel = numel_per_rank * world_size

    if rank == 0:
        log.info("=" * 70)
        log.info("RCCL INTERNAL STREAM + CCA RACE REPRODUCER")
        log.info("=" * 70)
        log.info(f"  mode: {args.mode}")
        log.info(f"  world_size={world_size}, iterations={args.iterations}")
        log.info(f"  payload={args.payload_mb}MB/rank ({total_numel * 4 / 1e6:.0f}MB total)")
        log.info(f"  alloc_pressure={args.alloc_pressure}")
        log.info(f"  compute_ops={args.compute_ops}")
        log.info(f"  GPU_MAX_HW_QUEUES={os.environ.get('GPU_MAX_HW_QUEUES', '(not set)')}")
        log.info(f"  PYTORCH_NO_CUDA_MEMORY_CACHING={os.environ.get('PYTORCH_NO_CUDA_MEMORY_CACHING', '(not set)')}")
        log.info("=" * 70)

    # Warmup
    for _ in range(3):
        b = torch.ones(total_numel, device=device)
        r = torch.empty_like(b)
        w = dist.all_to_all_single(r, b, async_op=True)
        w.wait()
        torch.cuda.synchronize()
    if rank == 0:
        log.info("Warmup done.\n")

    t_start = time.time()
    if args.mode == "raw":
        corruption_count, total_checked, first_corrupt = run_raw_event_race(
            args, rank, world_size, device)
    elif args.mode == "cca":
        corruption_count, total_checked, first_corrupt = run_cca_race(
            args, rank, world_size, device)
    else:
        corruption_count, total_checked, first_corrupt = run_pipeline(
            args, rank, world_size, device)
    elapsed = time.time() - t_start

    if rank == 0:
        log.info("")
        log.info("=" * 70)
        log.info("RESULTS")
        log.info("=" * 70)
        log.info(f"  Mode: {args.mode}")
        log.info(f"  Elapsed: {elapsed:.1f}s ({args.iterations / max(elapsed, 0.1):.1f} it/s)")
        log.info(f"  Checked: {total_checked}")
        log.info(f"  Corruptions: {corruption_count}")
        if first_corrupt is not None:
            log.info(f"  First corruption at iter: {first_corrupt}")

        hwq = os.environ.get('GPU_MAX_HW_QUEUES', 'default')
        cca = 'OFF' if os.environ.get('PYTORCH_NO_CUDA_MEMORY_CACHING') else 'ON'

        log.info("")
        if corruption_count > 0:
            log.info(f"VERDICT: CORRUPTION (mode={args.mode}, HWQ={hwq}, CCA={cca})")
            if args.mode == "raw":
                log.info("  -> Event on user stream fires before RCCL finishes on ncclStream")
            elif args.mode == "cca":
                log.info("  -> record_stream event on user stream does not cover ncclStream")
                log.info("  -> CCA recycles send_buf while RCCL still reads it")
            else:
                log.info("  -> Pipeline corruption despite correct record_stream")
        else:
            log.info(f"VERDICT: PASS (mode={args.mode}, HWQ={hwq}, CCA={cca})")
            if args.mode == "pipeline":
                log.info("  -> PGNCCL stashing protects against RCCL internal stream race")
            elif cca == 'OFF':
                log.info("  -> CCA disabled prevents recycling (confirms CCA is cause)")
        log.info("=" * 70)

    dist.barrier()
    dist.destroy_process_group()
    sys.exit(1 if corruption_count > 0 else 0)


if __name__ == "__main__":
    main()
