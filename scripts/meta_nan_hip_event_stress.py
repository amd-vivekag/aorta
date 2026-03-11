"""
Aggressive HIP event stress test (v4).

Key insight: previous tests used regular compute kernels. Meta's bug
involves RCCL collectives (all_to_all) which run on RCCL's internal
streams and have different event/signal interactions. We also need
to simulate the FULL TorchRec pipeline pattern: 3 streams, RCCL,
H2D, forward pass, CCA recycling -- all overlapping simultaneously.

This test creates the heaviest, most realistic load we can:

1. Many streams (8-16) with independent work
2. RCCL collectives (all_to_all, all_reduce) on some streams
3. Large matmuls on other streams (simulate forward pass)
4. H2D copies from pinned memory on copy streams
5. Cross-stream tensor sharing with record_stream
6. CCA allocation pressure (force process_events polling)
7. Raw HIP event polling in parallel with all of the above
8. All happening concurrently to maximize HW queue contention

Also tests:
- DMA engine vs compute engine contention
- Event recorded on stream with RCCL work
- Event recorded on stream right after hipMemcpyAsync
- Multiple events from different streams polled in tight interleave
"""

import argparse
import ctypes
import logging
import os
import sys
import time
import threading
from typing import List, Tuple

import torch
import torch.distributed as dist

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

FILL_A = 42.0
FILL_B = -999.0


def load_hip():
    try:
        hip = ctypes.CDLL("libamdhip64.so")
    except OSError:
        hip = ctypes.CDLL("libamdhip64.so.6")
    for fn_name, argtypes, restype in [
        ("hipEventCreateWithFlags", [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint], ctypes.c_int),
        ("hipEventRecord", [ctypes.c_void_p, ctypes.c_void_p], ctypes.c_int),
        ("hipEventQuery", [ctypes.c_void_p], ctypes.c_int),
        ("hipEventSynchronize", [ctypes.c_void_p], ctypes.c_int),
        ("hipEventDestroy", [ctypes.c_void_p], ctypes.c_int),
    ]:
        fn = getattr(hip, fn_name)
        fn.argtypes = argtypes
        fn.restype = restype
    return hip


def raw_stream(s):
    return ctypes.c_void_p(s.cuda_stream)


# -----------------------------------------------------------------------
# Test 1: Full pipeline simulation with RCCL + H2D + compute
# -----------------------------------------------------------------------

def run_full_pipeline(args, rank, device):
    """Simulate TorchRec's 3-stage pipeline with RCCL and heavy compute.

    3 streams:
      memcpy_stream:  H2D from pinned memory
      datadist_stream: RCCL all_to_all_single
      default_stream:  large matmul chain (forward pass)

    Each iteration:
      1. H2D on memcpy_stream (pinned -> device)
      2. datadist (RCCL all_to_all) on datadist_stream
      3. Forward (matmul chain) on default_stream
      4. record_stream on tensors that cross streams
      5. del old tensors -> CCA puts in pending
      6. Allocate new tensors -> CCA calls process_events

    We verify that no CCA-recycled block corrupts RCCL or compute results.
    """
    if not dist.is_initialized():
        log.warning("Skipping full_pipeline (requires torchrun)")
        return 0, 0, []

    world_size = dist.get_world_size()
    default_stream = torch.cuda.current_stream()
    datadist_stream = torch.cuda.Stream()
    memcpy_stream = torch.cuda.Stream()

    numel = args.size_mb * 256 * 1024
    total_numel = numel * world_size
    dim = min(int(numel ** 0.5), 4096)
    dim = max(dim, 1024)

    host_pool = [
        torch.full((total_numel,), FILL_A * (i + 1), dtype=torch.float32).pin_memory()
        for i in range(32)
    ]

    corruption_count = 0
    ptr_reuse_count = 0
    details = []
    start = time.time()
    last_log = start

    prev_send = None
    prev_compute_in = None

    for it in range(args.iterations):
        host_data = host_pool[it % len(host_pool)]
        expected_fill = FILL_A * ((it % 32) + 1)

        # Stage 1: H2D on memcpy_stream
        with torch.cuda.stream(memcpy_stream):
            device_buf = torch.empty(total_numel, device=device, dtype=torch.float32)
            device_buf.copy_(host_data, non_blocking=True)

        # Stage 2: RCCL all_to_all on datadist_stream
        datadist_stream.wait_stream(memcpy_stream)
        with torch.cuda.stream(datadist_stream):
            send_buf = device_buf.clone()
            recv_buf = torch.empty_like(send_buf)
            a2a_work = dist.all_to_all_single(recv_buf, send_buf, async_op=True)

        # Stage 3: Forward pass (matmul chain) on default_stream
        default_stream.wait_stream(memcpy_stream)
        with torch.cuda.stream(default_stream):
            mat = device_buf[:dim * dim].view(dim, dim)
            compute_out = mat
            for _ in range(args.chain_ops // 10 + 1):
                compute_out = torch.mm(compute_out, compute_out.t())
            compute_result = compute_out.diagonal().sum()

        # Record cross-stream usage for CCA
        device_buf.record_stream(datadist_stream)
        device_buf.record_stream(default_stream)
        send_buf.record_stream(datadist_stream)

        # Free previous iteration's tensors -> CCA pending
        if prev_send is not None:
            del prev_send
        if prev_compute_in is not None:
            del prev_compute_in

        # Allocation pressure on all streams to trigger process_events
        pressure_tensors = []
        for s in [memcpy_stream, datadist_stream, default_stream]:
            with torch.cuda.stream(s):
                for _ in range(args.alloc_pressure // 3 + 1):
                    p = torch.empty(total_numel, device=device, dtype=torch.float32)
                    pressure_tensors.append(p)

        # Wait for RCCL
        a2a_work.wait()
        torch.cuda.synchronize()

        # Verify RCCL result
        for src_rank in range(world_size):
            chunk = recv_buf[src_rank * numel:(src_rank + 1) * numel]
            max_diff = (chunk - expected_fill).abs().max().item()
            if max_diff > 0.01:
                corruption_count += 1
                details.append(
                    f"iter={it}: RCCL CORRUPTION from rank {src_rank} "
                    f"max_diff={max_diff:.4f} expected={expected_fill:.1f}"
                )
                break

        # Check compute for NaN/Inf
        cv = compute_result.item()
        if cv != cv or abs(cv) == float('inf'):
            corruption_count += 1
            details.append(f"iter={it}: COMPUTE NaN/Inf val={cv}")

        if corruption_count > 0 and args.stop_on_first:
            break

        prev_send = send_buf
        prev_compute_in = device_buf
        del recv_buf, pressure_tensors, compute_result

        now = time.time()
        if rank == 0 and (corruption_count > 0 or now - last_log >= 5):
            log.info(
                f"  [full_pipeline] iter={it+1}/{args.iterations}  "
                f"corrupt={corruption_count}"
            )
            last_log = now

    return corruption_count, ptr_reuse_count, details


# -----------------------------------------------------------------------
# Test 2: Massive stream fan-out with raw events
# -----------------------------------------------------------------------

def run_stream_fanout(args, rank, device):
    """Create 16 streams, each doing independent heavy work.

    Record raw HIP events on all 16 streams. Poll them in a round-robin
    tight loop. When an event reports done, immediately overwrite the
    source buffer on a DIFFERENT stream and verify the result.

    This creates maximum cross-HW-queue contention with 16 concurrent
    streams competing for 4 HW queues.
    """
    hip = load_hip()
    N = 16
    streams = [torch.cuda.Stream() for _ in range(N)]
    raw_streams = [raw_stream(s) for s in streams]

    hipEventDisableTiming = 0x02
    numel = args.size_mb * 256 * 1024

    corruption_count = 0
    details = []
    start = time.time()
    last_log = start

    for it in range(args.iterations):
        bufs = []
        results = []
        events = []

        for i in range(N):
            with torch.cuda.stream(streams[i]):
                buf = torch.full((numel,), FILL_A * (i + 1),
                                 device=device, dtype=torch.float32)
            bufs.append(buf)

            if i > 0:
                streams[i].wait_stream(streams[i - 1])
            with torch.cuda.stream(streams[i]):
                x = buf
                for _ in range(args.chain_ops):
                    x = x * 1.00001 + 0.00001
                results.append(x.sum().unsqueeze(0))

            ev = ctypes.c_void_p()
            hip.hipEventCreateWithFlags(ctypes.byref(ev), hipEventDisableTiming)
            hip.hipEventRecord(ev, raw_streams[i])
            events.append(ev)

        # Poll all events in tight round-robin
        completed = [False] * N
        overwritten = [False] * N
        while not all(completed):
            for i in range(N):
                if completed[i]:
                    continue
                if hip.hipEventQuery(events[i]) == 0:
                    completed[i] = True
                    # Overwrite on a DIFFERENT stream
                    other = (i + N // 2) % N
                    with torch.cuda.stream(streams[other]):
                        bufs[i].fill_(FILL_B)
                    overwritten[i] = True

        torch.cuda.synchronize()

        for i in range(N):
            expected = float(numel) * FILL_A * (i + 1)
            for _ in range(args.chain_ops):
                expected = expected * 1.00001 + 0.00001 * numel
            actual = results[i].item()
            rel_err = abs(actual - expected) / max(abs(expected), 1.0)
            if rel_err > 1e-3:
                corruption_count += 1
                details.append(
                    f"iter={it} stream={i}: CORRUPTION rel_err={rel_err:.6f} "
                    f"actual={actual:.1f} expected={expected:.1f}"
                )
            hip.hipEventDestroy(events[i])

        if corruption_count > 0 and args.stop_on_first:
            break
        del bufs, results, events

        now = time.time()
        if rank == 0 and (corruption_count > 0 or now - last_log >= 5):
            log.info(
                f"  [stream_fanout] iter={it+1}/{args.iterations}  "
                f"corrupt={corruption_count}"
            )
            last_log = now

    return corruption_count, 0, details


# -----------------------------------------------------------------------
# Test 3: H2D + compute race with raw events
# -----------------------------------------------------------------------

def run_h2d_compute_race(args, rank, device):
    """Race between H2D DMA engine and compute engine.

    The DMA engine (SDMA) and compute engine are independent hardware
    units. Events may complete differently depending on which engine
    the work ran on.

    Pattern:
      1. H2D copy of large buffer on stream_copy (uses DMA engine)
      2. Matmul on stream_compute reading from the SAME buffer
      3. Record event on stream_compute
      4. Poll event, then overwrite buffer via another H2D
      5. Check matmul result
    """
    hip = load_hip()
    hipEventDisableTiming = 0x02

    stream_copy = torch.cuda.Stream()
    stream_compute = torch.cuda.Stream()
    stream_overwrite = torch.cuda.Stream()

    dim = min(int((args.size_mb * 256 * 1024) ** 0.5), 4096)
    dim = max(dim, 2048)
    numel = dim * dim

    host_a = torch.full((numel,), FILL_A, dtype=torch.float32).pin_memory()
    host_b = torch.full((numel,), FILL_B, dtype=torch.float32).pin_memory()

    corruption_count = 0
    details = []
    start = time.time()
    last_log = start

    for it in range(args.iterations):
        with torch.cuda.stream(stream_copy):
            buf = torch.empty(numel, device=device, dtype=torch.float32)
            buf.copy_(host_a, non_blocking=True)

        stream_compute.wait_stream(stream_copy)
        with torch.cuda.stream(stream_compute):
            mat = buf.view(dim, dim)
            for _ in range(args.chain_ops // 10 + 1):
                result_mat = torch.mm(mat, mat.t())
            result_val = result_mat.diagonal().sum().unsqueeze(0)

        ev = ctypes.c_void_p()
        hip.hipEventCreateWithFlags(ctypes.byref(ev), hipEventDisableTiming)
        hip.hipEventRecord(ev, raw_stream(stream_compute))

        while hip.hipEventQuery(ev) != 0:
            pass

        # Event says done -- immediately DMA overwrite
        with torch.cuda.stream(stream_overwrite):
            buf.copy_(host_b, non_blocking=True)

        torch.cuda.synchronize()

        val = result_val.item()
        if val != val or abs(val) == float('inf'):
            corruption_count += 1
            details.append(f"iter={it}: NaN/Inf after H2D race val={val}")
        elif val < 0:
            corruption_count += 1
            details.append(f"iter={it}: NEGATIVE after H2D race val={val:.1f}")

        hip.hipEventDestroy(ev)
        del buf, result_val

        now = time.time()
        if rank == 0 and (corruption_count > 0 or now - last_log >= 5):
            log.info(
                f"  [h2d_compute_race] iter={it+1}/{args.iterations}  "
                f"corrupt={corruption_count}"
            )
            last_log = now

    return corruption_count, 0, details


# -----------------------------------------------------------------------
# Test 4: RCCL event race (event on RCCL work stream)
# -----------------------------------------------------------------------

def run_rccl_event_race(args, rank, device):
    """Test event correctness specifically on RCCL operations.

    RCCL internally uses its own streams and kernel launch patterns.
    We test:
      1. Launch async RCCL all_to_all on default_stream
      2. Record raw HIP event on default_stream
      3. Poll the event
      4. When done, overwrite the send buffer on a side stream
      5. Wait for RCCL and check recv buffer

    If hipEventQuery returns premature for the RCCL kernel's stream,
    the recv buffer will contain corrupted data.
    """
    if not dist.is_initialized():
        log.warning("Skipping rccl_event_race (requires torchrun)")
        return 0, 0, []

    hip = load_hip()
    hipEventDisableTiming = 0x02

    world_size = dist.get_world_size()
    default_stream = torch.cuda.current_stream()
    side_stream = torch.cuda.Stream()

    numel_per_rank = args.size_mb * 256 * 1024
    total_numel = numel_per_rank * world_size

    corruption_count = 0
    details = []
    start = time.time()
    last_log = start

    for it in range(args.iterations):
        send_buf = torch.full(
            (total_numel,), float(rank + 1) * FILL_A,
            device=device, dtype=torch.float32,
        )
        recv_buf = torch.empty(total_numel, device=device, dtype=torch.float32)
        expected_fill = float(rank + 1) * FILL_A

        # Launch async RCCL all_to_all
        work = dist.all_to_all_single(recv_buf, send_buf, async_op=True)

        # Also add heavy compute after the RCCL op on default_stream
        with torch.cuda.stream(default_stream):
            x = send_buf
            for _ in range(args.chain_ops):
                x = x * 1.00001 + 0.00001
            compute_check = x.sum().unsqueeze(0)

        # Record event on default_stream (after RCCL + compute)
        ev = ctypes.c_void_p()
        hip.hipEventCreateWithFlags(ctypes.byref(ev), hipEventDisableTiming)
        hip.hipEventRecord(ev, raw_stream(default_stream))

        # Poll event
        poll_count = 0
        while hip.hipEventQuery(ev) != 0:
            poll_count += 1

        # Event says default_stream is done -- overwrite send_buf on side_stream
        with torch.cuda.stream(side_stream):
            send_buf.fill_(FILL_B)

        # Wait for RCCL and sync
        work.wait()
        torch.cuda.synchronize()

        # Verify RCCL recv
        for src_rank in range(world_size):
            chunk = recv_buf[src_rank * numel_per_rank:(src_rank + 1) * numel_per_rank]
            expected_val = float(src_rank + 1) * FILL_A
            max_diff = (chunk - expected_val).abs().max().item()
            if max_diff > 0.01:
                corruption_count += 1
                details.append(
                    f"iter={it}: RCCL EVENT RACE from rank {src_rank} "
                    f"max_diff={max_diff:.4f} polls={poll_count}"
                )
                break

        # Verify compute
        expected_sum = float(total_numel) * expected_fill
        for _ in range(args.chain_ops):
            expected_sum = expected_sum * 1.00001 + 0.00001 * total_numel
        actual_sum = compute_check.item()
        rel_err = abs(actual_sum - expected_sum) / max(abs(expected_sum), 1.0)
        if rel_err > 1e-3:
            corruption_count += 1
            details.append(
                f"iter={it}: COMPUTE after RCCL rel_err={rel_err:.6f}"
            )

        if corruption_count > 0 and args.stop_on_first:
            break

        hip.hipEventDestroy(ev)
        del send_buf, recv_buf, compute_check

        now = time.time()
        if rank == 0 and (corruption_count > 0 or now - last_log >= 5):
            log.info(
                f"  [rccl_event_race] iter={it+1}/{args.iterations}  "
                f"corrupt={corruption_count}  polls_last={poll_count}"
            )
            last_log = now

    return corruption_count, 0, details


# -----------------------------------------------------------------------
# Test 5: Concurrent event polling from background thread
# -----------------------------------------------------------------------

def run_threaded_poll(args, rank, device):
    """Poll events from a background thread while GPU work continues.

    In Meta's real workload, the CPU thread is constantly submitting new
    work while the CCA is polling events in process_events(). This test
    simulates that concurrency:

    - Main thread: continuously launches heavy GPU work
    - Background thread: polls raw HIP events and overwrites buffers
      the instant they report done
    """
    hip = load_hip()
    hipEventDisableTiming = 0x02
    numel = args.size_mb * 256 * 1024

    streams = [torch.cuda.Stream() for _ in range(8)]
    corruption_count = 0
    details = []
    lock = threading.Lock()

    class PollItem:
        __slots__ = ['event', 'buf', 'overwrite_stream_idx', 'result', 'expected', 'iteration', 'stream_idx']

    pending = []
    verified = []
    done_flag = threading.Event()

    def poller_thread():
        """Background thread: poll events and overwrite buffers."""
        while not done_flag.is_set() or pending:
            to_remove = []
            with lock:
                items = list(pending)
            for idx, item in enumerate(items):
                if hip.hipEventQuery(item.event) == 0:
                    with torch.cuda.stream(streams[item.overwrite_stream_idx]):
                        item.buf.fill_(FILL_B)
                    to_remove.append(idx)
            if to_remove:
                with lock:
                    for idx in reversed(to_remove):
                        if idx < len(pending):
                            verified.append(pending.pop(idx))
            else:
                time.sleep(0.0001)

    poller = threading.Thread(target=poller_thread, daemon=True)
    poller.start()

    start = time.time()
    last_log = start

    for it in range(args.iterations):
        si = it % len(streams)
        oi = (si + len(streams) // 2) % len(streams)

        with torch.cuda.stream(streams[si]):
            buf = torch.full((numel,), FILL_A * (si + 1),
                             device=device, dtype=torch.float32)
            x = buf
            for _ in range(args.chain_ops):
                x = x * 1.00001 + 0.00001
            result = x.sum().unsqueeze(0)

        ev = ctypes.c_void_p()
        hip.hipEventCreateWithFlags(ctypes.byref(ev), hipEventDisableTiming)
        hip.hipEventRecord(ev, raw_stream(streams[si]))

        item = PollItem()
        item.event = ev
        item.buf = buf
        item.overwrite_stream_idx = oi
        item.result = result
        item.expected = FILL_A * (si + 1)
        item.iteration = it
        item.stream_idx = si

        with lock:
            pending.append(item)

        # Periodically sync and verify completed items
        if it % 100 == 99 or it == args.iterations - 1:
            torch.cuda.synchronize()
            with lock:
                to_verify = list(verified)
                verified.clear()
            for item in to_verify:
                exp = float(numel) * item.expected
                for _ in range(args.chain_ops):
                    exp = exp * 1.00001 + 0.00001 * numel
                actual = item.result.item()
                rel_err = abs(actual - exp) / max(abs(exp), 1.0)
                if rel_err > 1e-3:
                    corruption_count += 1
                    details.append(
                        f"iter={item.iteration} stream={item.stream_idx}: "
                        f"THREADED CORRUPTION rel_err={rel_err:.6f} "
                        f"actual={actual:.1f} expected={exp:.1f}"
                    )
                hip.hipEventDestroy(item.event)
                del item.buf, item.result

            if corruption_count > 0 and args.stop_on_first:
                break

        now = time.time()
        if rank == 0 and (corruption_count > 0 or now - last_log >= 5):
            with lock:
                plen = len(pending)
            log.info(
                f"  [threaded_poll] iter={it+1}/{args.iterations}  "
                f"corrupt={corruption_count}  pending={plen}"
            )
            last_log = now

    done_flag.set()
    torch.cuda.synchronize()
    poller.join(timeout=10)

    # Final verification of any remaining items
    with lock:
        remaining = list(verified) + list(pending)
        verified.clear()
        pending.clear()
    for item in remaining:
        torch.cuda.synchronize()
        exp = float(numel) * item.expected
        for _ in range(args.chain_ops):
            exp = exp * 1.00001 + 0.00001 * numel
        actual = item.result.item()
        rel_err = abs(actual - exp) / max(abs(exp), 1.0)
        if rel_err > 1e-3:
            corruption_count += 1
            details.append(
                f"iter={item.iteration} stream={item.stream_idx}: "
                f"FINAL CORRUPTION rel_err={rel_err:.6f}"
            )
        hip.hipEventDestroy(item.event)

    return corruption_count, 0, details


def main():
    parser = argparse.ArgumentParser(
        description="Aggressive HIP event stress test (v4)"
    )
    parser.add_argument(
        "--mode",
        choices=[
            "full_pipeline", "stream_fanout", "h2d_compute_race",
            "rccl_event_race", "threaded_poll", "all",
        ],
        default="all",
    )
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--size-mb", type=int, default=64)
    parser.add_argument("--chain-ops", type=int, default=50)
    parser.add_argument("--alloc-pressure", type=int, default=16)
    parser.add_argument("--stop-on-first", action="store_true", default=True)
    parser.add_argument("--no-stop-on-first", dest="stop_on_first", action="store_false")
    args = parser.parse_args()

    os.environ.setdefault("ROC_AQL_QUEUE_SIZE", "1024")

    distributed = False
    rank = 0
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        distributed = True
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        rank = dist.get_rank()

    device = torch.cuda.current_device()

    single_gpu = ["stream_fanout", "h2d_compute_race", "threaded_poll"]
    multi_gpu = ["full_pipeline", "rccl_event_race"]

    if args.mode == "all":
        modes = list(single_gpu)
        if distributed:
            modes.extend(multi_gpu)
    else:
        modes = [args.mode]

    if rank == 0:
        log.info("=" * 70)
        log.info("AGGRESSIVE HIP EVENT STRESS TEST (v4)")
        log.info("=" * 70)
        log.info(f"  modes: {modes}")
        log.info(f"  iterations: {args.iterations}")
        log.info(f"  size_mb: {args.size_mb}")
        log.info(f"  chain_ops: {args.chain_ops}")
        ws = dist.get_world_size() if distributed else 1
        log.info(f"  world_size: {ws}")
        log.info(f"  GPU_MAX_HW_QUEUES: {os.environ.get('GPU_MAX_HW_QUEUES', 'not set')}")
        log.info("=" * 70)

    dispatch = {
        "full_pipeline": run_full_pipeline,
        "stream_fanout": run_stream_fanout,
        "h2d_compute_race": run_h2d_compute_race,
        "rccl_event_race": run_rccl_event_race,
        "threaded_poll": run_threaded_poll,
    }

    total_issues = 0
    for mode in modes:
        if rank == 0:
            log.info(f"\n--- Running: {mode} ---")
        if mode in multi_gpu and not distributed:
            if rank == 0:
                log.warning(f"  Skipping {mode} (requires torchrun)")
            continue

        t0 = time.time()
        issues, _, details = dispatch[mode](args, rank, device)
        elapsed = time.time() - t0
        total_issues += issues

        if rank == 0:
            log.info(f"  [{mode}] Done in {elapsed:.1f}s  issues={issues}")
            for d in details[:10]:
                log.info(f"    {d}")

    if rank == 0:
        log.info("")
        log.info("=" * 70)
        log.info(f"TOTAL ISSUES: {total_issues}")
        if total_issues > 0:
            log.info("VERDICT: Issues detected.")
        else:
            log.info("VERDICT: hipEventQuery appears correct under all tested conditions.")
        log.info("=" * 70)

    if distributed:
        dist.barrier()
        dist.destroy_process_group()

    sys.exit(1 if total_issues > 0 else 0)


if __name__ == "__main__":
    main()
