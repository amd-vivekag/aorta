"""
HIP event premature completion reproducer (v3).

Multiple approaches to test whether hipEventQuery() can return
hipSuccess before the GPU has actually finished:

Approach 1 (hip_event_direct):
  Bypass PyTorch's CCA entirely. Use raw HIP API calls (via ctypes)
  to create events, record them, and poll hipEventQuery in a tight
  loop while a long-running kernel is still reading from a buffer.
  Then overwrite the buffer and check the kernel's output.

Approach 2 (rccl_cca):
  Use RCCL all_to_all_single (async_op=True) with proper
  record_stream, matching Meta's exact workload pattern.
  RCCL kernels are long-running and interact differently with
  the event/signal mechanism than regular compute kernels.

Approach 3 (saturate_with_record_stream):
  Same as v2 saturate but with much heavier GPU work (large matmuls)
  and more aggressive CPU-side polling to widen the race window.

Approach 4 (event_query_vs_synchronize):
  Record an event after heavy GPU work, then poll hipEventQuery
  in a tight CPU loop. When it returns success, immediately check
  the output buffer. Compare with hipEventSynchronize path.
  If results differ, hipEventQuery is premature.

Usage:
    python scripts/meta_nan_hip_event_race.py --mode <mode> [options]

    Modes: hip_event_direct, rccl_cca, saturate_rs, event_compare, all
"""

import argparse
import ctypes
import logging
import os
import sys
import time
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


def compute_expected(initial_val: float, numel: int, chain_ops: int) -> float:
    expected = float(numel) * initial_val
    for _ in range(chain_ops):
        expected = expected * 1.00001 + 0.00001 * numel
    return expected


# ---------------------------------------------------------------------------
# Approach 1: Direct HIP event API via ctypes
# ---------------------------------------------------------------------------

def load_hip_runtime():
    """Load libamdhip64.so and bind the HIP event API functions."""
    try:
        hip = ctypes.CDLL("libamdhip64.so")
    except OSError:
        log.warning("Could not load libamdhip64.so, trying libamdhip64.so.6")
        hip = ctypes.CDLL("libamdhip64.so.6")

    hip.hipEventCreateWithFlags.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
    hip.hipEventCreateWithFlags.restype = ctypes.c_int
    hip.hipEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    hip.hipEventRecord.restype = ctypes.c_int
    hip.hipEventQuery.argtypes = [ctypes.c_void_p]
    hip.hipEventQuery.restype = ctypes.c_int
    hip.hipEventSynchronize.argtypes = [ctypes.c_void_p]
    hip.hipEventSynchronize.restype = ctypes.c_int
    hip.hipEventDestroy.argtypes = [ctypes.c_void_p]
    hip.hipEventDestroy.restype = ctypes.c_int
    hip.hipStreamSynchronize.argtypes = [ctypes.c_void_p]
    hip.hipStreamSynchronize.restype = ctypes.c_int

    return hip


def get_raw_stream(torch_stream):
    """Get the raw HIP stream pointer from a PyTorch stream."""
    return ctypes.c_void_p(torch_stream.cuda_stream)


def run_hip_event_direct(args, rank, device):
    """Bypass CCA: use raw HIP events to test hipEventQuery directly.

    Strategy:
    1. Allocate a buffer, fill with known value
    2. Launch heavy GPU work reading from buffer on stream_b
    3. Record a raw HIP event on stream_b (with hipEventDisableTiming)
    4. Poll hipEventQuery in a tight CPU loop
    5. The INSTANT hipEventQuery returns success, read the output on CPU
    6. Also overwrite the buffer on stream_a
    7. Compare: does the GPU result match what we expected from the
       original data, or from the overwritten data?

    Also test with hipEventDisableSystemFence to see if that changes behavior.
    """
    hip = load_hip_runtime()

    stream_a = torch.cuda.Stream()
    stream_b = torch.cuda.Stream()
    raw_stream_b = get_raw_stream(stream_b)

    hipEventDisableTiming = 0x02
    hipEventDisableSystemFence = 0x20

    numel = args.size_mb * 256 * 1024
    corruption_count = 0
    premature_count = 0
    details = []
    start = time.time()
    last_log = start

    flag_variants = [("default", 0)]
    test_event = ctypes.c_void_p()
    ret = hip.hipEventCreateWithFlags(ctypes.byref(test_event), hipEventDisableTiming | hipEventDisableSystemFence)
    if ret == 0:
        hip.hipEventDestroy(test_event)
        flag_variants.append(("DisableSystemFence", hipEventDisableSystemFence))
    else:
        # Clear the HIP error state so it doesn't pollute later calls
        torch.cuda.synchronize()
        try:
            _ = torch.empty(1, device=device)
        except Exception:
            pass
        log.info("  hipEventDisableSystemFence not supported on this HW, skipping")

    for flag_label, extra_flags in flag_variants:
        log.info(f"  Testing with event flags: {flag_label}")
        mode_corrupt = 0
        mode_premature = 0

        for it in range(args.iterations):
            with torch.cuda.stream(stream_a):
                src = torch.full((numel,), FILL_A, device=device, dtype=torch.float32)

            stream_b.wait_stream(stream_a)

            with torch.cuda.stream(stream_b):
                x = src.clone()
                for _ in range(args.chain_ops):
                    x = x * 1.00001 + 0.00001
                result_tensor = x.sum().unsqueeze(0)

            event = ctypes.c_void_p()
            flags = hipEventDisableTiming | extra_flags
            err = hip.hipEventCreateWithFlags(ctypes.byref(event), flags)
            assert err == 0, f"hipEventCreateWithFlags failed: {err}"

            err = hip.hipEventRecord(event, raw_stream_b)
            assert err == 0, f"hipEventRecord failed: {err}"

            query_count = 0
            while True:
                err = hip.hipEventQuery(event)
                query_count += 1
                if err == 0:  # hipSuccess
                    break
                elif err == 600:  # hipErrorNotReady
                    continue
                else:
                    assert False, f"hipEventQuery unexpected error: {err}"

            query_result = result_tensor.item()

            with torch.cuda.stream(stream_a):
                src.fill_(FILL_B)

            torch.cuda.synchronize()
            sync_result = result_tensor.item()

            expected = compute_expected(FILL_A, numel, args.chain_ops)

            query_err = abs(query_result - expected) / max(abs(expected), 1.0)
            sync_err = abs(sync_result - expected) / max(abs(expected), 1.0)

            if query_err > 1e-3 and sync_err <= 1e-3:
                mode_premature += 1
                premature_count += 1
                details.append(
                    f"iter={it} flags={flag_label}: PREMATURE "
                    f"query_result={query_result:.1f} sync_result={sync_result:.1f} "
                    f"expected={expected:.1f} queries={query_count}"
                )
                if args.stop_on_first:
                    break
            elif query_err > 1e-3:
                mode_corrupt += 1
                corruption_count += 1
                details.append(
                    f"iter={it} flags={flag_label}: BOTH_WRONG "
                    f"query={query_result:.1f} sync={sync_result:.1f} "
                    f"expected={expected:.1f}"
                )
                if args.stop_on_first:
                    break

            hip.hipEventDestroy(event)
            del src, x, result_tensor

            now = time.time()
            if rank == 0 and (mode_corrupt + mode_premature > 0 or now - last_log >= 5):
                log.info(
                    f"    [{flag_label}] iter={it+1}/{args.iterations}  "
                    f"premature={mode_premature}  corrupt={mode_corrupt}"
                )
                last_log = now

        if rank == 0:
            log.info(f"  {flag_label}: premature={mode_premature} corrupt={mode_corrupt}")

    return corruption_count + premature_count, premature_count, details


# ---------------------------------------------------------------------------
# Approach 2: RCCL with proper record_stream
# ---------------------------------------------------------------------------

def run_rccl_cca(args, rank, device):
    """RCCL all_to_all_single with record_stream -- Meta's exact pattern.

    The key difference from v2: we record_stream on the RCCL stream's
    input buffer so the CCA actually has an event to poll. We also make
    the side_stream's allocation happen in a tight loop right after the
    alltoall launch to maximize the chance of hitting the race.
    """
    if not dist.is_initialized():
        log.warning("Skipping rccl_cca mode (requires torchrun)")
        return 0, 0, []

    world_size = dist.get_world_size()
    default_stream = torch.cuda.current_stream()
    side_stream = torch.cuda.Stream()

    numel_per_rank = args.size_mb * 256 * 1024
    total_numel = numel_per_rank * world_size

    corruption_count = 0
    ptr_reuse_count = 0
    details = []
    start = time.time()
    last_log = start

    for it in range(args.iterations):
        send_buf = torch.full(
            (total_numel,), float(rank + 1) * FILL_A,
            device=device, dtype=torch.float32,
        )
        recv_buf = torch.empty(total_numel, device=device, dtype=torch.float32)
        original_ptr = send_buf.data_ptr()

        send_buf.record_stream(side_stream)

        work = dist.all_to_all_single(recv_buf, send_buf, async_op=True)

        del send_buf

        alloc_pressure = []
        with torch.cuda.stream(side_stream):
            for _ in range(args.alloc_pressure):
                t_new = torch.full(
                    (total_numel,), FILL_B,
                    device=device, dtype=torch.float32,
                )
                if t_new.data_ptr() == original_ptr:
                    ptr_reuse_count += 1
                alloc_pressure.append(t_new)

        work.wait()
        torch.cuda.synchronize()

        for src_rank in range(world_size):
            chunk = recv_buf[src_rank * numel_per_rank:(src_rank + 1) * numel_per_rank]
            expected_val = float(src_rank + 1) * FILL_A
            max_diff = (chunk - expected_val).abs().max().item()
            if max_diff > 0.01:
                corruption_count += 1
                details.append(
                    f"iter={it}: RCCL CORRUPTION from rank {src_rank}: "
                    f"max_diff={max_diff:.4f} ptr_reuse={ptr_reuse_count}"
                )
                break

        if corruption_count > 0 and args.stop_on_first:
            break

        del alloc_pressure, recv_buf

        now = time.time()
        if rank == 0 and (corruption_count > 0 or now - last_log >= 5):
            log.info(
                f"  [rccl_cca] iter={it+1}/{args.iterations}  "
                f"corrupt={corruption_count}  ptr_reuse={ptr_reuse_count}"
            )
            last_log = now

    return corruption_count, ptr_reuse_count, details


# ---------------------------------------------------------------------------
# Approach 3: Saturate with record_stream + heavy matmul work
# ---------------------------------------------------------------------------

def run_saturate_rs(args, rank, device):
    """4 stream pairs with record_stream + large matmuls for GPU work.

    Use large matrix multiplies instead of chained element-wise ops to
    create genuinely long-running single kernel dispatches. A single
    large matmul can keep the GPU busy for milliseconds -- much longer
    than a chain of small element-wise ops.
    """
    num_pairs = 4
    producers = [torch.cuda.Stream() for _ in range(num_pairs)]
    consumers = [torch.cuda.Stream() for _ in range(num_pairs)]

    numel = args.size_mb * 256 * 1024
    dim = int(numel ** 0.5)
    dim = max(dim, 1024)
    dim = min(dim, 8192)

    corruption_count = 0
    ptr_reuse_count = 0
    details = []
    start = time.time()
    last_log = start

    for it in range(args.iterations):
        results = []
        original_ptrs = []

        for pair_idx in range(num_pairs):
            with torch.cuda.stream(producers[pair_idx]):
                a = torch.full((dim, dim), FILL_A * (pair_idx + 1) / dim,
                               device=device, dtype=torch.float32)
                original_ptrs.append(a.data_ptr())

            a.record_stream(consumers[pair_idx])

            consumers[pair_idx].wait_stream(producers[pair_idx])
            with torch.cuda.stream(consumers[pair_idx]):
                for _ in range(args.chain_ops // 10 + 1):
                    b = torch.mm(a, a)
                results.append(b.sum())

            del a

        new_tensors = []
        for pair_idx in range(num_pairs):
            with torch.cuda.stream(producers[pair_idx]):
                for _ in range(args.alloc_pressure):
                    nt = torch.full((dim, dim), FILL_B,
                                    device=device, dtype=torch.float32)
                    if nt.data_ptr() in original_ptrs:
                        ptr_reuse_count += 1
                    new_tensors.append(nt)

        torch.cuda.synchronize()
        for pair_idx in range(num_pairs):
            val = FILL_A * (pair_idx + 1) / dim
            expected_elem = val * val * dim
            expected_sum = expected_elem * dim * dim
            for _ in range(args.chain_ops // 10):
                expected_sum = expected_sum  # matmul chain doesn't compound simply
            actual = results[pair_idx].item()
            # For matmul chains, just check for obviously wrong values
            # (NaN, massive deviation from order of magnitude)
            if torch.isnan(results[pair_idx]) or torch.isinf(results[pair_idx]):
                corruption_count += 1
                details.append(
                    f"iter={it} pair={pair_idx}: NaN/Inf detected actual={actual}"
                )
            elif abs(actual) < 1e-10 and abs(expected_sum) > 1.0:
                corruption_count += 1
                details.append(
                    f"iter={it} pair={pair_idx}: ZERO result, expected ~{expected_sum:.1f}"
                )

        if corruption_count > 0 and args.stop_on_first:
            break
        del results, new_tensors

        now = time.time()
        if rank == 0 and (corruption_count > 0 or now - last_log >= 5):
            log.info(
                f"  [saturate_rs] iter={it+1}/{args.iterations}  "
                f"corrupt={corruption_count}  ptr_reuse={ptr_reuse_count}"
            )
            last_log = now

    return corruption_count, ptr_reuse_count, details


# ---------------------------------------------------------------------------
# Approach 4: Event query vs synchronize comparison
# ---------------------------------------------------------------------------

def run_event_compare(args, rank, device):
    """Record event after heavy work, compare hipEventQuery vs hipEventSynchronize.

    This directly tests whether hipEventQuery gives the same answer as
    hipEventSynchronize. We:
    1. Launch heavy work on stream_b (large matmul chain)
    2. Record event on stream_b
    3. Poll hipEventQuery in tight loop until success
    4. IMMEDIATELY read the result tensor (before any other sync)
    5. Then call hipEventSynchronize and read again
    6. If results differ, hipEventQuery returned prematurely
    """
    hip = load_hip_runtime()

    stream_a = torch.cuda.Stream()
    stream_b = torch.cuda.Stream()
    raw_b = get_raw_stream(stream_b)

    hipEventDisableTiming = 0x02
    dim = min(int((args.size_mb * 256 * 1024) ** 0.5), 8192)
    dim = max(dim, 2048)

    mismatch_count = 0
    details = []
    start = time.time()
    last_log = start

    for it in range(args.iterations):
        with torch.cuda.stream(stream_b):
            a = torch.randn(dim, dim, device=device, dtype=torch.float32)
            for _ in range(args.chain_ops // 5 + 1):
                a = torch.mm(a, a.t())
            result_tensor = a.diagonal().sum().unsqueeze(0)

        event = ctypes.c_void_p()
        hip.hipEventCreateWithFlags(ctypes.byref(event), hipEventDisableTiming)
        hip.hipEventRecord(event, raw_b)

        poll_count = 0
        while hip.hipEventQuery(event) != 0:
            poll_count += 1

        query_val = result_tensor.item()

        hip.hipEventSynchronize(event)
        sync_val = result_tensor.item()

        hip.hipEventDestroy(event)

        if query_val != sync_val:
            mismatch_count += 1
            details.append(
                f"iter={it}: MISMATCH query={query_val} sync={sync_val} "
                f"polls={poll_count}"
            )
            if args.stop_on_first:
                break

        del a, result_tensor

        now = time.time()
        if rank == 0 and (mismatch_count > 0 or now - last_log >= 5):
            log.info(
                f"  [event_compare] iter={it+1}/{args.iterations}  "
                f"mismatch={mismatch_count}  polls_last={poll_count}"
            )
            last_log = now

    return mismatch_count, 0, details


# ---------------------------------------------------------------------------
# Approach 5: Cross-stream overwrite race with raw events
# ---------------------------------------------------------------------------

def run_cross_stream_raw(args, rank, device):
    """Cross-stream race using raw HIP events (not CCA).

    This manually implements the CCA's logic but with raw HIP calls:
    1. Allocate buffer on stream_a, fill with FILL_A
    2. Launch heavy read on stream_b
    3. Record event on stream_b
    4. Poll hipEventQuery -- when it returns success, immediately
       overwrite the buffer on stream_a with FILL_B
    5. Synchronize everything
    6. Check stream_b's result

    This isolates the hipEventQuery correctness from the CCA entirely.
    """
    hip = load_hip_runtime()

    stream_a = torch.cuda.Stream()
    stream_b = torch.cuda.Stream()
    raw_a = get_raw_stream(stream_a)
    raw_b = get_raw_stream(stream_b)

    hipEventDisableTiming = 0x02
    numel = args.size_mb * 256 * 1024

    corruption_count = 0
    details = []
    start = time.time()
    last_log = start

    for it in range(args.iterations):
        with torch.cuda.stream(stream_a):
            buf = torch.full((numel,), FILL_A, device=device, dtype=torch.float32)

        stream_b.wait_stream(stream_a)

        with torch.cuda.stream(stream_b):
            x = buf
            for _ in range(args.chain_ops):
                x = x * 1.00001 + 0.00001
            result = x.sum().unsqueeze(0)

        event = ctypes.c_void_p()
        hip.hipEventCreateWithFlags(ctypes.byref(event), hipEventDisableTiming)
        hip.hipEventRecord(event, raw_b)

        while hip.hipEventQuery(event) != 0:
            pass

        # Event says stream_b is done -- immediately overwrite on stream_a
        with torch.cuda.stream(stream_a):
            buf.fill_(FILL_B)

        torch.cuda.synchronize()

        expected = compute_expected(FILL_A, numel, args.chain_ops)
        actual = result.item()
        rel_err = abs(actual - expected) / max(abs(expected), 1.0)

        if rel_err > 1e-3:
            corruption_count += 1
            details.append(
                f"iter={it}: CROSS_STREAM_RAW CORRUPTION rel_err={rel_err:.6f} "
                f"actual={actual:.1f} expected={expected:.1f}"
            )
            if args.stop_on_first:
                break

        hip.hipEventDestroy(event)
        del buf, result

        now = time.time()
        if rank == 0 and (corruption_count > 0 or now - last_log >= 5):
            log.info(
                f"  [cross_stream_raw] iter={it+1}/{args.iterations}  "
                f"corrupt={corruption_count}"
            )
            last_log = now

    return corruption_count, 0, details


# ---------------------------------------------------------------------------
# Approach 6: Multiple HW queues + raw events + large matmul
# ---------------------------------------------------------------------------

def run_hwq_stress(args, rank, device):
    """Stress test: many streams + raw events + large matmuls.

    Create N stream pairs where each consumer runs a large matmul.
    Record raw HIP events on all consumer streams, poll them in a
    round-robin fashion, and overwrite the source buffers as soon as
    each event reports success. This maximizes cross-HW-queue contention.
    """
    hip = load_hip_runtime()

    num_pairs = 8
    producers = [torch.cuda.Stream() for _ in range(num_pairs)]
    consumers = [torch.cuda.Stream() for _ in range(num_pairs)]
    raw_consumers = [get_raw_stream(c) for c in consumers]

    hipEventDisableTiming = 0x02
    dim = min(int((args.size_mb * 256 * 1024) ** 0.5), 4096)
    dim = max(dim, 2048)

    corruption_count = 0
    details = []
    start = time.time()
    last_log = start

    for it in range(args.iterations):
        bufs = []
        results = []
        events = []

        # Phase 1: Launch heavy work on all consumer streams
        for i in range(num_pairs):
            with torch.cuda.stream(producers[i]):
                a = torch.full((dim, dim), FILL_A * (i + 1) / dim,
                               device=device, dtype=torch.float32)
            bufs.append(a)

            consumers[i].wait_stream(producers[i])
            with torch.cuda.stream(consumers[i]):
                for _ in range(args.chain_ops // 10 + 1):
                    b = torch.mm(a, a)
                results.append(b.sum().unsqueeze(0))

            event = ctypes.c_void_p()
            hip.hipEventCreateWithFlags(ctypes.byref(event), hipEventDisableTiming)
            hip.hipEventRecord(event, raw_consumers[i])
            events.append(event)

        # Phase 2: Poll events and overwrite as soon as each reports done
        completed = [False] * num_pairs
        while not all(completed):
            for i in range(num_pairs):
                if completed[i]:
                    continue
                if hip.hipEventQuery(events[i]) == 0:
                    completed[i] = True
                    with torch.cuda.stream(producers[i]):
                        bufs[i].fill_(FILL_B)

        # Phase 3: Synchronize and verify
        torch.cuda.synchronize()
        for i in range(num_pairs):
            actual = results[i].item()
            if torch.isnan(torch.tensor(actual)) or torch.isinf(torch.tensor(actual)):
                corruption_count += 1
                details.append(f"iter={it} pair={i}: NaN/Inf")
            # Check if result contains FILL_B influence
            val = FILL_A * (i + 1) / dim
            expected_elem = val * val * dim
            expected_sum = expected_elem * dim * dim
            if actual < 0 and expected_sum > 0:
                corruption_count += 1
                details.append(
                    f"iter={it} pair={i}: SIGN FLIP actual={actual:.1f} expected>0"
                )

            hip.hipEventDestroy(events[i])

        if corruption_count > 0 and args.stop_on_first:
            break
        del bufs, results, events

        now = time.time()
        if rank == 0 and (corruption_count > 0 or now - last_log >= 5):
            log.info(
                f"  [hwq_stress] iter={it+1}/{args.iterations}  "
                f"corrupt={corruption_count}"
            )
            last_log = now

    return corruption_count, 0, details


def main():
    parser = argparse.ArgumentParser(
        description="HIP event premature completion reproducer (v3)"
    )
    parser.add_argument(
        "--mode",
        choices=[
            "hip_event_direct", "rccl_cca", "saturate_rs",
            "event_compare", "cross_stream_raw", "hwq_stress", "all"
        ],
        default="all",
    )
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--size-mb", type=int, default=16)
    parser.add_argument("--chain-ops", type=int, default=50)
    parser.add_argument("--alloc-pressure", type=int, default=16)
    parser.add_argument("--pressure", choices=["normal", "high"], default="normal")
    parser.add_argument("--stop-on-first", action="store_true", default=True)
    parser.add_argument("--no-stop-on-first", dest="stop_on_first", action="store_false")
    parser.add_argument("--skip-record-stream", action="store_true")
    parser.add_argument("--gc-off", action="store_true")
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

    if args.gc_off:
        import gc
        gc.disable()

    single_gpu_modes = [
        "hip_event_direct", "cross_stream_raw", "event_compare",
        "saturate_rs", "hwq_stress",
    ]
    multi_gpu_modes = ["rccl_cca"]

    if args.mode == "all":
        modes = list(single_gpu_modes)
        if distributed:
            modes.extend(multi_gpu_modes)
    else:
        modes = [args.mode]

    if rank == 0:
        log.info("=" * 70)
        log.info("HIP EVENT PREMATURE COMPLETION REPRODUCER (v3)")
        log.info("=" * 70)
        log.info(f"  modes: {modes}")
        log.info(f"  iterations: {args.iterations}")
        log.info(f"  size_mb: {args.size_mb}")
        log.info(f"  chain_ops: {args.chain_ops}")
        ws = dist.get_world_size() if distributed else 1
        log.info(f"  world_size: {ws}")
        log.info(f"  GPU_MAX_HW_QUEUES: {os.environ.get('GPU_MAX_HW_QUEUES', 'not set')}")
        log.info("=" * 70)

    total_issues = 0
    all_details = []

    dispatch = {
        "hip_event_direct": run_hip_event_direct,
        "rccl_cca": run_rccl_cca,
        "saturate_rs": run_saturate_rs,
        "event_compare": run_event_compare,
        "cross_stream_raw": run_cross_stream_raw,
        "hwq_stress": run_hwq_stress,
    }

    for mode in modes:
        if rank == 0:
            log.info(f"\n--- Running mode: {mode} ---")

        t_start = time.time()
        fn = dispatch.get(mode)
        if fn is None:
            continue

        if mode in multi_gpu_modes and not distributed:
            if rank == 0:
                log.warning(f"Skipping {mode} (requires torchrun)")
            continue

        issues, extra, details = fn(args, rank, device)
        elapsed = time.time() - t_start
        total_issues += issues
        all_details.extend(details)

        if rank == 0:
            log.info(f"  [{mode}] Done in {elapsed:.1f}s  issues={issues}")
            for d in details[:10]:
                log.info(f"    {d}")

    if rank == 0:
        log.info("")
        log.info("=" * 70)
        log.info("FINAL RESULTS")
        log.info("=" * 70)
        log.info(f"  Total issues: {total_issues}")
        if total_issues > 0:
            log.info("VERDICT: Issues detected -- see details above.")
        else:
            log.info("VERDICT: No premature hipEventQuery completion detected.")
        log.info("=" * 70)

    if distributed:
        dist.barrier()
        dist.destroy_process_group()

    sys.exit(1 if total_issues > 0 else 0)


if __name__ == "__main__":
    main()
