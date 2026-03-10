"""
HIP event premature completion reproducer.

Tests whether hipEventQuery() reports completion before the GPU has
actually finished using a memory block, causing the CachingAllocator
to recycle it too early.

Mechanism under test:
  1. Allocate tensor T on stream_a via CCA (normal torch.empty)
  2. Fill T with a known pattern on stream_a
  3. Use T on stream_b (cross-stream usage triggers CCA recordStream)
  4. Launch a LONG-RUNNING operation on stream_b that reads from T
  5. Drop T's Python reference -- CCA records hipEvent on stream_b,
     moves block to "pending free" list
  6. Allocate new tensor on stream_a -- CCA's process_events() polls
     hipEventQuery() on stream_b's event. If it returns success
     prematurely, CCA hands out the same block
  7. Overwrite the block with a different pattern on stream_a
  8. When stream_b's operation finishes, verify it read the ORIGINAL
     pattern (not the overwrite)

If corruption is detected, it means hipEventQuery() returned success
before stream_b finished reading -- confirming premature event completion.

The test has multiple modes to maximize the chance of hitting the race:
  - "alloc_free": Pure CCA alloc/free cycling with cross-stream reads
  - "rccl": Uses RCCL all_to_all to create long-running cross-stream ops
  - "multi_stream": Uses 3+ streams to match TorchRec pipeline depth

Usage:
    # Single-GPU: pure CCA event race test
    python scripts/meta_nan_hip_event_race.py --mode alloc_free

    # Multi-GPU: RCCL collectives create longer GPU-side operations
    torchrun --nproc_per_node=2 scripts/meta_nan_hip_event_race.py --mode rccl

    # 3-stream pipeline matching TorchRec pattern
    torchrun --nproc_per_node=2 scripts/meta_nan_hip_event_race.py --mode pipeline

    # Stress: maximize alloc/free pressure with multiple sizes
    python scripts/meta_nan_hip_event_race.py --mode alloc_free --pressure high

    # With GPU_MAX_HW_QUEUES control (important for reproducing)
    GPU_MAX_HW_QUEUES=4 python scripts/meta_nan_hip_event_race.py --mode alloc_free
"""

import argparse
import logging
import os
import sys
import time
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

FILL_ORIGINAL = 42.0
FILL_OVERWRITE = -999.0


def make_heavy_workload(tensor: torch.Tensor, num_ops: int = 20) -> torch.Tensor:
    """Chain enough GPU ops on a tensor to keep stream_b busy.

    The goal is to make stream_b's work take long enough that the CPU
    can run process_events() and potentially get a premature hipEventQuery
    success while stream_b is still reading from the tensor.
    """
    x = tensor
    for _ in range(num_ops):
        x = x * 1.00001 + 0.00001
    return x.sum()


def run_alloc_free_test(args, rank, device):
    """Pure CCA event race: alloc on stream_a, use on stream_b, free, reallocate.

    This is the most direct test of the hipEventQuery premature completion
    hypothesis. No model, no RCCL -- just the CCA event path.
    """
    stream_a = torch.cuda.Stream()
    stream_b = torch.cuda.Stream()

    corruption_count = 0
    false_reuse_count = 0
    details = []
    start = time.time()
    last_log = start

    sizes = [args.size_mb * 256 * 1024]  # float32 elements
    if args.pressure == "high":
        sizes = [s * 256 * 1024 for s in [1, 4, 16, 64, 256]]

    for it in range(args.iterations):
        numel = sizes[it % len(sizes)]

        with torch.cuda.stream(stream_a):
            t = torch.full((numel,), FILL_ORIGINAL, device=device, dtype=torch.float32)
            original_ptr = t.data_ptr()

        stream_b.wait_stream(stream_a)
        with torch.cuda.stream(stream_b):
            result = make_heavy_workload(t, num_ops=args.chain_ops)

        # Drop reference: CCA records hipEvent on stream_b, block goes to pending
        del t

        # Force CCA to run process_events() by allocating on stream_a.
        # If hipEventQuery(stream_b_event) returns success prematurely,
        # CCA will hand us back the SAME block.
        alloc_pressure = []
        with torch.cuda.stream(stream_a):
            for _ in range(args.alloc_pressure):
                new_t = torch.full((numel,), FILL_OVERWRITE, device=device, dtype=torch.float32)
                if new_t.data_ptr() == original_ptr:
                    false_reuse_count += 1
                alloc_pressure.append(new_t)

        torch.cuda.synchronize()

        expected = float(numel) * FILL_ORIGINAL
        for _ in range(args.chain_ops):
            expected = expected * 1.00001 + 0.00001 * numel
        actual = result.item()

        rel_err = abs(actual - expected) / max(abs(expected), 1.0)
        if rel_err > 1e-3:
            corruption_count += 1
            details.append(
                f"iter={it}: CORRUPTION rel_err={rel_err:.6f} "
                f"expected={expected:.1f} actual={actual:.1f} "
                f"reuse_count={false_reuse_count}"
            )
            if args.stop_on_first:
                break

        del alloc_pressure, new_t, result

        now = time.time()
        if rank == 0 and (corruption_count > 0 or now - last_log >= 5):
            elapsed = now - start
            rate = (it + 1) / elapsed
            log.info(
                f"  [alloc_free] iter={it+1}/{args.iterations}  "
                f"corrupt={corruption_count}  ptr_reuse={false_reuse_count}  "
                f"rate={rate:.0f} it/s"
            )
            last_log = now

    return corruption_count, false_reuse_count, details


def run_rccl_test(args, rank, device):
    """RCCL collectives on stream_b create a longer GPU-side operation.

    This matches Meta's pattern more closely: alltoall_base_ on the
    default stream reads from a tensor, while the data_dist_stream
    allocates a new tensor via CCA that gets the same block.
    """
    world_size = dist.get_world_size()
    default_stream = torch.cuda.current_stream()
    side_stream = torch.cuda.Stream()

    numel_per_rank = args.size_mb * 256 * 1024
    total_numel = numel_per_rank * world_size

    corruption_count = 0
    false_reuse_count = 0
    details = []
    start = time.time()
    last_log = start

    for it in range(args.iterations):
        # Step 1: Allocate send buffer on default_stream, fill with rank-specific pattern
        send_buf = torch.full(
            (total_numel,), float(rank + 1) * FILL_ORIGINAL,
            device=device, dtype=torch.float32,
        )
        recv_buf = torch.empty(total_numel, device=device, dtype=torch.float32)
        original_ptr = send_buf.data_ptr()
        expected_val = float(rank + 1) * FILL_ORIGINAL

        # Step 2: Launch async alltoall -- keeps default_stream busy reading send_buf
        work = dist.all_to_all_single(
            recv_buf, send_buf, async_op=True
        )

        # Step 3: Also do heavy compute on default stream reading from send_buf
        # to extend the window where send_buf is "in use" on GPU
        with torch.cuda.stream(default_stream):
            checksum = make_heavy_workload(send_buf, num_ops=args.chain_ops)

        # Step 4: Drop send_buf reference.
        # CCA should record event on default_stream and put block in pending.
        del send_buf

        # Step 5: On side_stream, allocate to trigger CCA process_events().
        # If hipEventQuery returns success prematurely for default_stream's event,
        # CCA recycles send_buf's block while alltoall is still reading it.
        alloc_pressure = []
        with torch.cuda.stream(side_stream):
            for _ in range(args.alloc_pressure):
                t_new = torch.full(
                    (total_numel,), FILL_OVERWRITE,
                    device=device, dtype=torch.float32,
                )
                if t_new.data_ptr() == original_ptr:
                    false_reuse_count += 1
                alloc_pressure.append(t_new)

        # Step 6: Wait for everything and verify
        work.wait()
        torch.cuda.synchronize()

        # Verify recv_buf: each chunk should contain (src_rank+1)*FILL_ORIGINAL
        corrupted = False
        for src_rank in range(world_size):
            chunk = recv_buf[src_rank * numel_per_rank:(src_rank + 1) * numel_per_rank]
            expected_chunk_val = float(src_rank + 1) * FILL_ORIGINAL
            max_diff = (chunk - expected_chunk_val).abs().max().item()
            if max_diff > 0.01:
                corrupted = True
                details.append(
                    f"iter={it}: RCCL CORRUPTION from rank {src_rank}: "
                    f"max_diff={max_diff:.4f} expected={expected_chunk_val:.1f} "
                    f"ptr_reuse={false_reuse_count}"
                )

        # Verify checksum
        expected_sum = float(total_numel) * expected_val
        for _ in range(args.chain_ops):
            expected_sum = expected_sum * 1.00001 + 0.00001 * total_numel
        actual_sum = checksum.item()
        rel_err = abs(actual_sum - expected_sum) / max(abs(expected_sum), 1.0)
        if rel_err > 1e-3:
            corrupted = True
            details.append(
                f"iter={it}: CHECKSUM CORRUPTION rel_err={rel_err:.6f}"
            )

        if corrupted:
            corruption_count += 1
            if args.stop_on_first:
                break

        del alloc_pressure, recv_buf, checksum

        now = time.time()
        if rank == 0 and (corruption_count > 0 or now - last_log >= 5):
            elapsed = now - start
            rate = (it + 1) / elapsed
            log.info(
                f"  [rccl] iter={it+1}/{args.iterations}  "
                f"corrupt={corruption_count}  ptr_reuse={false_reuse_count}  "
                f"rate={rate:.0f} it/s"
            )
            last_log = now

    return corruption_count, false_reuse_count, details


def run_pipeline_test(args, rank, device):
    """3-stream pipeline matching TorchRec's TrainPipelineSparseDist.

    Simulates the exact CSAN-detected pattern:
      - default_stream: forward pass + RCCL collectives reading from tensors
      - datadist_stream: waits for collectives, allocates output buffers (CCA)
      - memcpy_stream: H2D copies into rotating buffer slots

    The race: datadist_stream's torch.empty() gets a block that
    default_stream's alltoall is still reading from.
    """
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    use_dist = dist.is_initialized() and world_size > 1

    default_stream = torch.cuda.current_stream()
    datadist_stream = torch.cuda.Stream()
    memcpy_stream = torch.cuda.Stream()

    numel = args.size_mb * 256 * 1024
    dim = 256

    # Host-pinned pool for H2D
    host_pool = [
        torch.randn(numel, dtype=torch.float32).pin_memory()
        for _ in range(32)
    ]

    corruption_count = 0
    false_reuse_count = 0
    details = []
    start = time.time()
    last_log = start

    # Tracking which data pointers are "in use" by which stream
    active_ptrs = {}  # ptr -> (stream_name, iteration)

    for it in range(args.iterations):
        host_data = host_pool[it % len(host_pool)]

        # -- memcpy_stream: H2D copy (allocate via CCA each time) --
        with torch.cuda.stream(memcpy_stream):
            device_buf = torch.empty(numel, device=device, dtype=torch.float32)
            device_buf.copy_(host_data, non_blocking=True)
            h2d_ptr = device_buf.data_ptr()

        # -- default_stream: "forward pass" using the previous iteration's data --
        default_stream.wait_stream(memcpy_stream)
        with torch.cuda.stream(default_stream):
            compute_result = make_heavy_workload(device_buf, num_ops=args.chain_ops)

            if use_dist:
                send_t = torch.empty(
                    numel, device=device, dtype=torch.float32
                )
                send_t.copy_(device_buf)
                recv_t = torch.empty_like(send_t)
                a2a_work = dist.all_to_all_single(recv_t, send_t, async_op=True)

        # -- datadist_stream: allocate output buffer (this is where CSAN detected the race) --
        # Drop device_buf reference so CCA can potentially recycle it
        old_ptr = device_buf.data_ptr()
        del device_buf

        with torch.cuda.stream(datadist_stream):
            # These allocations trigger CCA's process_events()
            # If hipEventQuery for default_stream's event returns success
            # prematurely, CCA will hand out old_ptr's block
            for _ in range(args.alloc_pressure):
                datadist_buf = torch.full(
                    (numel,), FILL_OVERWRITE,
                    device=device, dtype=torch.float32,
                )
                if datadist_buf.data_ptr() == old_ptr:
                    false_reuse_count += 1
                    details.append(
                        f"iter={it}: PTR REUSE detected! "
                        f"datadist_stream got block 0x{old_ptr:x} "
                        f"while default_stream may still be reading it"
                    )
                del datadist_buf

        # Synchronize and verify
        if use_dist:
            a2a_work.wait()
        torch.cuda.synchronize()

        expected_sum = host_data.sum().item()
        for _ in range(args.chain_ops):
            expected_sum = expected_sum * 1.00001 + 0.00001 * numel
        actual_sum = compute_result.item()
        rel_err = abs(actual_sum - expected_sum) / max(abs(expected_sum), 1.0)

        if rel_err > 1e-3:
            corruption_count += 1
            details.append(
                f"iter={it}: PIPELINE CORRUPTION rel_err={rel_err:.6f} "
                f"expected={expected_sum:.1f} actual={actual_sum:.1f}"
            )
            if args.stop_on_first:
                break

        if use_dist:
            del send_t, recv_t
        del compute_result

        now = time.time()
        if rank == 0 and (corruption_count > 0 or now - last_log >= 5):
            elapsed = now - start
            rate = (it + 1) / elapsed
            log.info(
                f"  [pipeline] iter={it+1}/{args.iterations}  "
                f"corrupt={corruption_count}  ptr_reuse={false_reuse_count}  "
                f"rate={rate:.0f} it/s"
            )
            last_log = now

    return corruption_count, false_reuse_count, details


def run_saturate_test(args, rank, device):
    """Saturate the CCA's pending-event list to maximize hipEventQuery pressure.

    Strategy: rapidly create and destroy cross-stream tensors so the CCA
    has many pending events to poll. This increases the probability that
    at least one hipEventQuery returns success prematurely.

    Uses multiple producer/consumer stream pairs in parallel.
    """
    num_stream_pairs = 4
    producers = [torch.cuda.Stream() for _ in range(num_stream_pairs)]
    consumers = [torch.cuda.Stream() for _ in range(num_stream_pairs)]

    numel = args.size_mb * 256 * 1024
    corruption_count = 0
    false_reuse_count = 0
    details = []
    start = time.time()
    last_log = start

    for it in range(args.iterations):
        results = []
        original_ptrs = []

        # Phase 1: Create tensors on producers, use on consumers
        for pair_idx in range(num_stream_pairs):
            with torch.cuda.stream(producers[pair_idx]):
                t = torch.full(
                    (numel,), FILL_ORIGINAL * (pair_idx + 1),
                    device=device, dtype=torch.float32,
                )
                original_ptrs.append(t.data_ptr())

            consumers[pair_idx].wait_stream(producers[pair_idx])
            with torch.cuda.stream(consumers[pair_idx]):
                r = make_heavy_workload(t, num_ops=args.chain_ops)
                results.append(r)

            # Drop reference -- CCA records event on consumer stream
            del t

        # Phase 2: Immediately allocate on all producer streams
        # This floods CCA with process_events() calls
        new_tensors = []
        for pair_idx in range(num_stream_pairs):
            with torch.cuda.stream(producers[pair_idx]):
                for _ in range(args.alloc_pressure):
                    nt = torch.full(
                        (numel,), FILL_OVERWRITE,
                        device=device, dtype=torch.float32,
                    )
                    if nt.data_ptr() in original_ptrs:
                        false_reuse_count += 1
                    new_tensors.append(nt)

        # Phase 3: Verify
        torch.cuda.synchronize()
        for pair_idx in range(num_stream_pairs):
            expected = float(numel) * FILL_ORIGINAL * (pair_idx + 1)
            for _ in range(args.chain_ops):
                expected = expected * 1.00001 + 0.00001 * numel
            actual = results[pair_idx].item()
            rel_err = abs(actual - expected) / max(abs(expected), 1.0)
            if rel_err > 1e-3:
                corruption_count += 1
                details.append(
                    f"iter={it} pair={pair_idx}: SATURATE CORRUPTION "
                    f"rel_err={rel_err:.6f}"
                )

        if corruption_count > 0 and args.stop_on_first:
            break

        del results, new_tensors

        now = time.time()
        if rank == 0 and (corruption_count > 0 or now - last_log >= 5):
            elapsed = now - start
            rate = (it + 1) / elapsed
            log.info(
                f"  [saturate] iter={it+1}/{args.iterations}  "
                f"corrupt={corruption_count}  ptr_reuse={false_reuse_count}  "
                f"rate={rate:.0f} it/s"
            )
            last_log = now

    return corruption_count, false_reuse_count, details


def main():
    parser = argparse.ArgumentParser(
        description="HIP event premature completion reproducer"
    )
    parser.add_argument(
        "--mode",
        choices=["alloc_free", "rccl", "pipeline", "saturate", "all"],
        default="all",
    )
    parser.add_argument("--iterations", type=int, default=50000)
    parser.add_argument("--size-mb", type=int, default=16,
                        help="Tensor size in MB per allocation")
    parser.add_argument("--chain-ops", type=int, default=30,
                        help="Number of chained ops to extend GPU-side read time")
    parser.add_argument("--alloc-pressure", type=int, default=8,
                        help="Number of allocations to trigger CCA process_events")
    parser.add_argument("--pressure", choices=["normal", "high"], default="normal",
                        help="Allocation pressure level for alloc_free mode")
    parser.add_argument("--stop-on-first", action="store_true", default=True)
    parser.add_argument("--no-stop-on-first", dest="stop_on_first", action="store_false")
    parser.add_argument("--gc-off", action="store_true",
                        help="Disable Python GC to reduce interference")
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
        if rank == 0:
            log.info("Python GC disabled")

    modes_to_run = []
    if args.mode == "all":
        modes_to_run = ["alloc_free", "saturate"]
        if distributed:
            modes_to_run.extend(["rccl", "pipeline"])
    else:
        modes_to_run = [args.mode]

    if rank == 0:
        log.info("=" * 70)
        log.info("HIP EVENT PREMATURE COMPLETION REPRODUCER")
        log.info("=" * 70)
        log.info(f"  modes: {modes_to_run}")
        log.info(f"  iterations: {args.iterations}")
        log.info(f"  size_mb: {args.size_mb}")
        log.info(f"  chain_ops: {args.chain_ops}")
        log.info(f"  alloc_pressure: {args.alloc_pressure}")
        ws = dist.get_world_size() if distributed else 1
        log.info(f"  world_size: {ws}")
        log.info(f"  ROC_AQL_QUEUE_SIZE: {os.environ.get('ROC_AQL_QUEUE_SIZE', 'not set')}")
        log.info(f"  GPU_MAX_HW_QUEUES: {os.environ.get('GPU_MAX_HW_QUEUES', 'not set')}")
        log.info("=" * 70)

    total_corruption = 0
    total_reuse = 0
    all_details = []

    for mode in modes_to_run:
        if rank == 0:
            log.info(f"\n--- Running mode: {mode} ---")

        t_start = time.time()

        if mode == "alloc_free":
            corrupt, reuse, details = run_alloc_free_test(args, rank, device)
        elif mode == "rccl":
            if not distributed:
                if rank == 0:
                    log.warning("Skipping rccl mode (requires torchrun with 2+ GPUs)")
                continue
            corrupt, reuse, details = run_rccl_test(args, rank, device)
        elif mode == "pipeline":
            corrupt, reuse, details = run_pipeline_test(args, rank, device)
        elif mode == "saturate":
            corrupt, reuse, details = run_saturate_test(args, rank, device)
        else:
            continue

        elapsed = time.time() - t_start
        total_corruption += corrupt
        total_reuse += reuse
        all_details.extend(details)

        if rank == 0:
            log.info(
                f"  [{mode}] Done in {elapsed:.1f}s  "
                f"corruption={corrupt}  ptr_reuse={reuse}"
            )
            for d in details[:10]:
                log.info(f"    {d}")

    if rank == 0:
        log.info("")
        log.info("=" * 70)
        log.info("FINAL RESULTS")
        log.info("=" * 70)
        log.info(f"  Total corruptions: {total_corruption}")
        log.info(f"  Total ptr reuses: {total_reuse}")

        if total_corruption > 0:
            log.info("")
            log.info(
                "VERDICT: PREMATURE HIP EVENT COMPLETION CONFIRMED -- "
                "hipEventQuery() returned success before GPU finished "
                "reading from the tensor. CCA recycled the block too early."
            )
        elif total_reuse > 0:
            log.info("")
            log.info(
                "VERDICT: CCA recycled blocks (ptr reuse detected) but "
                "no data corruption observed. The GPU may have finished "
                "in time despite premature event query, or the recycled "
                "block was not overwritten before the read completed."
            )
        else:
            log.info("")
            log.info(
                "VERDICT: No premature event completion detected. "
                "hipEventQuery() appears correct for this workload pattern."
            )
            log.info(
                "NOTE: The bug may require specific HW queue configurations "
                "or RCCL collective patterns to trigger. Try:\n"
                "  GPU_MAX_HW_QUEUES=4 (more parallelism)\n"
                "  --chain-ops 50 (longer GPU operations)\n"
                "  --size-mb 64 (larger tensors)\n"
                "  --alloc-pressure 16 (more CCA pressure)\n"
                "  torchrun --nproc_per_node=8 (more RCCL traffic)"
            )
        log.info("=" * 70)

    if distributed:
        dist.barrier()
        dist.destroy_process_group()

    sys.exit(1 if total_corruption > 0 else 0)


if __name__ == "__main__":
    main()
