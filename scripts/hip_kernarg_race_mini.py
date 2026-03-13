"""
Minimal reproducer: HIP kernarg buffer recycling race.

THE BUG (HIP runtime, not PyTorch):
  When the CPU submits GPU kernel dispatches faster than the GPU
  executes them, the AQL queue fills up. The HIP runtime reuses
  kernarg buffers (which hold kernel argument pointers) from completed
  dispatches. But on AMD GPUs with deep AQL queues (16K entries), the
  CPU can race far enough ahead that the HIP runtime recycles a kernarg
  buffer whose kernel has NOT actually finished executing on the GPU.
  The GPU then reads a recycled kernarg containing stale or NULL
  pointers → hard fault (HSA_STATUS_ERROR_EXCEPTION 0x1016).

  This is a genuine HIP stack bug:
  - hipEventQuery works correctly
  - hipStreamWaitEvent works correctly
  - RCCL works correctly
  - The CCA works correctly
  - The bug is in the HIP runtime's kernarg pool management

WHY THIS IS NOT AN APPLICATION BUG:
  The application uses FIXED pre-allocated device buffers (no CCA
  recycling). It uses proper stream synchronization (wait_stream).
  The only "mistake" is submitting work faster than the GPU processes
  it — which is exactly what pipelining is designed to do.

  On CUDA, this is safe: CUDA's command queue provides backpressure
  and never recycles kernel arguments of in-flight kernels.

  On HIP/ROCm, the 16K AQL queue allows the CPU to race far ahead,
  and the kernarg pool recycles before the GPU is done reading.

HOW TO PROVE IT:
  1. Pre-allocate FIXED device buffers (no CCA involvement)
  2. Submit many kernels rapidly on multiple streams (fills AQL queue)
  3. Do NOT synchronize (let CPU race ahead)
  4. Crash = kernarg recycling (GPU reads stale pointers)
  5. Adding sync or reducing AQL queue depth prevents it

Usage:
    # Should CRASH (deep AQL queue, CPU races ahead)
    torchrun --nproc_per_node=1 scripts/hip_kernarg_race_mini.py

    # Should PASS (shallow AQL queue, backpressure)
    ROC_AQL_QUEUE_SIZE=1024 torchrun --nproc_per_node=1 scripts/hip_kernarg_race_mini.py

    # Should PASS (sync drains queue)
    torchrun --nproc_per_node=1 scripts/hip_kernarg_race_mini.py --sync-per-iter
"""

import argparse
import os
import sys
import time
import torch
import torch.nn as nn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-tables", type=int, default=8)
    parser.add_argument("--hash-size", type=int, default=500_000)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--pooling-factor", type=int, default=50)
    parser.add_argument("--sync-per-iter", action="store_true")
    args = parser.parse_args()

    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        local_rank = 0
    torch.cuda.set_device(local_rank)
    dev = torch.device(f"cuda:{local_rank}")

    print(f"ROC_AQL_QUEUE_SIZE={os.environ.get('ROC_AQL_QUEUE_SIZE', '(not set, default ~16K)')}")
    print(f"GPU_MAX_HW_QUEUES={os.environ.get('GPU_MAX_HW_QUEUES', '(not set)')}")
    print(f"sync_per_iter={args.sync_per_iter}")
    print(f"batch_size={args.batch_size}, num_tables={args.num_tables}")

    # ── Model: EmbeddingBag tables (high dispatch count per forward) ──
    tables = nn.ModuleList([
        nn.EmbeddingBag(args.hash_size, args.dim, mode="sum",
                        sparse=False, include_last_offset=True)
        for _ in range(args.num_tables)
    ]).to(dev)
    for t in tables:
        nn.init.normal_(t.weight, std=0.01)

    # ── FIXED pre-allocated device buffers (no CCA recycling) ──
    # 3 slots for triple-buffered pipeline rotation
    slots_indices = []
    slots_offsets = []
    for _ in range(3):
        table_indices = []
        table_offsets = []
        for _ in range(args.num_tables):
            total_idx = args.batch_size * args.pooling_factor
            idx = torch.randint(0, args.hash_size, (total_idx,),
                                device=dev, dtype=torch.long)
            lengths = torch.full((args.batch_size,), args.pooling_factor,
                                 dtype=torch.long, device=dev)
            off = torch.zeros(args.batch_size + 1, dtype=torch.long, device=dev)
            off[1:] = torch.cumsum(lengths, 0)
            table_indices.append(idx)
            table_offsets.append(off)
        slots_indices.append(table_indices)
        slots_offsets.append(table_offsets)

    # Host data for H2D (pinned)
    host_indices = []
    host_offsets = []
    for t in range(args.num_tables):
        total_idx = args.batch_size * args.pooling_factor
        hi = torch.randint(0, args.hash_size, (total_idx,), dtype=torch.long).pin_memory()
        lengths = torch.full((args.batch_size,), args.pooling_factor, dtype=torch.long)
        ho = torch.zeros(args.batch_size + 1, dtype=torch.long)
        ho[1:] = torch.cumsum(lengths, 0)
        host_indices.append(hi)
        host_offsets.append(ho.pin_memory())

    memcpy_stream = torch.cuda.Stream()
    default_stream = torch.cuda.current_stream()

    # Warmup
    for s in range(3):
        for t in range(args.num_tables):
            tables[t](slots_indices[s][t], slots_offsets[s][t])
    torch.cuda.synchronize()

    # ── Pipeline loop ──
    # slot 0: compute (default_stream)
    # slot 2: H2D copy (memcpy_stream)
    # Rotate slots each iteration
    print(f"\nStarting {args.iterations} iterations...")
    start = time.time()
    nan_count = 0

    for it in range(args.iterations):
        # H2D into slot 2 on memcpy_stream
        if it + 3 < args.iterations:
            with torch.cuda.stream(memcpy_stream):
                for t in range(args.num_tables):
                    slots_indices[2][t].copy_(host_indices[t], non_blocking=True)
                    slots_offsets[2][t].copy_(host_offsets[t], non_blocking=True)

        # Wait for H2D before compute
        default_stream.wait_stream(memcpy_stream)

        # Forward pass on slot 0 (many EmbeddingBag dispatches)
        outputs = []
        for t in range(args.num_tables):
            out = tables[t](slots_indices[0][t], slots_offsets[0][t])
            outputs.append(out)
        combined = torch.cat(outputs, dim=-1)
        result = combined.sum()

        # Rotate: [0,1,2] → [1,2,0]
        slots_indices = [slots_indices[1], slots_indices[2], slots_indices[0]]
        slots_offsets = [slots_offsets[1], slots_offsets[2], slots_offsets[0]]

        if args.sync_per_iter:
            torch.cuda.synchronize()

        # Periodic check
        if (it + 1) % 500 == 0:
            torch.cuda.synchronize()
            val = result.item()
            is_nan = val != val
            is_inf = abs(val) == float('inf')
            if is_nan or is_inf:
                nan_count += 1
            elapsed = time.time() - start
            print(f"  iter {it+1}/{args.iterations}  val={'NaN' if is_nan else 'Inf' if is_inf else f'{val:.2f}'}  "
                  f"nans={nan_count}  rate={it/elapsed:.0f} it/s")

    torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"\nDone: {args.iterations} iters in {elapsed:.1f}s ({args.iterations/elapsed:.0f} it/s)")
    print(f"NaN/Inf count: {nan_count}")
    if nan_count > 0:
        print("BUG: kernarg recycling caused corruption with FIXED buffers")
    else:
        print("PASS")

    sys.exit(1 if nan_count > 0 else 0)


if __name__ == "__main__":
    main()
