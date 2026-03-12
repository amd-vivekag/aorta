"""
Direct stream race condition reproducer for AMD MI355X.

This bypasses Shampoo entirely and directly tests the hypothesis:
  "bf16→fp32 gradient copies on separate CUDA streams, without
   synchronization to the default stream, cause data corruption
   when the default stream reads the fp32 copies."

The test:
  1. Create bf16 gradient tensors on stream 0 (simulating backward pass)
  2. Copy them to fp32 on N separate streams WITHOUT sync to stream 0
  3. Immediately read the fp32 copies on stream 0 (simulating preconditioner update)
  4. Check if the read values match the expected values

If the race exists, step 3 will sometimes read uninitialized/partial data.

Usage:
    # Test for race (should show corruption on AMD with GPU_MAX_HW_QUEUES>=4)
    GPU_MAX_HW_QUEUES=4 python scripts/stream_race_direct.py --iterations 100000

    # Same test with sync (should show no corruption)
    GPU_MAX_HW_QUEUES=4 python scripts/stream_race_direct.py --iterations 100000 --sync

    # With larger tensors (bigger race window)
    GPU_MAX_HW_QUEUES=4 python scripts/stream_race_direct.py --iterations 100000 --tensor-mb 16
"""

import argparse
import os
import time

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=100000)
    parser.add_argument("--num-streams", type=int, default=15,
                        help="Number of copy streams (Meta trace shows 15)")
    parser.add_argument("--num-tensors", type=int, default=30,
                        help="Number of gradient tensors to copy per iteration")
    parser.add_argument("--tensor-mb", type=float, default=1.0,
                        help="Size of each tensor in MB")
    parser.add_argument("--sync", action="store_true",
                        help="Add stream sync (should fix the race)")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    os.environ.setdefault("GPU_MAX_HW_QUEUES", "4")

    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)

    print(f"{'='*60}")
    print(f"DIRECT STREAM RACE TEST")
    print(f"{'='*60}")
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"Iterations: {args.iterations}")
    print(f"Streams: {args.num_streams}")
    print(f"Tensors: {args.num_tensors} x {args.tensor_mb}MB")
    print(f"Sync: {args.sync}")
    print(f"GPU_MAX_HW_QUEUES={os.environ.get('GPU_MAX_HW_QUEUES')}")
    print(f"{'='*60}")

    copy_streams = [torch.cuda.Stream() for _ in range(args.num_streams)]

    nelems = int(args.tensor_mb * 1024 * 1024 / 2)  # bf16 = 2 bytes

    bf16_tensors = [
        torch.randn(nelems, device=device, dtype=torch.bfloat16)
        for _ in range(args.num_tensors)
    ]
    fp32_targets = [
        torch.empty(nelems, device=device, dtype=torch.float32)
        for _ in range(args.num_tensors)
    ]

    corruptions = 0
    nan_corruptions = 0
    mismatch_corruptions = 0
    total_checks = 0
    t0 = time.time()

    for iteration in range(args.iterations):
        for i, (bf16_src, fp32_dst) in enumerate(zip(bf16_tensors, fp32_targets)):
            bf16_src.normal_()

        for i, (bf16_src, fp32_dst) in enumerate(zip(bf16_tensors, fp32_targets)):
            stream = copy_streams[i % args.num_streams]
            with torch.cuda.stream(stream):
                fp32_dst.copy_(bf16_src)

        if args.sync:
            for s in copy_streams:
                torch.cuda.current_stream().wait_stream(s)

        for i, (bf16_src, fp32_dst) in enumerate(zip(bf16_tensors, fp32_targets)):
            total_checks += 1
            has_nan = torch.isnan(fp32_dst).any().item()
            has_inf = torch.isinf(fp32_dst).any().item()

            if has_nan or has_inf:
                nan_corruptions += 1
                corruptions += 1
                nan_ct = torch.isnan(fp32_dst).sum().item()
                inf_ct = torch.isinf(fp32_dst).sum().item()
                print(f"  CORRUPTION (NaN/Inf) iter={iteration} tensor={i}: "
                      f"NaN={nan_ct} Inf={inf_ct}")

            expected = bf16_src.float()
            if not torch.equal(fp32_dst, expected):
                diff = (fp32_dst - expected).abs()
                max_diff = diff.max().item()
                if max_diff > 0:
                    mismatch_corruptions += 1
                    corruptions += 1
                    if mismatch_corruptions <= 10:
                        mismatched = (diff > 0).sum().item()
                        print(f"  CORRUPTION (mismatch) iter={iteration} tensor={i}: "
                              f"max_diff={max_diff:.6e} mismatched={mismatched}/{nelems}")

        if corruptions > 100:
            print(f"Too many corruptions, stopping")
            break

        if (iteration + 1) % 10000 == 0:
            elapsed = time.time() - t0
            rate = (iteration + 1) / elapsed
            print(f"  iter {iteration+1}/{args.iterations} | "
                  f"{rate:.0f} iter/s | corruptions={corruptions}")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Iterations: {args.iterations}")
    print(f"Total checks: {total_checks}")
    print(f"NaN/Inf corruptions: {nan_corruptions}")
    print(f"Mismatch corruptions: {mismatch_corruptions}")
    print(f"Total corruptions: {corruptions}")
    print(f"Elapsed: {elapsed:.1f}s")
    if corruptions > 0:
        print(f"\nSTREAM RACE CONFIRMED!")
    else:
        print(f"\nNo corruption detected (race may need different conditions)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
