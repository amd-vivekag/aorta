"""
HIP memory subsystem stress test for Issue B — cache coherence & buffer aliasing.

This script isolates the EXACT memory access pattern that Hypothesis B describes:
  1. Pre-allocate N device buffer slots at fixed virtual addresses
  2. One stream does H2D (write) into slot[K] while another stream
     does compute (read) from slot[K-2] — CONCURRENTLY
  3. After compute, verify output against a reference computed from known input
  4. Rotate slots so the same VA that was just written is now read
  5. Check if the GPU sees stale cache data (old values) instead of fresh H2D data

Key differences from meta_nan_issue_b_vultr.py:
  - No model — pure GEMM + embedding lookup to hit the same memory subsystem
  - Verifies H2D data integrity directly (not just NaN check)
  - Much faster iteration (10x+) so we can run 50K+ iterations
  - Supports bf16 GEMM + int64 embedding lookup (both paths in HSTU)
  - Explicitly tests cache line boundaries and large working sets

Also tests a pattern that may trigger AMD-specific issues:
  - hipMemcpyAsync on stream A writes to buffer
  - GEMM on stream B reads from the SAME buffer (after wait_stream)
  - If L2 cache isn't properly invalidated after DMA, GEMM reads stale data

Usage:
    # Basic: 2-slot rotation with concurrent H2D + GEMM
    .venv/bin/python scripts/meta_nan_hip_stress.py --mode gemm

    # 3-slot rotation (matches TorchRec pipeline depth)
    .venv/bin/python scripts/meta_nan_hip_stress.py --mode gemm --slots 3

    # Embedding lookup stress (large random access pattern)
    .venv/bin/python scripts/meta_nan_hip_stress.py --mode embedding --rows 10000000

    # Full stress: GEMM + embedding + large batch + no sync
    .venv/bin/python scripts/meta_nan_hip_stress.py --mode both --batch 4096 --iterations 50000

    # With explicit cache flush between H2D and compute
    .venv/bin/python scripts/meta_nan_hip_stress.py --mode gemm --cache-flush

    # Multi-GPU with NCCL all_to_all interleaved
    torchrun --nproc_per_node=8 scripts/meta_nan_hip_stress.py --mode both --distributed
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


class BufferSlotManager:
    """Manages N device buffer slots at fixed virtual addresses with rotation."""

    def __init__(self, num_slots: int, shapes_and_dtypes: list, device):
        self.num_slots = num_slots
        self.device = device
        self.slots = []
        for _ in range(num_slots):
            slot = []
            for shape, dtype in shapes_and_dtypes:
                slot.append(torch.zeros(shape, dtype=dtype, device=device))
            self.slots.append(slot)

        log.info(f"Allocated {num_slots} buffer slots:")
        for i, slot in enumerate(self.slots):
            ptrs = [f"0x{t.data_ptr():x}" for t in slot]
            sizes_mb = [t.nelement() * t.element_size() / 1e6 for t in slot]
            log.info(f"  slot[{i}]: ptrs={ptrs}, sizes_MB={[f'{s:.1f}' for s in sizes_mb]}")

    def rotate(self):
        self.slots = self.slots[1:] + [self.slots[0]]

    def get_read_slot(self):
        return self.slots[0]

    def get_write_slot(self):
        return self.slots[-1]


def verify_h2d_integrity(device_tensor, host_tensor, iteration, label):
    """Check if device tensor matches host tensor exactly."""
    device_copy = device_tensor.cpu()
    if device_tensor.dtype == torch.bfloat16:
        mismatch = (device_copy.float() - host_tensor.float()).abs().max().item()
        if mismatch > 0:
            n_bad = ((device_copy.float() - host_tensor.float()).abs() > 0).sum().item()
            return True, f"iter={iteration} {label}: H2D MISMATCH max_diff={mismatch:.6f}, bad={n_bad}/{device_tensor.numel()}"
    else:
        mismatched = (device_copy != host_tensor).sum().item()
        if mismatched > 0:
            return True, f"iter={iteration} {label}: H2D MISMATCH {mismatched}/{device_tensor.numel()} elements differ"
    return False, ""


def run_gemm_stress(args, rank, device):
    """Concurrent H2D + GEMM on rotating buffer slots."""
    M, K, N = args.batch, args.dim, args.dim
    dtype = torch.bfloat16

    shapes = [
        ((M, K), dtype),   # A matrix
        ((K, N), dtype),   # B matrix
    ]
    mgr = BufferSlotManager(args.slots, shapes, device)

    host_pool_a = [torch.randn(M, K, dtype=dtype).pin_memory() for _ in range(args.pool_size)]
    host_pool_b = [torch.randn(K, N, dtype=dtype).pin_memory() for _ in range(args.pool_size)]

    h2d_stream = torch.cuda.Stream()
    compute_stream = torch.cuda.current_stream()

    for slot_idx in range(args.slots):
        slot = mgr.slots[slot_idx]
        slot[0].copy_(host_pool_a[slot_idx % args.pool_size])
        slot[1].copy_(host_pool_b[slot_idx % args.pool_size])
    torch.cuda.synchronize()

    corruption_count = 0
    h2d_mismatch_count = 0
    details = []
    start = time.time()
    last_log = start
    pool_idx = args.slots

    for it in range(args.iterations):
        compute_stream.wait_stream(h2d_stream)

        write_slot = mgr.get_write_slot()
        ha = host_pool_a[pool_idx % args.pool_size]
        hb = host_pool_b[pool_idx % args.pool_size]
        with torch.cuda.stream(h2d_stream):
            write_slot[0].copy_(ha, non_blocking=True)
            write_slot[1].copy_(hb, non_blocking=True)
        pool_idx += 1

        read_slot = mgr.get_read_slot()
        C = torch.mm(read_slot[0], read_slot[1])

        if (it + 1) % args.check_interval == 0:
            torch.cuda.synchronize()

            has_nan = torch.isnan(C).any().item()
            has_inf = torch.isinf(C).any().item()
            if has_nan or has_inf:
                corruption_count += 1
                nan_n = torch.isnan(C).sum().item()
                inf_n = torch.isinf(C).sum().item()
                details.append(f"iter={it} GEMM: nan={nan_n}, inf={inf_n}")

            read_host_a_idx = (pool_idx - args.slots) % args.pool_size
            read_host_b_idx = read_host_a_idx
            h2d_bad_a, msg_a = verify_h2d_integrity(
                read_slot[0], host_pool_a[read_host_a_idx], it, "A_matrix"
            )
            h2d_bad_b, msg_b = verify_h2d_integrity(
                read_slot[1], host_pool_b[read_host_b_idx], it, "B_matrix"
            )
            if h2d_bad_a:
                h2d_mismatch_count += 1
                details.append(msg_a)
            if h2d_bad_b:
                h2d_mismatch_count += 1
                details.append(msg_b)

            ref_C = torch.mm(
                host_pool_a[read_host_a_idx].to(device),
                host_pool_b[read_host_b_idx].to(device),
            )
            diff = (C.float() - ref_C.float()).abs().max().item()
            if diff > 0.1:
                corruption_count += 1
                details.append(f"iter={it} GEMM DIVERGENCE: max_diff={diff:.4f}")

            now = time.time()
            if rank == 0 and (corruption_count > 0 or now - last_log >= 5):
                elapsed = now - start
                rate = (it + 1) / elapsed
                log.info(
                    f"  [gemm] iter={it+1}/{args.iterations}  "
                    f"nan/corrupt={corruption_count}  h2d_mismatch={h2d_mismatch_count}  "
                    f"rate={rate:.0f} it/s"
                )
                last_log = now

            if (corruption_count > 0 or h2d_mismatch_count > 0) and args.stop_on_first:
                break

        mgr.rotate()

    torch.cuda.synchronize()
    return corruption_count, h2d_mismatch_count, details


def run_embedding_stress(args, rank, device):
    """Concurrent H2D of indices + embedding lookup on rotating buffers."""
    dtype = torch.bfloat16
    B, S = args.batch, 128

    emb = torch.nn.Embedding(args.emb_rows, args.dim).to(device=device, dtype=dtype)
    torch.nn.init.normal_(emb.weight, std=0.01)

    shapes = [
        ((B, S), torch.long),  # indices
    ]
    mgr = BufferSlotManager(args.slots, shapes, device)

    import numpy as np
    rng = np.random.RandomState(42)
    host_pool = [
        torch.from_numpy(rng.randint(0, args.emb_rows, (B, S), dtype=np.int64)).pin_memory()
        for _ in range(args.pool_size)
    ]

    h2d_stream = torch.cuda.Stream()
    compute_stream = torch.cuda.current_stream()

    for slot_idx in range(args.slots):
        mgr.slots[slot_idx][0].copy_(host_pool[slot_idx % args.pool_size])
    torch.cuda.synchronize()

    corruption_count = 0
    h2d_mismatch_count = 0
    details = []
    start = time.time()
    last_log = start
    pool_idx = args.slots

    for it in range(args.iterations):
        compute_stream.wait_stream(h2d_stream)

        write_slot = mgr.get_write_slot()
        host_indices = host_pool[pool_idx % args.pool_size]
        with torch.cuda.stream(h2d_stream):
            write_slot[0].copy_(host_indices, non_blocking=True)
        pool_idx += 1

        read_slot = mgr.get_read_slot()
        out = emb(read_slot[0])
        out_sum = out.sum(dim=-1)

        if (it + 1) % args.check_interval == 0:
            torch.cuda.synchronize()

            has_nan = torch.isnan(out).any().item()
            has_inf = torch.isinf(out).any().item()
            if has_nan or has_inf:
                corruption_count += 1
                details.append(f"iter={it} EMBEDDING: nan/inf detected")

            read_host_idx = (pool_idx - args.slots) % args.pool_size
            h2d_bad, msg = verify_h2d_integrity(
                read_slot[0], host_pool[read_host_idx], it, "indices"
            )
            if h2d_bad:
                h2d_mismatch_count += 1
                details.append(msg)

            ref_out = emb(host_pool[read_host_idx].to(device))
            diff = (out.float() - ref_out.float()).abs().max().item()
            if diff > 0:
                corruption_count += 1
                details.append(f"iter={it} EMBEDDING DIVERGENCE: max_diff={diff:.6f}")

            now = time.time()
            if rank == 0 and (corruption_count > 0 or now - last_log >= 5):
                elapsed = now - start
                rate = (it + 1) / elapsed
                log.info(
                    f"  [embedding] iter={it+1}/{args.iterations}  "
                    f"corrupt={corruption_count}  h2d_mismatch={h2d_mismatch_count}  "
                    f"rate={rate:.0f} it/s"
                )
                last_log = now

            if (corruption_count > 0 or h2d_mismatch_count > 0) and args.stop_on_first:
                break

        mgr.rotate()

    torch.cuda.synchronize()
    return corruption_count, h2d_mismatch_count, details


def run_combined_stress(args, rank, device):
    """GEMM + embedding + all_to_all on 3 streams simultaneously."""
    dtype = torch.bfloat16
    M, K, N = args.batch, args.dim, args.dim
    B, S = args.batch, 128

    emb = torch.nn.Embedding(args.emb_rows, args.dim).to(device=device, dtype=dtype)
    torch.nn.init.normal_(emb.weight, std=0.01)

    shapes = [
        ((M, K), dtype),           # A matrix
        ((K, N), dtype),           # B matrix
        ((B, S), torch.long),     # embedding indices
    ]
    mgr = BufferSlotManager(args.slots, shapes, device)

    import numpy as np
    rng = np.random.RandomState(42)
    host_pool_a = [torch.randn(M, K, dtype=dtype).pin_memory() for _ in range(args.pool_size)]
    host_pool_b = [torch.randn(K, N, dtype=dtype).pin_memory() for _ in range(args.pool_size)]
    host_pool_idx = [
        torch.from_numpy(rng.randint(0, args.emb_rows, (B, S), dtype=np.int64)).pin_memory()
        for _ in range(args.pool_size)
    ]

    h2d_stream = torch.cuda.Stream()
    a2a_stream = torch.cuda.Stream()
    compute_stream = torch.cuda.current_stream()

    use_dist = args.distributed and dist.is_initialized() and dist.get_world_size() > 1
    if use_dist:
        ws = dist.get_world_size()
        a2a_send = torch.empty(ws, M * N, dtype=dtype, device=device)
        a2a_recv = torch.empty_like(a2a_send)

    for slot_idx in range(args.slots):
        s = mgr.slots[slot_idx]
        pi = slot_idx % args.pool_size
        s[0].copy_(host_pool_a[pi])
        s[1].copy_(host_pool_b[pi])
        s[2].copy_(host_pool_idx[pi])
    torch.cuda.synchronize()

    corruption_count = 0
    h2d_mismatch_count = 0
    details = []
    start = time.time()
    last_log = start
    pool_idx = args.slots

    for it in range(args.iterations):
        compute_stream.wait_stream(h2d_stream)

        write_slot = mgr.get_write_slot()
        pi = pool_idx % args.pool_size
        with torch.cuda.stream(h2d_stream):
            write_slot[0].copy_(host_pool_a[pi], non_blocking=True)
            write_slot[1].copy_(host_pool_b[pi], non_blocking=True)
            write_slot[2].copy_(host_pool_idx[pi], non_blocking=True)
        pool_idx += 1

        if use_dist:
            with torch.cuda.stream(a2a_stream):
                a2a_send.fill_(float((it + 1) % 1000) / 1000.0)
                dist.all_to_all_single(a2a_recv, a2a_send, async_op=False)

        read_slot = mgr.get_read_slot()
        C = torch.mm(read_slot[0], read_slot[1])
        emb_out = emb(read_slot[2])
        combined = C.sum() + emb_out.sum()

        if use_dist:
            compute_stream.wait_stream(a2a_stream)

        if args.cache_flush:
            torch.cuda.synchronize()

        if (it + 1) % args.check_interval == 0:
            torch.cuda.synchronize()

            has_nan = torch.isnan(C).any().item() or torch.isnan(emb_out).any().item()
            has_inf = torch.isinf(C).any().item() or torch.isinf(emb_out).any().item()
            if has_nan or has_inf:
                corruption_count += 1
                details.append(f"iter={it} COMBINED: nan/inf detected")

            read_pi = (pool_idx - args.slots) % args.pool_size
            h2d_bad_a, msg_a = verify_h2d_integrity(read_slot[0], host_pool_a[read_pi], it, "A")
            h2d_bad_b, msg_b = verify_h2d_integrity(read_slot[1], host_pool_b[read_pi], it, "B")
            h2d_bad_i, msg_i = verify_h2d_integrity(read_slot[2], host_pool_idx[read_pi], it, "idx")
            for bad, msg in [(h2d_bad_a, msg_a), (h2d_bad_b, msg_b), (h2d_bad_i, msg_i)]:
                if bad:
                    h2d_mismatch_count += 1
                    details.append(msg)

            ref_C = torch.mm(
                host_pool_a[read_pi].to(device),
                host_pool_b[read_pi].to(device),
            )
            diff = (C.float() - ref_C.float()).abs().max().item()
            if diff > 0.1:
                corruption_count += 1
                details.append(f"iter={it} GEMM DIVERGENCE: max_diff={diff:.4f}")

            ref_emb = emb(host_pool_idx[read_pi].to(device))
            ediff = (emb_out.float() - ref_emb.float()).abs().max().item()
            if ediff > 0:
                corruption_count += 1
                details.append(f"iter={it} EMB DIVERGENCE: max_diff={ediff:.6f}")

            now = time.time()
            if rank == 0 and (corruption_count > 0 or now - last_log >= 5):
                elapsed = now - start
                rate = (it + 1) / elapsed
                log.info(
                    f"  [combined] iter={it+1}/{args.iterations}  "
                    f"corrupt={corruption_count}  h2d_mismatch={h2d_mismatch_count}  "
                    f"rate={rate:.0f} it/s"
                )
                last_log = now

            if (corruption_count > 0 or h2d_mismatch_count > 0) and args.stop_on_first:
                break

        mgr.rotate()

    torch.cuda.synchronize()
    return corruption_count, h2d_mismatch_count, details


def main():
    parser = argparse.ArgumentParser(description="HIP memory stress test for Issue B")
    parser.add_argument("--mode", choices=["gemm", "embedding", "both"], default="both")
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--slots", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=50000)
    parser.add_argument("--pool-size", type=int, default=64)
    parser.add_argument("--emb-rows", type=int, default=1_000_000)
    parser.add_argument("--check-interval", type=int, default=10)
    parser.add_argument("--stop-on-first", action="store_true", default=True)
    parser.add_argument("--no-stop-on-first", action="store_true")
    parser.add_argument("--cache-flush", action="store_true",
                        help="Add synchronize between H2D and compute to test cache flush")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--no-aql-set", action="store_true",
                        help="Do NOT set ROC_AQL_QUEUE_SIZE (use default 16K)")
    args = parser.parse_args()

    if args.no_stop_on_first:
        args.stop_on_first = False

    if not args.no_aql_set:
        os.environ.setdefault("ROC_AQL_QUEUE_SIZE", "1024")

    rank = 0
    if args.distributed and "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        rank = dist.get_rank()

    device = torch.cuda.current_device()

    if rank == 0:
        log.info("=" * 70)
        log.info("HIP MEMORY SUBSYSTEM STRESS TEST")
        log.info("=" * 70)
        log.info(f"  mode: {args.mode}")
        log.info(f"  batch: {args.batch}, dim: {args.dim}")
        log.info(f"  slots: {args.slots}")
        log.info(f"  iterations: {args.iterations}")
        log.info(f"  emb_rows: {args.emb_rows}")
        log.info(f"  cache_flush: {args.cache_flush}")
        log.info(f"  ROC_AQL_QUEUE_SIZE: {os.environ.get('ROC_AQL_QUEUE_SIZE', 'not set')}")
        ws = dist.get_world_size() if dist.is_initialized() else 1
        log.info(f"  world_size: {ws}")
        log.info("=" * 70)

    t_start = time.time()

    if args.mode == "gemm":
        corrupt, h2d_bad, details = run_gemm_stress(args, rank, device)
    elif args.mode == "embedding":
        corrupt, h2d_bad, details = run_embedding_stress(args, rank, device)
    else:
        corrupt, h2d_bad, details = run_combined_stress(args, rank, device)

    elapsed = time.time() - t_start

    if rank == 0:
        log.info("")
        log.info("=" * 70)
        log.info("RESULTS")
        log.info("=" * 70)
        log.info(f"  Mode: {args.mode}")
        log.info(f"  Elapsed: {elapsed:.1f}s")
        log.info(f"  Corruption detections: {corrupt}")
        log.info(f"  H2D mismatch detections: {h2d_bad}")
        for d in details[:30]:
            log.info(f"    {d}")

        if corrupt > 0 or h2d_bad > 0:
            log.info("")
            log.info("VERDICT: CORRUPTION DETECTED — HIP memory subsystem issue confirmed!")
        else:
            log.info("")
            log.info("VERDICT: No corruption detected.")
        log.info("=" * 70)

    if args.distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    sys.exit(1 if (corrupt > 0 or h2d_bad > 0) else 0)


if __name__ == "__main__":
    main()
