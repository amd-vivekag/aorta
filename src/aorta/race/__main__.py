"""
CLI entry point for the minimal RCCL race condition reproducer.

Usage:
    # Run with default settings (GPU_MAX_HW_QUEUES=4, warmup=100, verify=10000)
    GPU_MAX_HW_QUEUES=4 torchrun --nproc_per_node=8 -m aorta.race
    
    # Run with custom settings
    torchrun --nproc_per_node=8 -m aorta.race \
        --warmup 100 \
        --verify 10000 \
        --simulate-compute \
        --gemm-layers 20
    
    # Run in same-stream mode (replicates customer experiment 4)
    torchrun --nproc_per_node=8 -m aorta.race --same-stream
    
    # Run with settings that mask the bug (for comparison)
    GPU_MAX_HW_QUEUES=2 torchrun --nproc_per_node=8 -m aorta.race

Environment variables to test (from customer experiments):
    GPU_MAX_HW_QUEUES=4          # Exposes bug (use 2 to mask)
    ROC_SIGNAL_POOL_SIZE=16384   # Customer tried, didn't help
    HSA_ENABLE_SDMA=0            # Customer tried, didn't help
    GPU_FORCE_BLIT_COPY_SIZE=128 # Customer tried, didn't help
    NCCL_LAUNCH_ORDER_IMPLICIT=1 # No NaN but slow
    RCCL_GFX9_CHEAP_FENCE_OFF=1  # Customer tried, didn't help
"""

import argparse
import logging
import os
import sys

import torch
import torch.distributed as dist

from aorta.race.minimal_reproducer import ReproducerConfig, run_reproducer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Minimal RCCL race condition reproducer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Iteration settings
    parser.add_argument(
        "--warmup", type=int, default=100,
        help="Number of warmup iterations (no verification). Default: 100"
    )
    parser.add_argument(
        "--verify", type=int, default=10000,
        help="Number of verification iterations. Default: 10000"
    )
    parser.add_argument(
        "--no-stop-on-first", action="store_true",
        help="Don't stop on first corruption (continue counting)"
    )
    parser.add_argument(
        "--log-interval", type=int, default=100,
        help="Log progress every N iterations. Default: 100"
    )
    
    # Tensor sizes
    parser.add_argument(
        "--h2d-size", type=int, default=1_000_000,
        help="H2D tensor size (elements). Default: 1M"
    )
    parser.add_argument(
        "--a2a-size", type=int, default=100_000,
        help="all_to_all tensor size per rank (elements). Default: 100K"
    )
    parser.add_argument(
        "--ar-size", type=int, default=100_000,
        help="all_reduce tensor size (elements). Default: 100K"
    )
    parser.add_argument(
        "--dtype", type=str, default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Data type. Default: bfloat16"
    )
    
    # Compute simulation
    parser.add_argument(
        "--simulate-compute", action="store_true", default=True,
        help="Add GEMM work between collectives (default: enabled)"
    )
    parser.add_argument(
        "--no-compute", action="store_true",
        help="Disable compute simulation (fast but may not trigger bug)"
    )
    parser.add_argument(
        "--gemm-size", type=int, default=4096,
        help="GEMM matrix size. Default: 4096"
    )
    parser.add_argument(
        "--gemm-layers", type=int, default=20,
        help="Number of GEMM layers. Default: 20"
    )
    parser.add_argument(
        "--no-backward", action="store_true",
        help="Skip backward pass simulation"
    )
    
    # Stream configuration
    parser.add_argument(
        "--same-stream", action="store_true",
        help="Put H2D and datadist on same stream (replicates experiment 4)"
    )
    
    # Hardware settings
    parser.add_argument(
        "--hw-queues", type=int, default=None,
        help="Set GPU_MAX_HW_QUEUES (4 exposes bug, 2 masks it)"
    )
    
    return parser.parse_args()


def init_distributed() -> tuple[int, int]:
    """Initialize distributed process group."""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Set device
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    
    return rank, world_size


def main():
    """Main entry point."""
    args = parse_args()
    
    # Initialize distributed
    rank, world_size = init_distributed()
    
    if rank == 0:
        log.info("=" * 70)
        log.info("MINIMAL RCCL RACE CONDITION REPRODUCER")
        log.info("=" * 70)
        log.info("")
        log.info("This reproducer uses PROPER SYNCHRONIZATION everywhere.")
        log.info("If corruption occurs, it indicates a RUNTIME BUG in RCCL/HIP,")
        log.info("not an application-level issue.")
        log.info("")
        log.info(f"World size: {world_size}")
        log.info(f"Warmup iterations: {args.warmup}")
        log.info(f"Verify iterations: {args.verify}")
        log.info(f"Simulate compute: {args.simulate_compute and not args.no_compute}")
        log.info(f"Same stream mode: {args.same_stream}")
        log.info("")
        
        # Log relevant env vars
        env_vars = [
            "GPU_MAX_HW_QUEUES",
            "ROC_SIGNAL_POOL_SIZE",
            "HSA_ENABLE_SDMA",
            "GPU_FORCE_BLIT_COPY_SIZE",
            "NCCL_LAUNCH_ORDER_IMPLICIT",
            "RCCL_GFX9_CHEAP_FENCE_OFF",
            "RCCL_GFX942_CHEAP_FENCE_OFF",
        ]
        log.info("Environment variables:")
        for var in env_vars:
            value = os.environ.get(var, "(not set)")
            log.info(f"  {var}={value}")
        log.info("")
        log.info("=" * 70)
    
    # Build config
    config = ReproducerConfig(
        warmup_iterations=args.warmup,
        verify_iterations=args.verify,
        stop_on_first_corruption=not args.no_stop_on_first,
        log_interval=args.log_interval,
        h2d_tensor_size=args.h2d_size,
        alltoall_tensor_size=args.a2a_size,
        allreduce_tensor_size=args.ar_size,
        dtype=args.dtype,
        simulate_compute=args.simulate_compute and not args.no_compute,
        gemm_size=args.gemm_size,
        gemm_layers=args.gemm_layers,
        include_backward_compute=not args.no_backward,
        same_stream_mode=args.same_stream,
        gpu_max_hw_queues=args.hw_queues,
    )
    
    # Run reproducer
    result = run_reproducer(config, rank, world_size)
    
    # Sync all ranks
    dist.barrier()
    
    # Report results (rank 0 only)
    if rank == 0:
        log.info("")
        log.info("=" * 70)
        log.info("RESULTS")
        log.info("=" * 70)
        log.info(f"Passed: {result.passed}")
        log.info(f"Total iterations: {result.total_iterations}")
        log.info(f"Corruption count: {result.corruption_count}")
        log.info(f"Elapsed time: {result.elapsed_time_sec:.2f}s")
        log.info(f"Avg step time: {result.avg_step_time_ms:.2f}ms")
        
        if result.first_corruption_iter is not None:
            log.info(f"First corruption at iteration: {result.first_corruption_iter}")
        
        if result.passed:
            log.info("")
            log.info("VERDICT: No runtime bug detected with current settings.")
            log.info("If client still sees corruption, their bug is likely")
            log.info("application-level (missing syncs in TorchRec).")
        else:
            log.info("")
            log.info("VERDICT: RUNTIME BUG DETECTED!")
            log.info("Corruption occurred DESPITE proper synchronization.")
            log.info("This is a bug in RCCL/HIP runtime - report to AMD.")
        
        log.info("=" * 70)
    
    # Exit with appropriate code
    sys.exit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
