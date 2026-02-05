"""
RCCL Runtime Race Condition Reproducer.

This module provides a minimal reproducer for detecting runtime-level race
conditions in RCCL/HIP during multi-stream distributed training. The reproducer:

1. Uses PROPER synchronization everywhere - if corruption occurs, it's a RUNTIME BUG
2. Uses known-pattern data to detect ANY data corruption
3. Simulates training timing profile with interleaved compute
4. Supports warmup phase to build up RCCL/runtime state
5. Creates REAL data dependencies between streams

Data Flow Architecture (with compute simulation):
  memcpy_stream:  [H2D] → batch_gpu
                            ↓ (Forward READS batch_gpu - DATA DEPENDENCY)
  default_stream:          [Forward] → [Backward] → [all_reduce]
                                         (same stream, naturally ordered)

  datadist_stream:         [all_to_all]
                            (overlaps with backward - TIMING PRESSURE)

  All buffers use known patterns for verification:
    - batch_gpu: iteration % 1000
    - send_buf/recv_buf: rank ID
    - reduce_buf: rank + 1

Key insight: If corruption occurs despite proper synchronization, it indicates
a bug in RCCL/HIP runtime, not application-level missing syncs.
"""

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

log = logging.getLogger(__name__)


@dataclass
class ReproducerConfig:
    """
    Configuration for the minimal RCCL race condition reproducer.
    
    This reproducer tests for RUNTIME bugs (not application bugs) by:
    - Using proper synchronization everywhere
    - Using known-pattern data to detect any corruption
    - Simulating training timing profile
    
    If corruption occurs with proper syncs, it's a RUNTIME BUG in RCCL/HIP.
    """
    
    # =========================================================================
    # Iteration settings
    # =========================================================================
    warmup_iterations: int = 100
    """
    Number of iterations to run WITHOUT verification.
    
    Builds up RCCL/runtime state before checking for corruption.
    Runtime bugs may manifest after ~100+ steps due to internal state
    accumulation. Warmup helps reach similar conditions.
    """
    
    verify_iterations: int = 10000
    """
    Number of iterations to run WITH verification.
    
    After warmup, we check every iteration for corruption using
    known-pattern data verification.
    """
    
    stop_on_first_corruption: bool = True
    """Stop immediately when corruption is detected."""
    
    log_interval: int = 100
    """Log progress every N iterations."""
    
    # =========================================================================
    # Tensor sizes (stress factors)
    # =========================================================================
    h2d_tensor_size: int = 1_000_000
    """
    Size of H2D tensor (number of elements).
    
    Larger = longer DMA = more overlap opportunity.
    Recommended: 1M+ for stress testing.
    """
    
    alltoall_tensor_size: int = 100_000
    """
    Size of all_to_all tensor per rank (number of elements).
    
    Total transfer size = alltoall_tensor_size * world_size.
    Larger = longer collective = more overlap opportunity.
    """
    
    allreduce_tensor_size: int = 100_000
    """
    Size of all_reduce tensor (number of elements).
    
    Simulates gradient synchronization (e.g., FSDP-style).
    """
    
    dtype: str = "bfloat16"
    """Data type for tensors. Options: bfloat16, float16, float32."""
    
    # =========================================================================
    # Compute simulation (match training timing)
    # =========================================================================
    simulate_compute: bool = True
    """
    Add GEMM work between collectives to match training timing.
    
    Without this, the reproducer runs ~100x faster than real training,
    which may not trigger the same timing conditions.
    """
    
    gemm_size: int = 5120
    """
    Matrix size for GEMM operations.
    
    5120x5120 GEMM takes ~14ms on MI300X (1.95x compute vs 4096).
    Adjust based on GPU to achieve ~500ms/step target.
    """
    
    gemm_layers: int = 26
    """
    Number of GEMM layers to simulate forward/backward.
    
    26 layers × 14ms = ~364ms per forward/backward pass.
    With forward + backward: ~500ms/step (configurable timing profile).
    """
    
    include_backward_compute: bool = True
    """Also simulate backward pass GEMMs (doubles compute time)."""
    
    # =========================================================================
    # Buffer management
    # =========================================================================
    reuse_buffers: bool = True
    """
    Reuse tensor buffers across iterations.
    
    Real training reuses buffers; fresh allocations each iteration
    may change memory layout and timing.
    """
    
    pin_memory: bool = True
    """Use pinned memory for H2D source tensors."""
    
    # =========================================================================
    # Stream configuration
    # =========================================================================
    same_stream_mode: bool = False
    """
    Put H2D and datadist on the SAME new stream (not separate streams).
    
    If corruption occurs in this mode, it's DEFINITIVE proof of a 
    runtime bug because operations on the same stream are guaranteed 
    to be ordered by CUDA/HIP specification.
    """
    
    # =========================================================================
    # Environment variables
    # =========================================================================
    gpu_max_hw_queues: Optional[int] = 4
    """
    Set GPU_MAX_HW_QUEUES environment variable.
    
    - 2: Reduced parallelism (may mask bugs)
    - 4+: Full parallelism (exposes timing-sensitive bugs)
    """
    
    # Additional env vars can be set via CLI or environment:
    # ROC_SIGNAL_POOL_SIZE, HSA_ENABLE_SDMA, GPU_FORCE_BLIT_COPY_SIZE,
    # NCCL_LAUNCH_ORDER_IMPLICIT, RCCL_GFX9_CHEAP_FENCE_OFF, etc.


@dataclass
class ReproducerResult:
    """Result from running the reproducer."""
    
    passed: bool
    """True if no corruption was detected."""
    
    total_iterations: int
    """Total iterations run (warmup + verify)."""
    
    corruption_count: int
    """Number of corruptions detected."""
    
    first_corruption_iter: Optional[int]
    """Iteration where first corruption was detected (None if passed)."""
    
    corruption_details: List[Dict]
    """Details of each corruption detected."""
    
    elapsed_time_sec: float
    """Total elapsed time in seconds."""
    
    avg_step_time_ms: float
    """Average time per step in milliseconds."""


class MinimalReproducer:
    """
    Minimal reproducer for RCCL/HIP runtime race conditions.
    
    This class implements a 3-stream pattern with PROPER synchronization:
    - memcpy_stream: H2D data transfers
    - datadist_stream: all_to_all collectives
    - default_stream: compute + all_reduce
    
    If corruption occurs despite proper syncs, it indicates a RUNTIME BUG
    in RCCL/HIP, not an application-level issue.
    
    Usage:
        config = ReproducerConfig(warmup_iterations=100, verify_iterations=10000)
        reproducer = MinimalReproducer(config, rank=0, world_size=8)
        result = reproducer.run()
        
        if not result.passed:
            print("RUNTIME BUG DETECTED!")
    """
    
    def __init__(self, config: ReproducerConfig, rank: int, world_size: int):
        """
        Initialize the reproducer.
        
        Args:
            config: Reproducer configuration.
            rank: Current process rank.
            world_size: Total number of processes.
        """
        self.config = config
        self.rank = rank
        self.world_size = world_size
        
        # Streams
        self.memcpy_stream: Optional[torch.cuda.Stream] = None
        self.datadist_stream: Optional[torch.cuda.Stream] = None
        self.default_stream: Optional[torch.cuda.Stream] = None
        
        # Buffers (allocated in setup())
        self.batch_cpu: Optional[torch.Tensor] = None
        self.batch_gpu: Optional[torch.Tensor] = None
        self.send_buf: Optional[torch.Tensor] = None
        self.recv_buf: Optional[torch.Tensor] = None
        self.reduce_buf: Optional[torch.Tensor] = None
        
        # Compute simulation buffers
        self.weight_matrices: List[torch.Tensor] = []
        self.activation_buffer: Optional[torch.Tensor] = None
        self.grad_buffer: Optional[torch.Tensor] = None
        
        # State
        self.in_verification_phase: bool = False
        self.corruption_details: List[Dict] = []
        
        # Get dtype
        self.dtype = self._get_dtype()
    
    def _get_dtype(self) -> torch.dtype:
        """Get torch dtype from config string."""
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        return dtype_map.get(self.config.dtype, torch.bfloat16)
    
    def setup(self) -> None:
        """Allocate streams and buffers."""
        cfg = self.config

        # Set GPU_MAX_HW_QUEUES if specified
        if cfg.gpu_max_hw_queues is not None:
            os.environ["GPU_MAX_HW_QUEUES"] = str(cfg.gpu_max_hw_queues)
            log.info(f"Set GPU_MAX_HW_QUEUES={cfg.gpu_max_hw_queues}")

        # Create streams
        if cfg.same_stream_mode:
            # Same stream for H2D and datadist (definitive runtime bug test)
            shared_stream = torch.cuda.Stream()
            self.memcpy_stream = shared_stream
            self.datadist_stream = shared_stream
            log.info("Using SAME stream for H2D and datadist (experiment 4 mode)")
        else:
            self.memcpy_stream = torch.cuda.Stream()
            self.datadist_stream = torch.cuda.Stream()
            log.info("Using separate streams for H2D and datadist")

        self.default_stream = torch.cuda.current_stream()

        # Validate buffer sizes for compute simulation
        if cfg.simulate_compute:
            min_h2d_size = cfg.gemm_size * cfg.gemm_size
            if cfg.h2d_tensor_size < min_h2d_size:
                log.warning(
                    f"h2d_tensor_size ({cfg.h2d_tensor_size}) < gemm_size² ({min_h2d_size}). "
                    f"Increasing to {min_h2d_size} for compute simulation."
                )
                cfg.h2d_tensor_size = min_h2d_size

        # Allocate H2D buffers
        if cfg.pin_memory:
            self.batch_cpu = torch.empty(
                cfg.h2d_tensor_size, dtype=self.dtype, pin_memory=True
            )
        else:
            self.batch_cpu = torch.empty(cfg.h2d_tensor_size, dtype=self.dtype)

        self.batch_gpu = torch.empty(
            cfg.h2d_tensor_size, dtype=self.dtype, device="cuda"
        )
        
        # Allocate all_to_all buffers
        self.send_buf = torch.empty(
            self.world_size, cfg.alltoall_tensor_size,
            dtype=self.dtype, device="cuda"
        )
        self.recv_buf = torch.empty_like(self.send_buf)
        
        # Allocate all_reduce buffer
        self.reduce_buf = torch.empty(
            cfg.allreduce_tensor_size, dtype=self.dtype, device="cuda"
        )
        
        # Allocate compute simulation buffers
        if cfg.simulate_compute:
            self.weight_matrices = [
                torch.randn(cfg.gemm_size, cfg.gemm_size, dtype=self.dtype, device="cuda")
                for _ in range(cfg.gemm_layers)
            ]
            self.activation_buffer = torch.randn(
                cfg.gemm_size, cfg.gemm_size, dtype=self.dtype, device="cuda"
            )
            self.grad_buffer = torch.randn(
                cfg.gemm_size, cfg.gemm_size, dtype=self.dtype, device="cuda"
            )
        
        log.info(
            f"Reproducer setup complete: rank={self.rank}, world_size={self.world_size}, "
            f"h2d_size={cfg.h2d_tensor_size}, a2a_size={cfg.alltoall_tensor_size}, "
            f"ar_size={cfg.allreduce_tensor_size}, dtype={cfg.dtype}"
        )
    
    def _fill_known_patterns(self, iteration: int) -> None:
        """Fill buffers with known patterns for verification."""
        # H2D: batch = iteration % 1000 (avoid overflow in bfloat16)
        self.batch_cpu.fill_(float(iteration % 1000))
        
        # all_to_all: send_buf[i] = rank for all i
        self.send_buf.fill_(float(self.rank))
        
        # all_reduce: reduce_buf = rank + 1
        self.reduce_buf.fill_(float(self.rank + 1))
    
    def _run_h2d(self) -> None:
        """Run H2D transfer on memcpy_stream."""
        with torch.cuda.stream(self.memcpy_stream):
            self.batch_gpu.copy_(self.batch_cpu, non_blocking=True)
    
    def _run_forward_compute(self) -> None:
        """Simulate forward pass with GEMMs on default stream."""
        if not self.config.simulate_compute:
            return

        # CRITICAL: Create real data dependency on H2D transfer
        # Forward MUST read batch_gpu to create H2D→Forward race opportunity
        # Reshape batch_gpu slice to match activation size (simulates embedding output)
        batch_slice = self.batch_gpu[:self.config.gemm_size * self.config.gemm_size]
        batch_reshaped = batch_slice.view(self.config.gemm_size, self.config.gemm_size)

        # Start forward pass with batch data
        x = batch_reshaped
        for layer_idx in range(self.config.gemm_layers):
            x = torch.mm(self.weight_matrices[layer_idx], x)
            x = torch.nn.functional.gelu(x)

        # Store result to prevent optimization
        self.activation_buffer = x
    
    def _run_alltoall(self) -> torch.distributed.Work:
        """Run all_to_all on datadist_stream."""
        with torch.cuda.stream(self.datadist_stream):
            work = dist.all_to_all_single(
                self.recv_buf, self.send_buf, async_op=True
            )
        return work
    
    def _run_backward_compute(self) -> None:
        """Simulate backward pass with GEMMs on default stream."""
        if not self.config.simulate_compute or not self.config.include_backward_compute:
            return

        grad = self.grad_buffer
        for layer_idx in reversed(range(self.config.gemm_layers)):
            grad = torch.mm(self.weight_matrices[layer_idx].T, grad)

        # Store result to prevent optimization
        self.grad_buffer = grad

        # NOTE: We do NOT write grad to reduce_buf because:
        # 1. Chained GEMMs produce numerically unstable gradients
        # 2. This would overwrite the known pattern needed for verification
        # 3. Backward and all_reduce are on same stream (naturally ordered)
        #
        # The critical race opportunity is H2D→Forward (different streams)
        # not Backward→all_reduce (same stream)
    
    def _run_allreduce(self) -> None:
        """Run all_reduce on default stream."""
        dist.all_reduce(self.reduce_buf)
    
    def _verify_patterns(self, iteration: int) -> bool:
        """
        Verify all buffers contain expected patterns.
        
        Returns True if all patterns correct, False if corruption detected.
        """
        all_correct = True
        
        # Check H2D result
        expected_h2d = float(iteration % 1000)
        h2d_expected = torch.full_like(self.batch_gpu, expected_h2d)
        if not torch.allclose(self.batch_gpu, h2d_expected, rtol=1e-3, atol=1e-3):
            actual = self.batch_gpu[0].item()
            log.error(
                f"H2D CORRUPTION (RUNTIME BUG!): iter={iteration} rank={self.rank} "
                f"expected={expected_h2d} actual={actual}"
            )
            self.corruption_details.append({
                "type": "h2d",
                "iteration": iteration,
                "rank": self.rank,
                "expected": expected_h2d,
                "actual": actual,
            })
            all_correct = False
        
        # Check all_to_all result: recv_buf[j] should be all j's
        for src_rank in range(self.world_size):
            expected_a2a = float(src_rank)
            a2a_expected = torch.full_like(self.recv_buf[src_rank], expected_a2a)
            if not torch.allclose(self.recv_buf[src_rank], a2a_expected, rtol=1e-3, atol=1e-3):
                actual = self.recv_buf[src_rank, 0].item()
                log.error(
                    f"ALL_TO_ALL CORRUPTION (RUNTIME BUG!): iter={iteration} "
                    f"rank={self.rank} src_rank={src_rank} expected={expected_a2a} actual={actual}"
                )
                self.corruption_details.append({
                    "type": "all_to_all",
                    "iteration": iteration,
                    "rank": self.rank,
                    "src_rank": src_rank,
                    "expected": expected_a2a,
                    "actual": actual,
                })
                all_correct = False
        
        # Check all_reduce result: should be sum(1..world_size)
        expected_ar = float(sum(range(1, self.world_size + 1)))
        ar_expected = torch.full_like(self.reduce_buf, expected_ar)
        if not torch.allclose(self.reduce_buf, ar_expected, rtol=1e-3, atol=1e-3):
            actual = self.reduce_buf[0].item()
            log.error(
                f"ALL_REDUCE CORRUPTION (RUNTIME BUG!): iter={iteration} "
                f"rank={self.rank} expected={expected_ar} actual={actual}"
            )
            self.corruption_details.append({
                "type": "all_reduce",
                "iteration": iteration,
                "rank": self.rank,
                "expected": expected_ar,
                "actual": actual,
            })
            all_correct = False
        
        return all_correct
    
    def run_iteration(self, iteration: int) -> bool:
        """
        Run one iteration of the reproducer.

        Uses PROPER synchronization everywhere. If corruption occurs,
        it's a RUNTIME BUG in RCCL/HIP.

        Data Flow Chain (with compute simulation):
          memcpy_stream:  [H2D] → batch_gpu
                                    ↓ (Forward READS - DATA DEPENDENCY)
          default_stream:          [Forward READS batch_gpu]
                                    ↓
                                   [Backward] (computes but doesn't write to reduce_buf)
                                    ↓
                                   [all_reduce READS reduce_buf (known pattern)]

          datadist_stream:         [all_to_all]
                                    (overlaps with backward)

        Note: reduce_buf maintains its known pattern (rank+1) throughout.
        Backward does NOT overwrite it to avoid numerical instability from GEMMs.

        Returns True if patterns verified correctly (or not in verification phase).
        """
        # Fill buffers with known patterns
        self._fill_known_patterns(iteration)

        # ─────────────────────────────────────────────────────────────
        # Phase 1: H2D on memcpy_stream
        #   Writes: batch_gpu
        # ─────────────────────────────────────────────────────────────
        self._run_h2d()

        # ─────────────────────────────────────────────────────────────
        # Phase 2: PROPER SYNC - wait for H2D
        #   Ensures: batch_gpu ready before forward reads it
        # ─────────────────────────────────────────────────────────────
        self.default_stream.wait_stream(self.memcpy_stream)

        # ─────────────────────────────────────────────────────────────
        # Phase 3: Simulate forward pass (if enabled)
        #   Reads: batch_gpu (created by H2D)
        #   Creates: H2D→Forward data dependency
        # ─────────────────────────────────────────────────────────────
        self._run_forward_compute()

        # ─────────────────────────────────────────────────────────────
        # Phase 4: all_to_all on datadist_stream
        #   Overlaps with backward (timing pressure)
        # ─────────────────────────────────────────────────────────────
        work_a2a = self._run_alltoall()

        # ─────────────────────────────────────────────────────────────
        # Phase 5: Simulate backward pass (if enabled)
        #   Adds compute work to match training timing profile
        #   Does NOT write to reduce_buf (keeps known pattern)
        # ─────────────────────────────────────────────────────────────
        self._run_backward_compute()

        # ─────────────────────────────────────────────────────────────
        # Phase 6: PROPER SYNC - wait for all_to_all
        #   Ensures: all_to_all completes before all_reduce
        # ─────────────────────────────────────────────────────────────
        self.default_stream.wait_stream(self.datadist_stream)
        work_a2a.wait()

        # ─────────────────────────────────────────────────────────────
        # Phase 7: all_reduce on default stream
        #   Reads: reduce_buf (known pattern: rank+1)
        # ─────────────────────────────────────────────────────────────
        self._run_allreduce()

        # ─────────────────────────────────────────────────────────────
        # Phase 8: Verify patterns (only during verification phase)
        #   Checks all buffers for corruption:
        #     - batch_gpu: H2D corruption (if H2D raced with forward)
        #     - recv_buf: all_to_all corruption
        #     - reduce_buf: all_reduce corruption (should be sum of rank+1)
        #
        #   Note: reduce_buf was NOT modified by backward, so any corruption
        #   indicates an all_reduce bug, not gradient instability.
        # ─────────────────────────────────────────────────────────────
        if self.in_verification_phase:
            torch.cuda.synchronize()
            return self._verify_patterns(iteration)

        return True
    
    def run(self) -> ReproducerResult:
        """
        Run the full reproducer: warmup + verification.
        
        Returns:
            ReproducerResult with pass/fail status and details.
        """
        cfg = self.config
        
        # Setup
        self.setup()
        
        start_time = time.time()
        corruption_count = 0
        first_corruption_iter = None
        total_iterations = 0
        
        # ─────────────────────────────────────────────────────────────
        # Phase 1: Warmup (no verification)
        # ─────────────────────────────────────────────────────────────
        self.in_verification_phase = False
        log.info(f"Starting warmup phase: {cfg.warmup_iterations} iterations")

        warmup_start = time.time()
        for i in range(cfg.warmup_iterations):
            iter_start = time.time()

            self.run_iteration(i)

            # CRITICAL: Synchronize during warmup to actually execute GPU work
            # Without this, iterations just queue work and finish instantly
            torch.cuda.synchronize()

            total_iterations += 1
            iter_elapsed = time.time() - iter_start

            if (i + 1) % cfg.log_interval == 0:
                warmup_elapsed = time.time() - warmup_start
                avg_iter_ms = (warmup_elapsed * 1000) / (i + 1)
                log.info(
                    f"Warmup progress: {i + 1}/{cfg.warmup_iterations}, "
                    f"avg_step_time: {avg_iter_ms:.1f}ms"
                )

        warmup_total = time.time() - warmup_start
        warmup_avg_ms = (warmup_total * 1000) / cfg.warmup_iterations if cfg.warmup_iterations > 0 else 0
        log.info(
            f"Warmup complete: {warmup_total:.1f}s total, "
            f"{warmup_avg_ms:.1f}ms avg per step"
        )

        # Warn if timing is too fast (may not trigger timing-sensitive bugs)
        if cfg.simulate_compute and warmup_avg_ms < 400:
            log.warning(
                f"Step time ({warmup_avg_ms:.1f}ms) is faster than target (~500ms/step). "
                f"Consider increasing gemm_size or gemm_layers to match timing profile."
            )
        
        # ─────────────────────────────────────────────────────────────
        # Phase 2: Verification
        # ─────────────────────────────────────────────────────────────
        self.in_verification_phase = True
        
        for i in range(cfg.verify_iterations):
            iteration = cfg.warmup_iterations + i
            
            passed = self.run_iteration(iteration)
            total_iterations += 1
            
            if not passed:
                corruption_count += 1
                if first_corruption_iter is None:
                    first_corruption_iter = iteration
                
                if cfg.stop_on_first_corruption:
                    log.error(
                        f"Stopping on first corruption at iteration {iteration}"
                    )
                    break
            
            if (i + 1) % cfg.log_interval == 0:
                log.info(
                    f"Verification progress: {i + 1}/{cfg.verify_iterations}, "
                    f"corruptions={corruption_count}"
                )
        
        elapsed = time.time() - start_time
        avg_step_ms = (elapsed * 1000) / total_iterations if total_iterations > 0 else 0
        
        # Summary
        if corruption_count > 0:
            log.error(
                f"RUNTIME BUG DETECTED: {corruption_count} corruptions in "
                f"{total_iterations} iterations"
            )
            log.error(
                "Corruption occurred DESPITE proper synchronization - "
                "this is a bug in RCCL/HIP runtime"
            )
        else:
            log.info(
                f"PASSED: No corruption in {total_iterations} iterations "
                f"with proper synchronization"
            )
            log.info(
                "If corruption still occurs in real workloads, check for "
                "application-level synchronization issues."
            )
        
        return ReproducerResult(
            passed=(corruption_count == 0),
            total_iterations=total_iterations,
            corruption_count=corruption_count,
            first_corruption_iter=first_corruption_iter,
            corruption_details=self.corruption_details,
            elapsed_time_sec=elapsed,
            avg_step_time_ms=avg_step_ms,
        )


def run_reproducer(
    config: Optional[ReproducerConfig] = None,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
) -> ReproducerResult:
    """
    Run the minimal RCCL race condition reproducer.
    
    This is the main entry point for running the reproducer.
    
    Args:
        config: Reproducer configuration. If None, uses defaults.
        rank: Process rank. If None, reads from dist.get_rank().
        world_size: World size. If None, reads from dist.get_world_size().
    
    Returns:
        ReproducerResult with pass/fail status and details.
    """
    if config is None:
        config = ReproducerConfig()
    
    if rank is None:
        rank = dist.get_rank()
    
    if world_size is None:
        world_size = dist.get_world_size()
    
    reproducer = MinimalReproducer(config, rank, world_size)
    return reproducer.run()


__all__ = [
    "ReproducerConfig",
    "ReproducerResult",
    "MinimalReproducer",
    "run_reproducer",
]
