"""
DDP mode reproducer (Distributed Data Parallel pattern).

This mode simulates a DDP-style workload with:
- H2D transfer with double buffering for prefetch
- Forward/backward compute with GEMMs
- Gradient all_reduce (no all_to_all)
- Cross-rank gradient consistency verification

Data Flow (with H2D prefetch):
    Iteration N:
        memcpy_stream:  [H2D batch_N+1] ────────────────────────┐
                                                                │ (prefetch overlaps)
        default_stream: [Forward(batch_N)] → [Backward] → [all_reduce grads]
                                                                │
                        ← swap buffers ─────────────────────────┘
"""

import logging
from typing import Optional

import torch
import torch.distributed as dist

from ..base import BaseReproducer
from ..compute import GEMMCompute
from ..config import ReproducerConfig

log = logging.getLogger(__name__)


class DDPModeReproducer(BaseReproducer):
    """
    DDP reproducer with gradient all_reduce and H2D prefetch.
    
    This mode tests the 2-stream pattern common in DDP training:
    - memcpy_stream: H2D data transfers (with prefetch)
    - default_stream: compute + gradient all_reduce
    
    Key features:
    - Double buffering: overlaps next batch H2D with current compute
    - Deterministic: same seed across ranks for gradient verification
    - Gradient sync: all_reduce on actual computed gradients
    
    Verification checks:
    - H2D: batch_gpu == iteration % 1000
    - Gradient consistency: all ranks have identical gradient checksums
    """
    
    def __init__(self, config: ReproducerConfig, rank: int, world_size: int):
        super().__init__(config, rank, world_size)
        
        # Double buffers for H2D prefetch
        self.batch_cpu_current: Optional[torch.Tensor] = None
        self.batch_cpu_next: Optional[torch.Tensor] = None
        self.batch_gpu_current: Optional[torch.Tensor] = None
        self.batch_gpu_next: Optional[torch.Tensor] = None
    
    def setup_buffers(self) -> None:
        """Allocate double buffers for DDP mode with prefetch."""
        cfg = self.config
        
        # Double buffers for H2D prefetch
        if cfg.pin_memory:
            self.batch_cpu_current = torch.empty(
                cfg.h2d_tensor_size, dtype=self.dtype, pin_memory=True
            )
            self.batch_cpu_next = torch.empty(
                cfg.h2d_tensor_size, dtype=self.dtype, pin_memory=True
            )
        else:
            self.batch_cpu_current = torch.empty(
                cfg.h2d_tensor_size, dtype=self.dtype
            )
            self.batch_cpu_next = torch.empty(
                cfg.h2d_tensor_size, dtype=self.dtype
            )
        
        self.batch_gpu_current = torch.empty(
            cfg.h2d_tensor_size, dtype=self.dtype, device="cuda"
        )
        self.batch_gpu_next = torch.empty(
            cfg.h2d_tensor_size, dtype=self.dtype, device="cuda"
        )
        
        log.info(f"Allocated DDP double buffers for H2D prefetch (size={cfg.h2d_tensor_size})")
    
    def _prefetch_next_batch(self, next_iteration: int) -> None:
        """
        Start H2D for next batch on memcpy_stream (overlaps with compute).
        
        Args:
            next_iteration: The iteration number for the next batch.
        """
        # Fill next CPU buffer with known pattern
        self.batch_cpu_next.fill_(float(next_iteration % 1000))
        
        # Async copy to next GPU buffer
        with torch.cuda.stream(self.memcpy_stream):
            self.batch_gpu_next.copy_(self.batch_cpu_next, non_blocking=True)
    
    def _transfer_first_batch(self, iteration: int) -> None:
        """Transfer the first batch (no previous prefetch to rely on)."""
        self.batch_cpu_current.fill_(float(iteration % 1000))
        with torch.cuda.stream(self.memcpy_stream):
            self.batch_gpu_current.copy_(self.batch_cpu_current, non_blocking=True)
    
    def _gradient_allreduce(self) -> None:
        """
        All-reduce actual gradients (DDP-style).
        
        Averages gradients across all ranks by summing and dividing by world_size.
        """
        if self.compute is None:
            return
        
        for param in self.compute.parameters:
            if param.grad is not None:
                # Sum gradients across ranks
                dist.all_reduce(param.grad)
                # Average (DDP default behavior)
                param.grad.div_(self.world_size)
    
    def _swap_buffers(self) -> None:
        """Swap current and next buffers for the next iteration."""
        self.batch_gpu_current, self.batch_gpu_next = (
            self.batch_gpu_next, self.batch_gpu_current
        )
        self.batch_cpu_current, self.batch_cpu_next = (
            self.batch_cpu_next, self.batch_cpu_current
        )
    
    def run_iteration(self, iteration: int) -> bool:
        """
        Run one iteration of DDP mode with H2D prefetch.

        Data Flow:
          default_stream:  [Forward(batch_N)] → [Backward] → [all_reduce grads]
          memcpy_stream:                       [H2D batch_N+1 (prefetch)]
                                                ↑ overlaps with backward

        Returns True if patterns verified correctly (or not in verification phase).
        """
        # ─────────────────────────────────────────────────────────────
        # Phase 1: Ensure current batch is ready
        # First iteration needs explicit H2D since there's no previous prefetch
        # ─────────────────────────────────────────────────────────────
        if iteration == 0:
            self._transfer_first_batch(iteration)
        
        # Wait for current batch H2D to complete
        self.default_stream.wait_stream(self.memcpy_stream)
        
        # ─────────────────────────────────────────────────────────────
        # Phase 2: Forward pass (uses current batch)
        # ─────────────────────────────────────────────────────────────
        forward_output = None
        if self.compute:
            forward_output = self.compute.forward(self.batch_gpu_current)
        
        # ─────────────────────────────────────────────────────────────
        # Phase 3: Start prefetching NEXT batch (overlaps with backward)
        # ─────────────────────────────────────────────────────────────
        self._prefetch_next_batch(iteration + 1)
        
        # ─────────────────────────────────────────────────────────────
        # Phase 4: Backward pass (computes gradients)
        # ─────────────────────────────────────────────────────────────
        if self.compute:
            self.compute.backward(
                forward_output, 
                use_autograd=(self.optimizer is not None)
            )
        
        # ─────────────────────────────────────────────────────────────
        # Phase 5: DDP gradient sync (all_reduce on actual gradients)
        # ─────────────────────────────────────────────────────────────
        self._gradient_allreduce()
        
        # ─────────────────────────────────────────────────────────────
        # Phase 6: Optimizer step (if enabled)
        # ─────────────────────────────────────────────────────────────
        self._run_optimizer_step()
        
        # ─────────────────────────────────────────────────────────────
        # Phase 7: Verify patterns (before swapping buffers)
        # ─────────────────────────────────────────────────────────────
        result = True
        if self.in_verification_phase:
            torch.cuda.synchronize()
            result = self._verify(iteration)
        
        # ─────────────────────────────────────────────────────────────
        # Phase 8: Swap buffers for next iteration
        # ─────────────────────────────────────────────────────────────
        self._swap_buffers()
        
        return result
    
    def _verify(self, iteration: int) -> bool:
        """Verify DDP patterns: H2D correctness and gradient consistency."""
        all_correct = True
        
        # Check H2D result
        if not self._verify_h2d(self.batch_gpu_current, iteration):
            all_correct = False
        
        # Check gradient consistency across ranks
        if not self._verify_gradient_consistency(iteration):
            all_correct = False
        
        return all_correct
    
    def _verify_gradient_consistency(self, iteration: int) -> bool:
        """
        Verify gradient consistency across ranks.
        
        Since all ranks use same input and same weights (deterministic mode),
        gradients should be identical after averaging.
        """
        if self.compute is None or not self.compute.parameters:
            return True
        
        # Compute local gradient checksum (sum of all grad elements)
        local_checksum = torch.zeros(1, device="cuda", dtype=torch.float32)
        for param in self.compute.parameters:
            if param.grad is not None:
                local_checksum += param.grad.sum().float()
        
        # Gather checksums from all ranks
        all_checksums = [
            torch.zeros(1, device="cuda", dtype=torch.float32)
            for _ in range(self.world_size)
        ]
        dist.all_gather(all_checksums, local_checksum)
        
        # All checksums should be identical (same input → same grad after avg)
        reference_checksum = all_checksums[0]
        all_correct = True
        
        for r, cs in enumerate(all_checksums):
            if not torch.allclose(cs, reference_checksum, rtol=1e-2, atol=1e-2):
                log.error(
                    f"GRADIENT DIVERGENCE (RUNTIME BUG!): iter={iteration} "
                    f"rank {r} checksum={cs.item():.6f} differs from "
                    f"rank 0 checksum={reference_checksum.item():.6f}"
                )
                self.corruption_details.append({
                    "type": "gradient_divergence",
                    "iteration": iteration,
                    "rank": self.rank,
                    "divergent_rank": r,
                    "expected_checksum": reference_checksum.item(),
                    "actual_checksum": cs.item(),
                })
                all_correct = False
        
        return all_correct


__all__ = ["DDPModeReproducer"]
