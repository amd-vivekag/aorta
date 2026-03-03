"""
Mixture of Experts Training Workload

Pattern: Multiple expert MLPs execute in parallel
- Each expert gets its own stream
- All-to-all for token routing between experts
- Gating network on separate stream

With 8-16 experts, this naturally stresses high stream counts.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from aorta.hw_queue_eval.workloads.base import DistributedWorkload, MultiGPUMixin
from aorta.hw_queue_eval.workloads.registry import WorkloadRegistry


class ExpertMLP(nn.Module):
    """Single expert MLP."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class GatingNetwork(nn.Module):
    """Router/gating network for MoE."""

    def __init__(self, hidden_size: int, num_experts: int):
        super().__init__()
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(self, x: torch.Tensor, top_k: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute gating scores and select top-k experts.

        Returns:
            routing_weights: [batch, seq, top_k] - weights for selected experts
            expert_indices: [batch, seq, top_k] - indices of selected experts
        """
        logits = self.gate(x)  # [batch, seq, num_experts]
        routing_weights, expert_indices = torch.topk(
            F.softmax(logits, dim=-1), top_k, dim=-1
        )
        # Normalize weights
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        return routing_weights, expert_indices


@WorkloadRegistry.register
class MoEWorkload(MultiGPUMixin, DistributedWorkload):
    """
    Mixture of Experts with parallel expert execution.

    This workload stress-tests high stream counts by running each expert
    on a separate stream. With 8-16 experts, this requires 8-16+ concurrent
    streams for maximum parallelism.

    Stream assignment:
    - Stream 0: Gating network
    - Streams 1-N: One per expert
    - Remaining streams: All-to-all communication

    The pattern is:
    1. Gating computes routing weights
    2. Tokens are dispatched to experts (all-to-all in distributed setting)
    3. Each expert processes its tokens in parallel
    4. Results are combined (another all-to-all)
    """

    name = "moe"
    description = "Mixture of Experts with parallel expert execution"
    category = "distributed"
    min_streams = 4
    max_streams = 32
    recommended_streams = 16
    switch_latency_sensitivity = "critical"
    memory_requirements_gb = 8.0
    multi_gpu_capable = True

    def __init__(
        self,
        num_experts: int = 8,
        hidden_size: int = 1024,
        intermediate_size: int = 4096,
        top_k: int = 2,
        batch_size: int = 4,
        seq_length: int = 512,
        simulate_collectives: bool = True,
        use_multi_gpu: bool = True,
    ):
        """
        Initialize MoE workload.

        Args:
            num_experts: Number of experts
            hidden_size: Model hidden size
            intermediate_size: Expert intermediate size
            top_k: Number of experts per token
            batch_size: Batch size
            seq_length: Sequence length
            simulate_collectives: Mock all-to-all operations
            use_multi_gpu: If True, distribute experts across all available GPUs
        """
        super().__init__(simulate_collectives)

        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.top_k = top_k
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.use_multi_gpu = use_multi_gpu

        self._gating: Optional[GatingNetwork] = None
        self._experts: List[ExpertMLP] = []
        self._expert_streams: List[int] = []
        self._devices: List[str] = []
        self._stream_to_device: Dict[int, str] = {}

    def setup(self, stream_count: int, device: str = "cuda:0") -> None:
        """Setup gating network and experts."""
        self._stream_count = stream_count
        self._is_setup = True

        # Setup multi-GPU device mapping
        self._setup_multi_gpu(stream_count, device, self.use_multi_gpu)

        # Assign streams to experts
        # Stream 0 for gating, rest for experts (round-robin if fewer streams than experts)
        self._gating_stream = 0
        self._expert_streams = []
        available_expert_streams = stream_count - 1  # Exclude gating stream

        for i in range(self.num_experts):
            stream_idx = 1 + (i % max(1, available_expert_streams))
            self._expert_streams.append(stream_idx)

        # Create gating network on gating stream's device
        gating_device = self._get_device_for_stream(self._gating_stream)
        self._gating = GatingNetwork(self.hidden_size, self.num_experts).to(gating_device)
        self._gating.eval()

        # Create experts - each on its stream's device
        self._experts = []
        for i in range(self.num_experts):
            stream_idx = self._expert_streams[i]
            expert_device = self._get_device_for_stream(stream_idx)
            expert = ExpertMLP(self.hidden_size, self.intermediate_size).to(expert_device)
            expert.eval()
            self._experts.append(expert)

        # Input tensor on gating device
        self._input = torch.randn(
            self.batch_size, self.seq_length, self.hidden_size,
            dtype=torch.float32, device=gating_device
        )
        self._tensors["input"] = self._input

        # Buffers for expert outputs (on their respective devices)
        self._expert_outputs = []
        for i in range(self.num_experts):
            stream_idx = self._expert_streams[i]
            expert_device = self._get_device_for_stream(stream_idx)
            out = torch.zeros(
                self.batch_size, self.seq_length, self.hidden_size,
                dtype=torch.float32, device=expert_device
            )
            self._expert_outputs.append(out)
            self._tensors[f"expert_out_{i}"] = out

    def run_iteration(self, streams: List[torch.cuda.Stream]) -> None:
        """
        Execute one MoE forward pass.

        1. Gating computes routing (stream 0)
        2. Each expert processes assigned tokens (parallel streams)
        3. Results are combined
        """
        gating_stream = streams[self._gating_stream]
        gating_device = self._get_device_for_stream(self._gating_stream)
        x = self._input

        # Step 1: Compute routing weights
        with torch.cuda.stream(gating_stream):
            routing_weights, expert_indices = self._gating(x, self.top_k)

        # Wait for gating to complete before expert dispatch
        for stream_idx in set(self._expert_streams):
            streams[stream_idx].wait_stream(gating_stream)

        # Step 2: Dispatch tokens to experts (parallel)
        # In real MoE, this would involve all-to-all
        # Here we simulate by having each expert process all tokens
        # weighted by their routing weights

        for expert_idx, expert in enumerate(self._experts):
            stream_idx = self._expert_streams[expert_idx]
            expert_stream = streams[stream_idx]
            expert_device = self._get_device_for_stream(stream_idx)

            with torch.cuda.stream(expert_stream):
                # Move input and routing info to expert's device if needed
                x_local = x.to(expert_device, non_blocking=True) if x.device != torch.device(expert_device) else x
                routing_weights_local = routing_weights.to(expert_device, non_blocking=True) if routing_weights.device != torch.device(expert_device) else routing_weights
                expert_indices_local = expert_indices.to(expert_device, non_blocking=True) if expert_indices.device != torch.device(expert_device) else expert_indices

                # Find tokens routed to this expert
                # Simplified: process all tokens, will be masked by routing weights
                expert_out = expert(x_local)

                # Compute contribution from this expert
                # Get mask for tokens using this expert
                expert_mask = (expert_indices_local == expert_idx).any(dim=-1)  # [batch, seq]

                # Get weights for this expert (where it's selected)
                expert_weight = torch.zeros_like(expert_mask, dtype=torch.float32)
                for k in range(self.top_k):
                    mask_k = expert_indices_local[..., k] == expert_idx
                    expert_weight = torch.where(
                        mask_k, routing_weights_local[..., k], expert_weight
                    )

                # Weight the output
                weighted_out = expert_out * expert_weight.unsqueeze(-1)
                self._expert_outputs[expert_idx].copy_(weighted_out)

        # Step 3: Combine expert outputs
        # In distributed, this would be another all-to-all
        # Use gating stream for combination
        for stream_idx in set(self._expert_streams):
            gating_stream.wait_stream(streams[stream_idx])

        with torch.cuda.stream(gating_stream):
            combined = torch.zeros_like(x)
            for expert_out in self._expert_outputs:
                # Move expert outputs to gating device for combination
                expert_out_local = expert_out.to(gating_device, non_blocking=True) if expert_out.device != torch.device(gating_device) else expert_out
                combined = combined + expert_out_local

    def get_throughput_unit(self) -> str:
        return "tokens/sec"

    def compute_throughput(self, iterations: int, total_time_sec: float) -> float:
        if total_time_sec <= 0:
            return 0.0
        tokens = iterations * self.batch_size * self.seq_length
        return tokens / total_time_sec

    def get_config(self) -> Dict[str, Any]:
        config = {
            "name": self.name,
            "num_experts": self.num_experts,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "top_k": self.top_k,
            "batch_size": self.batch_size,
            "seq_length": self.seq_length,
            "expert_stream_assignment": self._expert_streams,
            "simulate_collectives": self._simulate_collectives,
        }
        config.update(self._get_multi_gpu_config())
        return config

    def cleanup(self) -> None:
        """Cleanup experts and buffers."""
        super().cleanup()
        self._gating = None
        self._experts = []
        self._expert_outputs = []
