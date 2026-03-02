#!/usr/bin/env python3
"""
Distributed SDPA backward testing script for multi-node execution.

This script:
1. Initializes distributed environment with FSDP context
2. Loads SDPA inputs from saved files (broadcast from rank 0)
3. Runs SDPA backward operations in a custom iteration loop
4. Detects and reports NaN/Inf issues across all ranks
5. Supports multi-node execution via master_launch.sh

Usage:
    # Single node
    torchrun --nproc_per_node=8 scripts/run_sdpa_distributed_test.py --config config/multi_node/sdpa_test_multi_node.yaml

    # Multi-node via master_launch.sh
    ./scripts/multi_node/master_launch.sh --config config/multi_node/sdpa_test_multi_node.yaml --nproc 8
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist

# Import AORTA utilities
from aorta.utils import (
    detect_accelerator,
    get_device,
    get_distributed_backend,
    load_config,
    setup_logging,
)

log = logging.getLogger(__name__)


@dataclass
class SDPATestConfig:
    """Configuration for SDPA distributed testing."""
    input_dir: str = "/home/vivekag/scratch/apps/aorta_work/nan_issue/sdpa/input"
    max_iterations: int = 1000
    device: str = "cuda"
    verbose: bool = False
    broadcast_inputs: bool = True
    output_dir: Path = Path("artifacts/sdpa_test")


@dataclass
class LoggingConfig:
    level: str = "INFO"


def load_tensor(file_path: str, device: torch.device) -> Optional[torch.Tensor]:
    """Load a tensor from a local file."""
    if not os.path.exists(file_path):
        return None
    try:
        tensor = torch.load(file_path, map_location="cpu", weights_only=False)
        return tensor
    except Exception as e:
        log.warning(f"Could not load {file_path}: {e}")
        return None


def load_metadata(file_path: str) -> Optional[dict]:
    """Load metadata JSON from a local file."""
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path) as f:
            return json.load(f)
    except Exception as e:
        log.warning(f"Could not load metadata from {file_path}: {e}")
        return None


def load_sdpa_inputs(input_dir: str, device: torch.device, rank: int) -> Dict[str, Any]:
    """
    Load SDPA inputs from saved files.

    Only rank 0 loads from disk; other ranks will receive via broadcast.
    """
    inputs = {}

    if rank != 0:
        # Non-rank-0 processes return empty dict; will receive via broadcast
        return inputs

    log.info(f"[Rank {rank}] Loading SDPA inputs from {input_dir}")

    # Load metadata
    metadata_path = os.path.join(input_dir, "metadata.json")
    metadata = load_metadata(metadata_path)
    if not metadata:
        log.error(f"[Rank {rank}] ERROR: metadata.json not found in {input_dir}")
        return inputs

    log.info(f"[Rank {rank}] Loaded metadata: func={metadata.get('func_name')}, rank={metadata.get('rank')}")

    saved_inputs = metadata.get("saved_inputs", {})

    # Load tensor inputs
    tensor_names = [
        "grad_out",
        "query",
        "key",
        "value",
        "out",
        "logsumexp",
        "cum_seq_q",
        "cum_seq_k",
        "philox_seed",
        "philox_offset",
    ]

    for name in tensor_names:
        value = saved_inputs.get(name, "None")
        if value in ["None", "null", None]:
            continue

        file_path = os.path.join(input_dir, value)
        tensor = load_tensor(file_path, device)
        if tensor is not None:
            inputs[name] = tensor
            log.info(f"[Rank {rank}] Loaded: {name} - shape={list(tensor.shape)}, dtype={tensor.dtype}")

    # Load scalar inputs
    scalar_names = ["max_q", "max_k", "dropout_p", "is_causal", "scale"]

    for name in scalar_names:
        value = saved_inputs.get(name, "None")
        if value in ["None", "null", None]:
            continue
        try:
            if name in ["max_q", "max_k"]:
                inputs[name] = int(value)
            elif name in ["dropout_p", "scale"]:
                inputs[name] = float(value)
            elif name == "is_causal":
                inputs[name] = str(value).lower() in ["true", "1"]
            log.info(f"[Rank {rank}] Loaded scalar: {name} = {inputs[name]}")
        except (ValueError, AttributeError) as e:
            log.warning(f"[Rank {rank}] Could not parse {name}={value}: {e}")

    # Validate required inputs
    required_inputs = ["grad_out", "query", "key", "value", "out", "logsumexp"]
    missing_inputs = [name for name in required_inputs if name not in inputs]
    if missing_inputs:
        log.error(f"[Rank {rank}] ERROR: Missing required inputs: {missing_inputs}")
        return {}

    return inputs


def broadcast_inputs(inputs: Dict[str, Any], device: torch.device, rank: int, world_size: int) -> Dict[str, Any]:
    """
    Broadcast inputs from rank 0 to all other ranks.

    Strategy:
    1. Rank 0 broadcasts tensor metadata (shapes, dtypes, scalar values)
    2. All ranks allocate tensors based on metadata
    3. Rank 0 broadcasts actual tensor data
    """
    if world_size == 1:
        return inputs

    log.info(f"[Rank {rank}] Broadcasting inputs from rank 0...")

    # Broadcast metadata
    if rank == 0:
        # Prepare metadata
        metadata = {
            "tensor_names": [],
            "tensor_shapes": [],
            "tensor_dtypes": [],
            "scalar_names": [],
            "scalar_values": [],
        }

        for name, value in inputs.items():
            if isinstance(value, torch.Tensor):
                metadata["tensor_names"].append(name)
                metadata["tensor_shapes"].append(list(value.shape))
                metadata["tensor_dtypes"].append(str(value.dtype))
            else:
                metadata["scalar_names"].append(name)
                metadata["scalar_values"].append(value)

        metadata_json = json.dumps(metadata)
        metadata_len = torch.tensor([len(metadata_json)], dtype=torch.int64, device=device)
    else:
        metadata_len = torch.tensor([0], dtype=torch.int64, device=device)

    # Broadcast metadata length
    dist.broadcast(metadata_len, src=0)

    # Broadcast metadata string
    if rank == 0:
        metadata_bytes = metadata_json.encode('utf-8')
        metadata_tensor = torch.tensor(list(metadata_bytes), dtype=torch.uint8, device=device)
    else:
        metadata_tensor = torch.zeros(metadata_len.item(), dtype=torch.uint8, device=device)

    dist.broadcast(metadata_tensor, src=0)

    # Parse metadata on non-rank-0 processes
    if rank != 0:
        metadata_bytes = bytes(metadata_tensor.cpu().numpy())
        metadata_json = metadata_bytes.decode('utf-8')
        metadata = json.loads(metadata_json)

        # Allocate tensors based on metadata
        for i, name in enumerate(metadata["tensor_names"]):
            shape = metadata["tensor_shapes"][i]
            dtype_str = metadata["tensor_dtypes"][i]
            # Parse dtype
            dtype = getattr(torch, dtype_str.split('.')[-1])
            inputs[name] = torch.zeros(shape, dtype=dtype, device=device)

        # Store scalar values
        for i, name in enumerate(metadata["scalar_names"]):
            inputs[name] = metadata["scalar_values"][i]

    # Broadcast tensors
    tensor_names = metadata["tensor_names"] if rank != 0 else [n for n in inputs.keys() if isinstance(inputs[n], torch.Tensor)]

    for name in tensor_names:
        log.debug(f"[Rank {rank}] Broadcasting tensor: {name}")
        dist.broadcast(inputs[name], src=0)

    log.info(f"[Rank {rank}] Broadcast complete. Received {len(tensor_names)} tensors, {len(metadata.get('scalar_names', []))} scalars")

    return inputs


def check_nan_inf(tensor: torch.Tensor) -> Tuple[bool, int, bool, int]:
    """Check if a tensor contains NaN or Inf values."""
    if tensor is None:
        return False, 0, False, 0

    nan_mask = torch.isnan(tensor)
    has_nan = bool(nan_mask.any().item())
    nan_count = int(nan_mask.sum().item())

    inf_mask = torch.isinf(tensor)
    has_inf = bool(inf_mask.any().item())
    inf_count = int(inf_mask.sum().item())

    return has_nan, nan_count, has_inf, inf_count


def check_nan_inf_distributed(
    tensors: Tuple[torch.Tensor, ...],
    tensor_names: list[str],
    rank: int,
    device: torch.device,
) -> Tuple[bool, Dict[str, Dict[str, int]]]:
    """
    Check for NaN/Inf across all ranks with all_reduce.

    Returns:
        (global_nan_found, stats_dict)
    """
    local_stats = {}

    for name, tensor in zip(tensor_names, tensors):
        has_nan, nan_count, has_inf, inf_count = check_nan_inf(tensor)
        local_stats[name] = {
            "has_nan": has_nan,
            "nan_count": nan_count,
            "has_inf": has_inf,
            "inf_count": inf_count,
        }

    # Create tensor for all_reduce: [has_nan, has_inf]
    local_issues = torch.tensor(
        [any(s["has_nan"] for s in local_stats.values()), any(s["has_inf"] for s in local_stats.values())],
        dtype=torch.float32,
        device=device,
    )

    # All-reduce to detect if any rank has issues
    global_issues = local_issues.clone()
    dist.all_reduce(global_issues, op=dist.ReduceOp.MAX)

    global_nan_found = bool(global_issues[0].item() > 0) or bool(global_issues[1].item() > 0)

    return global_nan_found, local_stats


def run_sdpa_backward_iteration(
    inputs: Dict[str, Any],
    device: torch.device,
    iteration: int,
    rank: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run a single SDPA backward iteration."""

    # Move tensors to device
    grad_out = inputs["grad_out"].to(device)
    query = inputs["query"].to(device)
    key = inputs["key"].to(device)
    value = inputs["value"].to(device)
    out = inputs["out"].to(device)
    logsumexp = inputs["logsumexp"].to(device)

    cum_seq_q = inputs.get("cum_seq_q")
    if cum_seq_q is not None:
        cum_seq_q = cum_seq_q.to(device)

    cum_seq_k = inputs.get("cum_seq_k")
    if cum_seq_k is not None:
        cum_seq_k = cum_seq_k.to(device)

    max_q = inputs.get("max_q", 0)
    max_k = inputs.get("max_k", 0)
    dropout_p = inputs.get("dropout_p", 0.0)
    is_causal = inputs.get("is_causal", False)

    philox_seed = inputs.get("philox_seed")
    if philox_seed is not None:
        philox_seed = philox_seed.to(device)

    philox_offset = inputs.get("philox_offset")
    if philox_offset is not None:
        philox_offset = philox_offset.to(device)

    scale = inputs.get("scale")

    # Run SDPA backward
    log.debug(f"[Rank {rank}] Iteration {iteration}: Running SDPA backward...")

    try:
        result = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(
            grad_out,
            query,
            key,
            value,
            out,
            logsumexp,
            cum_seq_q,
            cum_seq_k,
            max_q,
            max_k,
            dropout_p,
            is_causal,
            philox_seed,
            philox_offset,
            scale=scale,
        )

        grad_query = result[0].detach()
        grad_key = result[1].detach()
        grad_value = result[2].detach()

        return grad_query, grad_key, grad_value

    except Exception as e:
        log.error(f"[Rank {rank}] Iteration {iteration}: ERROR running SDPA backward: {e}")
        raise


def run_sdpa_test_loop(
    inputs: Dict[str, Any],
    cfg: SDPATestConfig,
    device: torch.device,
    rank: int,
    world_size: int,
) -> int:
    """
    Main test loop - runs until NaN detected or max iterations reached.

    Returns:
        iteration where NaN was detected (or 0 if none)
    """
    log.info(f"[Rank {rank}] Starting SDPA test loop (max_iterations={cfg.max_iterations})...")

    for iteration in range(1, cfg.max_iterations + 1):
        if iteration % 10 == 0 or iteration == 1:
            log.info(f"[Rank {rank}] Iteration {iteration}/{cfg.max_iterations}")

        # Run SDPA backward
        try:
            grad_query, grad_key, grad_value = run_sdpa_backward_iteration(
                inputs, device, iteration, rank
            )
        except Exception as e:
            log.error(f"[Rank {rank}] Iteration {iteration}: Failed with exception: {e}")
            # Sync failure across ranks
            failure_flag = torch.tensor([1.0], device=device)
            dist.all_reduce(failure_flag, op=dist.ReduceOp.MAX)
            return -iteration  # Negative indicates failure

        # Move to CPU for checking
        grad_query_cpu = grad_query.cpu()
        grad_key_cpu = grad_key.cpu()
        grad_value_cpu = grad_value.cpu()

        # Check for NaN/Inf
        nan_found, local_stats = check_nan_inf_distributed(
            (grad_query_cpu, grad_key_cpu, grad_value_cpu),
            ["grad_query", "grad_key", "grad_value"],
            rank,
            device,
        )

        if nan_found:
            log.warning(f"[Rank {rank}] Iteration {iteration}: NaN/Inf DETECTED!")

            # Print local stats
            for name, stats in local_stats.items():
                if stats["has_nan"] or stats["has_inf"]:
                    log.warning(
                        f"[Rank {rank}]   {name}: NaN={stats['has_nan']} (count={stats['nan_count']}), "
                        f"Inf={stats['has_inf']} (count={stats['inf_count']})"
                    )

            return iteration

        # Synchronize to ensure all ranks complete iteration
        dist.barrier()

    log.info(f"[Rank {rank}] Completed {cfg.max_iterations} iterations without NaN/Inf")
    return 0


def init_distributed(cfg: SDPATestConfig, log_level: str) -> Dict[str, Any]:
    """Initialize distributed environment."""
    backend = get_distributed_backend()

    # Get timeout from environment
    timeout_env_raw = os.environ.get("TORCH_DIST_INIT_TIMEOUT", "ENV_NOT_SET")
    if timeout_env_raw != "ENV_NOT_SET":
        timeout_seconds = int(timeout_env_raw)
    else:
        timeout_seconds = 30

    dist.init_process_group(backend=backend, timeout=timedelta(seconds=timeout_seconds))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
    device = get_device(local_rank)
    torch.cuda.set_device(device)

    # Create output directory
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    setup_logging(level=log_level, log_file=cfg.output_dir / f"rank{rank}.log", rank=rank)

    log.info(
        f"Initialized distributed testing | backend={backend} rank={rank} world={world_size} "
        f"local_rank={local_rank} device={device} timeout={timeout_seconds}s"
    )

    return {
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "device": device,
    }


def set_seed(seed: int, rank: int) -> None:
    """Set all random seeds for reproducibility."""
    seed_value = seed + rank
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    log.info(f"Set random seed={seed_value} for rank={rank}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Distributed SDPA backward testing")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--local-rank",
        type=int,
        default=None,
        help="Local rank (set by torchrun)",
    )
    args = parser.parse_args()

    # Load configuration
    raw_config = load_config(args.config)

    # Build configs
    sdpa_cfg = SDPATestConfig()
    if "sdpa_test" in raw_config:
        for key, value in raw_config["sdpa_test"].items():
            if hasattr(sdpa_cfg, key):
                if key == "output_dir":
                    setattr(sdpa_cfg, key, Path(value))
                else:
                    setattr(sdpa_cfg, key, value)

    # Logging config
    log_cfg = LoggingConfig()
    if "logging" in raw_config:
        for key, value in raw_config["logging"].items():
            if hasattr(log_cfg, key):
                setattr(log_cfg, key, value)

    # Initialize distributed
    dist_info = init_distributed(sdpa_cfg, log_cfg.level)
    rank = dist_info["rank"]
    world_size = dist_info["world_size"]
    device = dist_info["device"]

    log.info(f"[Rank {rank}] Starting SDPA distributed test")
    log.info(f"[Rank {rank}] Configuration: input_dir={sdpa_cfg.input_dir}, max_iterations={sdpa_cfg.max_iterations}")

    # Set seed
    set_seed(42, rank)

    # Load SDPA inputs (rank 0 only)
    inputs = load_sdpa_inputs(sdpa_cfg.input_dir, device, rank)

    if rank == 0 and not inputs:
        log.error(f"[Rank {rank}] Failed to load SDPA inputs. Exiting.")
        dist.destroy_process_group()
        sys.exit(1)

    # Broadcast inputs to all ranks
    if sdpa_cfg.broadcast_inputs and world_size > 1:
        inputs = broadcast_inputs(inputs, device, rank, world_size)
    elif rank != 0:
        log.error(f"[Rank {rank}] broadcast_inputs=False but not rank 0. Cannot load inputs.")
        dist.destroy_process_group()
        sys.exit(1)

    # Synchronize before starting test loop
    dist.barrier()

    # Run test loop
    nan_iteration = run_sdpa_test_loop(inputs, sdpa_cfg, device, rank, world_size)

    # Gather results from all ranks
    all_nan_iterations = [torch.tensor([0], device=device) for _ in range(world_size)]
    dist.all_gather(all_nan_iterations, torch.tensor([nan_iteration], device=device))

    # Rank 0 reports results
    if rank == 0:
        log.info("=" * 70)
        log.info("SDPA DISTRIBUTED TEST RESULTS")
        log.info("=" * 70)

        all_iterations = [int(t.item()) for t in all_nan_iterations]

        if any(it > 0 for it in all_iterations):
            log.warning(f"NaN/Inf DETECTED across ranks!")
            for r, it in enumerate(all_iterations):
                if it > 0:
                    log.warning(f"  Rank {r}: NaN detected at iteration {it}")
                else:
                    log.info(f"  Rank {r}: No NaN detected")
        elif any(it < 0 for it in all_iterations):
            log.error(f"FAILURES detected across ranks!")
            for r, it in enumerate(all_iterations):
                if it < 0:
                    log.error(f"  Rank {r}: Failed at iteration {-it}")
        else:
            log.info(f"PASS - Completed {sdpa_cfg.max_iterations} iterations without NaN/Inf on all ranks")

        log.info("=" * 70)

    # Cleanup
    dist.destroy_process_group()

    # Exit with appropriate code
    if nan_iteration > 0:
        sys.exit(1)  # NaN detected
    elif nan_iteration < 0:
        sys.exit(2)  # Failure
    else:
        sys.exit(0)  # Success


if __name__ == "__main__":
    main()
