"""Device discovery and GPU property utilities.

This module combines device detection for distributed training (original aorta)
with comprehensive GPU property queries (from hw_queue_eval).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import torch


Accelerator = Literal["nvidia", "amd", "cpu"]

log = logging.getLogger(__name__)

# Detect if running on ROCm or CUDA
IS_ROCM = hasattr(torch.version, "hip") and torch.version.hip is not None
BACKEND_NAME = "ROCm/HIP" if IS_ROCM else "CUDA"


# =============================================================================
# Original aorta device utilities (for distributed training)
# =============================================================================


def detect_accelerator() -> Accelerator:
    """Detect the active accelerator type."""
    if not torch.cuda.is_available():
        return "cpu"
    if getattr(torch.version, "hip", None):
        return "amd"
    return "nvidia"


def _visible_device_indices() -> List[int]:
    """Return list of visible device indices from environment variables."""
    env = os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ.get("HIP_VISIBLE_DEVICES")
    if env:
        indices = [token.strip() for token in env.split(",") if token.strip()]
        try:
            return [int(token) for token in indices]
        except ValueError:
            log.warning("Non-integer device identifiers in *_VISIBLE_DEVICES=%s", env)
            return list(range(len(indices)))

    try:
        count = torch.cuda.device_count()
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("Unable to query device count via torch: %s", exc)
        count = 0
    return list(range(count))


def get_device(local_rank: int) -> torch.device:
    """Return the torch.device for the current process.

    Args:
        local_rank: Local rank of the current process

    Returns:
        torch.device for the current process
    """
    accelerator = detect_accelerator()
    if accelerator == "cpu":
        return torch.device("cpu")

    visible = _visible_device_indices()
    if not visible:
        raise RuntimeError("CUDA/HIP backend reports zero visible devices")

    mapped_index = visible[local_rank % len(visible)]
    if local_rank >= len(visible):
        log.warning(
            "local_rank %s exceeds visible device count %s; remapping to device %s",
            local_rank,
            len(visible),
            mapped_index,
        )

    torch.cuda.set_device(mapped_index)
    return torch.device("cuda", mapped_index)


def get_distributed_backend() -> str:
    """Select the distributed backend based on accelerator."""
    accelerator = detect_accelerator()
    if accelerator == "cpu":
        return "gloo"
    return "nccl"


# =============================================================================
# Extended device utilities (from hw_queue_eval)
# =============================================================================


@dataclass
class DeviceProperties:
    """GPU device properties relevant to queue evaluation and benchmarking."""

    name: str
    compute_capability: Tuple[int, int]
    total_memory_gb: float
    multi_processor_count: int
    max_threads_per_mp: int
    warp_size: int
    is_rocm: bool
    device_index: int

    # ROCm-specific properties (may be None on CUDA)
    gcn_arch: Optional[str] = None
    compute_units: Optional[int] = None

    def __str__(self) -> str:
        backend = "ROCm" if self.is_rocm else "CUDA"
        return (
            f"{self.name} ({backend})\n"
            f"  Memory: {self.total_memory_gb:.1f} GB\n"
            f"  Compute Units/SMs: {self.multi_processor_count}\n"
            f"  Warp/Wavefront Size: {self.warp_size}"
        )


def get_device_properties(device: str = "cuda:0") -> DeviceProperties:
    """Get properties of the specified GPU device.

    Args:
        device: Device string (e.g., "cuda:0", "cuda:1")

    Returns:
        DeviceProperties object with device information
    """
    device_obj = torch.device(device)
    device_idx = device_obj.index if device_obj.index is not None else 0

    props = torch.cuda.get_device_properties(device_idx)

    # Get compute capability
    if IS_ROCM:
        # ROCm reports capability differently
        compute_cap = (props.major, props.minor)
        gcn_arch = getattr(props, "gcnArchName", None)
    else:
        compute_cap = (props.major, props.minor)
        gcn_arch = None

    return DeviceProperties(
        name=props.name,
        compute_capability=compute_cap,
        total_memory_gb=props.total_memory / (1024**3),
        multi_processor_count=props.multi_processor_count,
        max_threads_per_mp=props.max_threads_per_multi_processor,
        warp_size=props.warp_size if hasattr(props, "warp_size") else 64 if IS_ROCM else 32,
        is_rocm=IS_ROCM,
        device_index=device_idx,
        gcn_arch=gcn_arch,
        compute_units=props.multi_processor_count if IS_ROCM else None,
    )


def ensure_gpu_available(device: str = "cuda:0") -> bool:
    """Check if GPU is available and accessible.

    Args:
        device: Device string to check

    Returns:
        True if GPU is available and accessible
    """
    if not torch.cuda.is_available():
        return False

    try:
        device_obj = torch.device(device)
        device_idx = device_obj.index if device_obj.index is not None else 0
        if device_idx >= torch.cuda.device_count():
            return False
        # Try to allocate a small tensor
        with torch.cuda.device(device_idx):
            _ = torch.empty(1, device=device)
        return True
    except Exception:
        return False


def get_memory_stats(device: str = "cuda:0") -> dict:
    """Get current GPU memory statistics.

    Args:
        device: Device to query

    Returns:
        Dictionary with memory statistics
    """
    device_idx = torch.device(device).index or 0

    return {
        "allocated_gb": torch.cuda.memory_allocated(device_idx) / (1024**3),
        "reserved_gb": torch.cuda.memory_reserved(device_idx) / (1024**3),
        "max_allocated_gb": torch.cuda.max_memory_allocated(device_idx) / (1024**3),
        "max_reserved_gb": torch.cuda.max_memory_reserved(device_idx) / (1024**3),
    }


def reset_memory_stats(device: str = "cuda:0") -> None:
    """Reset memory statistics for device."""
    device_idx = torch.device(device).index or 0
    torch.cuda.reset_peak_memory_stats(device_idx)


def get_rocm_env_info() -> dict:
    """Get ROCm-specific environment information.

    Returns:
        Dictionary with ROCm environment details
    """
    info = {
        "is_rocm": IS_ROCM,
        "hip_version": getattr(torch.version, "hip", None),
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
    }

    if IS_ROCM:
        # ROCm-specific environment variables
        rocm_env_vars = [
            "ROCM_PATH",
            "HIP_VISIBLE_DEVICES",
            "AMD_LOG_LEVEL",
            "HSA_TOOLS_LIB",
            "GPU_MAX_HW_QUEUES",
            "HSA_ENABLE_SDMA",
        ]
        info["env_vars"] = {var: os.environ.get(var) for var in rocm_env_vars}

    return info


def get_driver_info() -> dict:
    """Get AMDGPU driver/DKMS version information.

    Detects the active driver variant by:
    1. Checking AMDGPU_DRIVER_VARIANT environment variable (for Docker use)
    2. Reading the actually loaded kernel module version from /sys/module/amdgpu/version
    3. Checking for a .driver_variant marker file in the source
    4. Comparing the active source against backup directories
    5. Reporting all available variants

    Note: This function is designed to work both on bare metal and inside
    Docker containers. It prioritizes /sys/module/amdgpu/version over
    `dkms status` since the latter returns container package info, not
    the host's loaded driver.

    Environment Variables:
        AMDGPU_DRIVER_VARIANT: Explicitly set the driver variant name
            (e.g., "patched", "base", "mqd_vram"). Useful when running
            in Docker where host source directories aren't visible.

    Returns:
        Dictionary with driver_version, dkms_version, driver_type, kernel,
        available_variants, and active_variant keys
    """
    import glob
    import hashlib
    import subprocess

    info = {
        "driver_version": None,
        "dkms_version": None,
        "driver_type": "unknown",
        "kernel": None,
        "available_variants": [],
        "active_variant": None,
        "active_source_dir": None,
    }

    # Method 0: Check environment variable override (highest priority)
    # This is useful in Docker where host source dirs aren't visible
    env_variant = os.environ.get("AMDGPU_DRIVER_VARIANT")
    if env_variant:
        info["active_variant"] = env_variant
        info["driver_type"] = env_variant
        log.info(f"Driver variant set from AMDGPU_DRIVER_VARIANT: {env_variant}")

    # Get kernel version
    try:
        result = subprocess.run(
            ["uname", "-r"], capture_output=True, text=True, timeout=5
        )
        info["kernel"] = result.stdout.strip()
    except Exception:
        pass

    # Method 1: Get ACTUAL loaded driver version from sysfs (works in containers)
    # This reflects the host's loaded kernel module, not container packages
    loaded_version = None
    try:
        with open("/sys/module/amdgpu/version", "r") as f:
            loaded_version = f.read().strip()
            info["driver_version"] = loaded_version
    except Exception:
        pass

    # Method 2: Get DKMS version (may differ in containers)
    # This is useful on bare metal but returns container package info in Docker
    active_source_dir = None
    try:
        result = subprocess.run(
            ["dkms", "status"], capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.splitlines():
            if "amdgpu" in line and "installed" in line:
                # Parse: amdgpu/6.14.14-2281817.22.04, 5.15.0-153-generic, x86_64: installed
                parts = line.split(",")
                if parts:
                    version = parts[0].replace("amdgpu/", "").strip()
                    info["dkms_version"] = version
                    active_source_dir = f"/usr/src/amdgpu-{version}"
                    info["active_source_dir"] = active_source_dir
                break
    except Exception:
        pass

    # If loaded version differs from DKMS version, prefer loaded version for source dir
    # This handles the container case where dkms reports container packages
    if loaded_version and (info["dkms_version"] is None or
                           not info["dkms_version"].startswith(loaded_version)):
        # Search for source directory matching the loaded version
        matching_dirs = glob.glob(f"/usr/src/amdgpu-{loaded_version}*")
        # Filter to find the main directory (not backups like -base, -patched)
        for d in sorted(matching_dirs):
            # Skip backup directories
            if any(d.endswith(f"-{suffix}") for suffix in ["base", "patched", "custom"]):
                continue
            # Check if it looks like a version dir (ends with version-buildid pattern)
            active_source_dir = d
            info["active_source_dir"] = d
            # Extract the full version from directory name
            dir_version = d.replace("/usr/src/amdgpu-", "")
            info["dkms_version"] = dir_version
            break

    # Find all variant directories (backups like amdgpu-xxx-base, amdgpu-xxx-patched, etc.)
    if active_source_dir:
        base_pattern = active_source_dir + "-*"
        variant_dirs = glob.glob(base_pattern)
        for vdir in variant_dirs:
            # Extract variant name from directory (e.g., /usr/src/amdgpu-xxx-patched -> patched)
            variant_name = vdir.replace(active_source_dir + "-", "")
            info["available_variants"].append(variant_name)

    # Method 1: Check for explicit marker file (skip if env var already set)
    # Users can create a .driver_variant file with the variant name
    if info["active_variant"] is None and active_source_dir:
        marker_file = f"{active_source_dir}/.driver_variant"
        try:
            with open(marker_file, "r") as f:
                variant = f.read().strip()
                info["active_variant"] = variant
                info["driver_type"] = variant
        except FileNotFoundError:
            pass
        except Exception:
            pass

    # Method 2: Compare active source against backups using file hashes
    if info["active_variant"] is None and active_source_dir and info["available_variants"]:
        active_hash = _compute_source_hash(active_source_dir)
        if active_hash:
            for variant in info["available_variants"]:
                variant_dir = f"{active_source_dir}-{variant}"
                variant_hash = _compute_source_hash(variant_dir)
                if variant_hash and active_hash == variant_hash:
                    info["active_variant"] = variant
                    info["driver_type"] = variant
                    break

        # If no match found, it's a custom/modified version
        if info["active_variant"] is None:
            info["active_variant"] = "custom"
            info["driver_type"] = "custom (modified)"

    # Method 3: If no backups exist, just report as "default"
    if not info["available_variants"] and info["active_variant"] is None:
        info["active_variant"] = "default"
        info["driver_type"] = "default"

    return info


def _compute_source_hash(source_dir: str, sample_files: list = None) -> str:
    """Compute a hash of key source files to identify driver variant.

    Args:
        source_dir: Path to the driver source directory
        sample_files: List of relative paths to hash (uses defaults if None)

    Returns:
        Hash string or None if files not found
    """
    import hashlib
    import os

    # Sample key files that are likely to differ between patches
    if sample_files is None:
        sample_files = [
            "amd/amdkfd/kfd_mqd_manager_v9.c",
            "amd/amdkfd/kfd_device_queue_manager.c",
            "amd/amdgpu/amdgpu_amdkfd.c",
            "amd/amdkfd/kfd_priv.h",
        ]

    hasher = hashlib.md5()
    files_found = 0

    for rel_path in sample_files:
        full_path = os.path.join(source_dir, rel_path)
        try:
            with open(full_path, "rb") as f:
                hasher.update(f.read())
                files_found += 1
        except Exception:
            continue

    if files_found == 0:
        return None

    return hasher.hexdigest()


def set_driver_variant_marker(variant_name: str, source_dir: str = None) -> bool:
    """Set a marker file to explicitly label the active driver variant.

    This is called by the switch_driver.sh script after switching variants.

    Args:
        variant_name: Name of the variant (e.g., "base", "patched", "mqd_vram")
        source_dir: Path to the driver source directory (auto-detected if None)

    Returns:
        True if marker was set successfully
    """
    import glob

    if source_dir is None:
        # Find active source directory
        dirs = glob.glob("/usr/src/amdgpu-[0-9]*")
        # Filter out backup directories (those with -suffix)
        dirs = [d for d in dirs if not any(d.endswith(f"-{v}") for v in
                ["base", "patched", "custom"] + [d.split("-")[-1] for d in glob.glob("/usr/src/amdgpu-*-*")])]
        if dirs:
            source_dir = sorted(dirs)[0]

    if source_dir is None:
        return False

    try:
        marker_file = f"{source_dir}/.driver_variant"
        with open(marker_file, "w") as f:
            f.write(variant_name)
        return True
    except Exception:
        return False


def get_system_info() -> dict:
    """Get comprehensive system information for logging.

    Returns:
        Dictionary with hostname, node info, GPU info, ROCm info, driver info
    """
    import socket
    import subprocess

    info = {
        "hostname": socket.gethostname(),
        "driver": get_driver_info(),
        "rocm": get_rocm_env_info(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpus": [],
    }

    # Get GPU names
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info["gpus"].append({
                "index": i,
                "name": props.name,
                "memory_gb": props.total_memory / (1024**3),
            })

    # Get ROCm version from rocm-smi if available
    try:
        result = subprocess.run(
            ["rocm-smi", "--showdriverversion"],
            capture_output=True, text=True, timeout=10
        )
        for line in result.stdout.splitlines():
            if "Driver version" in line:
                info["rocm_driver_version"] = line.split(":")[-1].strip()
                break
    except Exception:
        pass

    # Try to get ROCm version from rocminfo
    try:
        result = subprocess.run(
            ["rocminfo"], capture_output=True, text=True, timeout=10
        )
        for line in result.stdout.splitlines():
            if "ROCm Runtime Version" in line:
                info["rocm_runtime_version"] = line.split(":")[-1].strip()
                break
    except Exception:
        pass

    return info


def log_environment_info(
    stream_counts: list,
    iterations: int,
    output_dir: str = None,
    logger=None,
) -> dict:
    """Log comprehensive environment information at the start of a run.

    Args:
        stream_counts: List of stream counts being tested
        iterations: Number of iterations per config
        output_dir: Output directory for results
        logger: Optional logger to use (defaults to print)

    Returns:
        Dictionary with all environment info (for saving to results)
    """
    import json
    from datetime import datetime
    from pathlib import Path

    log_fn = logger.info if logger else print

    info = get_system_info()
    info["run_config"] = {
        "stream_counts": stream_counts,
        "iterations": iterations,
        "output_dir": str(output_dir) if output_dir else None,
        "timestamp": datetime.now().isoformat(),
    }

    # Print formatted output
    log_fn("=" * 70)
    log_fn("ENVIRONMENT INFORMATION")
    log_fn("=" * 70)
    log_fn("")
    log_fn("SYSTEM:")
    log_fn(f"  Hostname:       {info['hostname']}")
    log_fn(f"  Kernel:         {info['driver'].get('kernel', 'unknown')}")
    log_fn("")
    log_fn("DRIVER:")
    log_fn(f"  DKMS Version:   {info['driver'].get('dkms_version', 'unknown')}")
    log_fn(f"  Active Variant: {info['driver'].get('active_variant', 'unknown')}")
    log_fn(f"  Driver Type:    {info['driver'].get('driver_type', 'unknown')}")
    available = info['driver'].get('available_variants', [])
    if available:
        log_fn(f"  Available:      {', '.join(available)}")
    log_fn("")
    log_fn("ROCm:")
    log_fn(f"  HIP Version:    {info['rocm'].get('hip_version', 'unknown')}")
    log_fn(f"  PyTorch:        {info['rocm'].get('torch_version', 'unknown')}")
    if info.get('rocm_runtime_version'):
        log_fn(f"  Runtime:        {info['rocm_runtime_version']}")
    log_fn("")
    log_fn("GPUs:")
    log_fn(f"  Count:          {info['gpu_count']}")
    for gpu in info.get('gpus', []):
        log_fn(f"  [{gpu['index']}] {gpu['name']} ({gpu['memory_gb']:.1f} GB)")
    log_fn("")
    log_fn("RUN CONFIGURATION:")
    log_fn(f"  Stream Counts:  {stream_counts}")
    log_fn(f"  Iterations:     {iterations}")
    if output_dir:
        log_fn(f"  Output Dir:     {output_dir}")
    log_fn("")
    log_fn("=" * 70)

    # Save environment info to file if output_dir specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        env_file = output_path / "environment_info.json"
        with open(env_file, "w") as f:
            json.dump(info, f, indent=2, default=str)
        log_fn(f"Environment info saved to: {env_file}")

    return info


__all__ = [
    # Type aliases
    "Accelerator",
    # Constants
    "IS_ROCM",
    "BACKEND_NAME",
    # Original aorta functions
    "detect_accelerator",
    "get_device",
    "get_distributed_backend",
    # Extended device utilities
    "DeviceProperties",
    "get_device_properties",
    "ensure_gpu_available",
    "get_memory_stats",
    "reset_memory_stats",
    "get_rocm_env_info",
    "get_driver_info",
    "get_system_info",
    "log_environment_info",
    "set_driver_variant_marker",
]
