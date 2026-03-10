"""GPU hardware control for deterministic benchmarking.

Provides GPU power/frequency management (lock-clock, power-limit) using
direct subprocess calls to rocm-smi (AMD) and nvidia-smi (NVIDIA).
"""

from __future__ import annotations

import logging
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GPU vendor / hardware info types
# ---------------------------------------------------------------------------


class GPUVendor(Enum):
    """GPU vendor type."""

    AMD = auto()
    NVIDIA = auto()
    UNKNOWN = auto()


@dataclass
class GPUHardwareInfo:
    """GPU hardware information captured for benchmark metadata."""

    vendor: GPUVendor
    architecture: Optional[str] = None
    device_id: int = 0
    device_name: Optional[str] = None

    power_current_watts: Optional[float] = None
    power_limit_watts: Optional[float] = None
    power_max_watts: Optional[float] = None

    gpu_clock_current: Optional[int] = None
    gpu_clock_max: Optional[int] = None
    mem_clock_current: Optional[int] = None
    mem_clock_max: Optional[int] = None

    temperature: Optional[float] = None

    memory_total_gb: Optional[float] = None
    memory_used_gb: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "vendor": self.vendor.name,
            "architecture": self.architecture,
            "device_id": self.device_id,
            "device_name": self.device_name,
            "power_current_watts": self.power_current_watts,
            "power_limit_watts": self.power_limit_watts,
            "power_max_watts": self.power_max_watts,
            "gpu_clock_current": self.gpu_clock_current,
            "gpu_clock_max": self.gpu_clock_max,
            "mem_clock_current": self.mem_clock_current,
            "mem_clock_max": self.mem_clock_max,
            "temperature": self.temperature,
            "memory_total_gb": self.memory_total_gb,
            "memory_used_gb": self.memory_used_gb,
        }


# ---------------------------------------------------------------------------
# GPU config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class GPUConfig:
    """Low-level GPU configuration for power and frequency control."""

    device_id: int = 0
    power_limit_watts: Optional[int] = None
    gpu_clock_mhz: Optional[Tuple[int, int]] = None
    mem_clock_mhz: Optional[Tuple[int, int]] = None
    gpu_clock_level: Optional[int] = None
    mem_clock_level: Optional[int] = None


@dataclass
class MultiGPUConfig:
    """Configuration for multiple GPUs."""

    default_config: Optional[GPUConfig] = None
    gpu_configs: Dict[int, GPUConfig] = field(default_factory=dict)
    device_ids: Optional[List[int]] = None
    parallel: bool = True

    def get_config_for_device(self, device_id: int) -> Optional[GPUConfig]:
        if device_id in self.gpu_configs:
            return self.gpu_configs[device_id]
        if self.default_config:
            return GPUConfig(
                device_id=device_id,
                power_limit_watts=self.default_config.power_limit_watts,
                gpu_clock_mhz=self.default_config.gpu_clock_mhz,
                mem_clock_mhz=self.default_config.mem_clock_mhz,
                gpu_clock_level=self.default_config.gpu_clock_level,
                mem_clock_level=self.default_config.mem_clock_level,
            )
        return None


# ---------------------------------------------------------------------------
# Vendor detection
# ---------------------------------------------------------------------------


def _detect_amd_gpu() -> Optional[str]:
    """Detect AMD GPU architecture via rocminfo."""
    try:
        result = subprocess.run(
            ["rocminfo"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if "gfx" in line.lower():
                    match = re.search(r"(gfx\w+)", line, re.IGNORECASE)
                    if match:
                        return match.group(1).lower()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    except Exception as exc:
        log.debug("AMD GPU detection failed: %s", exc)
    return None


def _detect_nvidia_gpu() -> Optional[str]:
    """Detect NVIDIA GPU architecture via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            compute_cap = result.stdout.strip().split("\n")[0]
            if compute_cap:
                major, minor = compute_cap.split(".")
                return f"sm_{major}{minor}"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    except Exception as exc:
        log.debug("NVIDIA GPU detection failed: %s", exc)
    return None


def detect_gpu() -> Tuple[GPUVendor, Optional[str]]:
    """Auto-detect GPU vendor and architecture.

    Returns (GPUVendor, arch_string) -- e.g. (AMD, "gfx942") or (NVIDIA, "sm_90").
    """
    amd_arch = _detect_amd_gpu()
    if amd_arch:
        return GPUVendor.AMD, amd_arch

    nvidia_arch = _detect_nvidia_gpu()
    if nvidia_arch:
        return GPUVendor.NVIDIA, nvidia_arch

    log.warning("No GPU detected")
    return GPUVendor.UNKNOWN, None


def get_gpu_count() -> int:
    """Return the number of available GPUs (0 if none detected)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=count", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
            return len(lines)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    try:
        result = subprocess.run(
            ["rocm-smi", "--showid"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            gpu_ids: set[int] = set()
            for line in result.stdout.split("\n"):
                match = re.match(r"^\s*GPU\[(\d+)\]", line)
                if match:
                    gpu_ids.add(int(match.group(1)))
            if gpu_ids:
                return len(gpu_ids)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return 0


# ---------------------------------------------------------------------------
# Single-GPU controller
# ---------------------------------------------------------------------------


class GPUController:
    """Controls power/frequency for a single GPU via rocm-smi or nvidia-smi."""

    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.vendor, self.arch = detect_gpu()

    def get_hardware_info(self) -> GPUHardwareInfo:
        if self.vendor == GPUVendor.AMD:
            return self._get_amd_info()
        elif self.vendor == GPUVendor.NVIDIA:
            return self._get_nvidia_info()
        return GPUHardwareInfo(vendor=GPUVendor.UNKNOWN, device_id=self.device_id)

    # -- AMD -----------------------------------------------------------------

    def _get_amd_info(self) -> GPUHardwareInfo:
        info = GPUHardwareInfo(
            vendor=GPUVendor.AMD, architecture=self.arch, device_id=self.device_id
        )
        dev = str(self.device_id)
        try:
            out = self._run(["rocm-smi", "-d", dev, "--showpower"])
            if out:
                for line in out.split("\n"):
                    if "Power" in line and "W" in line:
                        m = re.search(r"(\d+\.?\d*)\s*$", line)
                        if m:
                            info.power_current_watts = float(m.group(1))
                            break

            out = self._run(["rocm-smi", "-d", dev, "--showclocks"])
            if out:
                for line in out.split("\n"):
                    if "sclk" in line.lower():
                        m = re.search(r"(\d+)", line)
                        if m:
                            info.gpu_clock_current = int(m.group(1))
                    elif "mclk" in line.lower():
                        m = re.search(r"(\d+)", line)
                        if m:
                            info.mem_clock_current = int(m.group(1))

            out = self._run(["rocm-smi", "-d", dev, "--showtemp"])
            if out:
                for line in out.split("\n"):
                    if "Temperature" in line:
                        m = re.search(r"(\d+\.?\d*)", line)
                        if m:
                            info.temperature = float(m.group(1))
                            break

            out = self._run(["rocm-smi", "-d", dev, "--showmeminfo", "vram"])
            if out:
                for line in out.split("\n"):
                    if "Total" in line:
                        m = re.search(r"(\d+)", line)
                        if m:
                            info.memory_total_gb = int(m.group(1)) / (1024**3)
                    elif "Used" in line:
                        m = re.search(r"(\d+)", line)
                        if m:
                            info.memory_used_gb = int(m.group(1)) / (1024**3)
        except Exception as exc:
            log.warning("Error getting AMD GPU info: %s", exc)
        return info

    def _apply_amd_config(self, config: GPUConfig) -> bool:
        success = True
        dev = str(config.device_id)
        try:
            if config.power_limit_watts is not None:
                r = self._run(
                    ["rocm-smi", "-d", dev, "--setpoweroverdrive",
                     str(config.power_limit_watts)],
                    check=True,
                )
                if r is None:
                    success = False
                else:
                    log.info("Set power limit to %dW", config.power_limit_watts)

            if config.gpu_clock_level is not None:
                r = self._run(
                    ["rocm-smi", "-d", dev, "--setsclk",
                     str(config.gpu_clock_level)],
                    check=True,
                )
                if r is None:
                    success = False
                else:
                    log.info("Set GPU clock level to %d", config.gpu_clock_level)

            if config.mem_clock_level is not None:
                r = self._run(
                    ["rocm-smi", "-d", dev, "--setmclk",
                     str(config.mem_clock_level)],
                    check=True,
                )
                if r is None:
                    success = False
                else:
                    log.info("Set memory clock level to %d", config.mem_clock_level)
        except Exception as exc:
            log.error("Error applying AMD config: %s", exc)
            success = False
        return success

    # -- NVIDIA --------------------------------------------------------------

    def _get_nvidia_info(self) -> GPUHardwareInfo:
        info = GPUHardwareInfo(
            vendor=GPUVendor.NVIDIA, architecture=self.arch, device_id=self.device_id
        )
        try:
            out = self._run([
                "nvidia-smi", "-i", str(self.device_id),
                "--query-gpu=name,power.draw,power.limit,power.max_limit,"
                "clocks.current.graphics,clocks.max.graphics,"
                "clocks.current.memory,clocks.max.memory,"
                "temperature.gpu,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ])
            if out:
                vals = [v.strip() for v in out.strip().split(",")]
                if len(vals) >= 11:
                    info.device_name = vals[0]
                    info.power_current_watts = float(vals[1]) if vals[1] != "[N/A]" else None
                    info.power_limit_watts = float(vals[2]) if vals[2] != "[N/A]" else None
                    info.power_max_watts = float(vals[3]) if vals[3] != "[N/A]" else None
                    info.gpu_clock_current = int(vals[4]) if vals[4] != "[N/A]" else None
                    info.gpu_clock_max = int(vals[5]) if vals[5] != "[N/A]" else None
                    info.mem_clock_current = int(vals[6]) if vals[6] != "[N/A]" else None
                    info.mem_clock_max = int(vals[7]) if vals[7] != "[N/A]" else None
                    info.temperature = float(vals[8]) if vals[8] != "[N/A]" else None
                    info.memory_total_gb = float(vals[9]) / 1024 if vals[9] != "[N/A]" else None
                    info.memory_used_gb = float(vals[10]) / 1024 if vals[10] != "[N/A]" else None
        except Exception as exc:
            log.warning("Error getting NVIDIA GPU info: %s", exc)
        return info

    def _apply_nvidia_config(self, config: GPUConfig) -> bool:
        success = True
        dev = str(config.device_id)
        try:
            if config.power_limit_watts is not None:
                r = self._run(
                    ["nvidia-smi", "-i", dev, "-pl", str(config.power_limit_watts)],
                    check=True,
                )
                if r is None:
                    success = False
                else:
                    log.info("Set power limit to %dW", config.power_limit_watts)

            if config.gpu_clock_mhz is not None:
                min_clk, max_clk = config.gpu_clock_mhz
                r = self._run(
                    ["nvidia-smi", "-i", dev, "-lgc", f"{min_clk},{max_clk}"],
                    check=True,
                )
                if r is None:
                    success = False
                else:
                    log.info("Locked GPU clocks to %d-%d MHz", min_clk, max_clk)

            if config.mem_clock_mhz is not None:
                min_clk, max_clk = config.mem_clock_mhz
                r = self._run(
                    ["nvidia-smi", "-i", dev, "-lmc", f"{min_clk},{max_clk}"],
                    check=True,
                )
                if r is None:
                    success = False
                else:
                    log.info("Locked memory clocks to %d-%d MHz", min_clk, max_clk)
        except Exception as exc:
            log.error("Error applying NVIDIA config: %s", exc)
            success = False
        return success

    # -- Common --------------------------------------------------------------

    def apply_config(self, config: GPUConfig) -> bool:
        if self.vendor == GPUVendor.AMD:
            return self._apply_amd_config(config)
        elif self.vendor == GPUVendor.NVIDIA:
            return self._apply_nvidia_config(config)
        return False

    def reset_config(self) -> bool:
        if self.vendor == GPUVendor.AMD:
            try:
                self._run(
                    ["rocm-smi", "-d", str(self.device_id), "--resetclocks"],
                    check=False,
                )
                log.info("Reset AMD GPU %d clocks", self.device_id)
                return True
            except Exception as exc:
                log.error("Error resetting AMD GPU: %s", exc)
                return False
        elif self.vendor == GPUVendor.NVIDIA:
            try:
                self._run(["nvidia-smi", "-i", str(self.device_id), "-rgc"], check=False)
                self._run(["nvidia-smi", "-i", str(self.device_id), "-rmc"], check=False)
                log.info("Reset NVIDIA GPU %d clocks", self.device_id)
                return True
            except Exception as exc:
                log.error("Error resetting NVIDIA GPU: %s", exc)
                return False
        return False

    @staticmethod
    def _run(cmd: List[str], *, check: bool = False) -> Optional[str]:
        """Run a subprocess, returning stdout on success or None on failure."""
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if check and result.returncode != 0:
            log.error("Command %s failed: %s", cmd, result.stderr.strip())
            return None
        return result.stdout


# ---------------------------------------------------------------------------
# Multi-GPU controller
# ---------------------------------------------------------------------------


class MultiGPUController:
    """Manages GPU control across multiple devices."""

    def __init__(self, device_ids: Optional[List[int]] = None):
        self.gpu_count = get_gpu_count()
        if device_ids is None:
            self.device_ids = list(range(self.gpu_count))
        else:
            self.device_ids = [d for d in device_ids if 0 <= d < self.gpu_count]

        self.controllers: Dict[int, GPUController] = {
            d: GPUController(device_id=d) for d in self.device_ids
        }

    def get_all_hardware_info(self, parallel: bool = True) -> Dict[int, GPUHardwareInfo]:
        results: Dict[int, GPUHardwareInfo] = {}
        if parallel and len(self.device_ids) > 1:
            with ThreadPoolExecutor(max_workers=len(self.device_ids)) as pool:
                futs = {
                    pool.submit(ctrl.get_hardware_info): dev_id
                    for dev_id, ctrl in self.controllers.items()
                }
                for fut in as_completed(futs):
                    dev_id = futs[fut]
                    try:
                        results[dev_id] = fut.result()
                    except Exception as exc:
                        log.error("Error getting info for GPU %d: %s", dev_id, exc)
                        results[dev_id] = GPUHardwareInfo(
                            vendor=GPUVendor.UNKNOWN, device_id=dev_id
                        )
        else:
            for dev_id, ctrl in self.controllers.items():
                try:
                    results[dev_id] = ctrl.get_hardware_info()
                except Exception as exc:
                    log.error("Error getting info for GPU %d: %s", dev_id, exc)
                    results[dev_id] = GPUHardwareInfo(
                        vendor=GPUVendor.UNKNOWN, device_id=dev_id
                    )
        return results

    def apply_config(self, config: MultiGPUConfig) -> Dict[int, bool]:
        results: Dict[int, bool] = {}
        device_ids = config.device_ids if config.device_ids else self.device_ids

        def _apply_single(dev_id: int) -> Tuple[int, bool]:
            if dev_id not in self.controllers:
                return dev_id, False
            gpu_cfg = config.get_config_for_device(dev_id)
            if gpu_cfg is None:
                return dev_id, True
            return dev_id, self.controllers[dev_id].apply_config(gpu_cfg)

        if config.parallel and len(device_ids) > 1:
            with ThreadPoolExecutor(max_workers=len(device_ids)) as pool:
                futs = [pool.submit(_apply_single, d) for d in device_ids]
                for fut in as_completed(futs):
                    dev_id, ok = fut.result()
                    results[dev_id] = ok
        else:
            for dev_id in device_ids:
                _, ok = _apply_single(dev_id)
                results[dev_id] = ok
        return results

    def reset_all(self, parallel: bool = True) -> Dict[int, bool]:
        results: Dict[int, bool] = {}

        def _reset_single(dev_id: int) -> Tuple[int, bool]:
            return dev_id, self.controllers[dev_id].reset_config()

        if parallel and len(self.device_ids) > 1:
            with ThreadPoolExecutor(max_workers=len(self.device_ids)) as pool:
                futs = [pool.submit(_reset_single, d) for d in self.device_ids]
                for fut in as_completed(futs):
                    dev_id, ok = fut.result()
                    results[dev_id] = ok
        else:
            for dev_id in self.device_ids:
                _, ok = _reset_single(dev_id)
                results[dev_id] = ok
        return results


# ---------------------------------------------------------------------------
# Public API (unchanged interface)
# ---------------------------------------------------------------------------


@dataclass
class GPUControlConfig:
    """Configuration for GPU hardware control in aorta benchmarks.

    Attributes:
        enabled: Whether GPU control is active.
        power_limit_watts: GPU power cap in watts (None = unchanged).
        gpu_clock_level: AMD clock level 0-7 (None = unchanged).
        mem_clock_level: AMD memory clock level (None = unchanged).
        gpu_clock_mhz: GPU clock range (min, max) in MHz for NVIDIA (None = unchanged).
        mem_clock_mhz: Memory clock range (min, max) in MHz for NVIDIA (None = unchanged).
        reset_on_exit: Reset GPU settings after benchmark completes.
        device_ids: Specific GPU IDs to manage (None = all available).
    """

    enabled: bool = False
    power_limit_watts: Optional[int] = None
    gpu_clock_level: Optional[int] = None
    mem_clock_level: Optional[int] = None
    gpu_clock_mhz: Optional[Tuple[int, int]] = None
    mem_clock_mhz: Optional[Tuple[int, int]] = None
    reset_on_exit: bool = True
    device_ids: Optional[List[int]] = None


class GPUControlManager:
    """Manages GPU power/frequency state for deterministic benchmarking.

    Uses direct subprocess calls to rocm-smi / nvidia-smi. Designed as a
    context manager so GPU settings are automatically restored after the
    benchmark.

    Usage::

        mgr = GPUControlManager(config)
        with mgr:
            # GPU clocks are now locked
            run_benchmark()
        # GPU clocks restored to defaults
    """

    def __init__(self, config: GPUControlConfig) -> None:
        self.config = config
        self._controller: Optional[MultiGPUController] = None
        self._applied = False

    @property
    def available(self) -> bool:
        """True if GPU control is enabled."""
        return self.config.enabled

    def apply(self) -> Dict[str, Any]:
        """Apply GPU configuration and return hardware snapshot.

        Returns:
            Dictionary with pre-benchmark GPU hardware state for metadata.
        """
        if not self.available:
            return {}

        self._controller = MultiGPUController(device_ids=self.config.device_ids)

        gpu_cfg = GPUConfig(
            power_limit_watts=self.config.power_limit_watts,
            gpu_clock_level=self.config.gpu_clock_level,
            mem_clock_level=self.config.mem_clock_level,
            gpu_clock_mhz=self.config.gpu_clock_mhz,
            mem_clock_mhz=self.config.mem_clock_mhz,
        )

        multi_cfg = MultiGPUConfig(
            default_config=gpu_cfg,
            device_ids=self.config.device_ids,
            parallel=True,
        )

        results = self._controller.apply_config(multi_cfg)
        self._applied = True

        success_count = sum(1 for v in results.values() if v)
        total = len(results)
        log.info(
            "GPU control applied to %d/%d GPUs (power=%s W, gpu_clk_level=%s, mem_clk_level=%s)",
            success_count,
            total,
            self.config.power_limit_watts,
            self.config.gpu_clock_level,
            self.config.mem_clock_level,
        )

        return self._snapshot()

    def reset(self) -> None:
        """Reset GPUs to default settings."""
        if not self._applied or self._controller is None:
            return

        results = self._controller.reset_all()
        success_count = sum(1 for v in results.values() if v)
        log.info("GPU control reset on %d/%d GPUs", success_count, len(results))
        self._applied = False

    def _snapshot(self) -> Dict[str, Any]:
        """Capture current GPU hardware state for result metadata."""
        if self._controller is None:
            return {}

        try:
            infos = self._controller.get_all_hardware_info()
            snapshot = {}
            for dev_id, info in infos.items():
                snapshot[f"gpu_{dev_id}"] = info.to_dict()
            return {"gpu_hardware_state": snapshot}
        except Exception as e:
            log.debug("Failed to capture GPU hardware snapshot: %s", e)
            return {}

    def __enter__(self) -> "GPUControlManager":
        self.apply()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.config.reset_on_exit:
            self.reset()
        return None
