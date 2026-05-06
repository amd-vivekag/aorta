"""``aorta env probe`` implementation (issue #147).

Library-first per the updated A1 spec: the primary deliverable is

* ``collect_env() -> EnvSnapshot``

which B1 (per-trial runner) and B2 (matrix runner) call **in-process** so
every trial / matrix run records its environment without shelling out.
``aorta env probe`` (CLI) is a thin wrapper around it.

Captured blocks:

* ``system_health`` -- verbatim ``rdhc --quick --json`` output (or null).
* ``rocm`` -- explicit reads of ``/opt/rocm/.info/version{,_dev}`` and
  ``/sys/module/amdgpu/version``.
* ``hip`` -- ``hipconfig`` toolchain outputs.
* ``hipblaslt`` -- commit + library hash + Tensile fingerprint + applied
  PR flags.
* ``rocblas`` -- mirror of ``hipblaslt`` for the rocBLAS library.
* ``composable_kernel`` -- two sub-blocks: ``system`` (header version +
  commit + ck_tile presence from the composablekernel-dev install) and
  ``pytorch_bundled`` (CK symbol count inside ``libtorch_hip.so``).
* ``tensile`` -- optional pip version + combined kernel-DB fingerprint
  across hipBLASLt and rocBLAS.
* ``triton``, ``fbgemm``, ``aiter`` -- Python-package version probes.
  ``fbgemm`` also surfaces the ``-DUSE_FBGEMM`` / ``-DUSE_FBGEMM_GENAI``
  build-time flags from ``torch.__config__.show()``.
* ``runtime_context`` -- container runtime + Python env detection.
* ``docker`` -- image + digest when in a container.
* ``env_vars`` -- canonical list of HSA / RCCL / FBGEMM / PyTorch vars.
* ``python_version``, ``pytorch_version``.

Fail-soft contract: ``collect_env()`` NEVER raises. Every probe that falls
back to ``None`` appends a human-readable reason to ``partial_reasons``;
the snapshot is then marked ``partial=True``. This keeps a triage matrix
running when the probe hits something missing (no rdhc, no sudo,
restricted dmesg) instead of aborting the whole run.

No GPU compute. No tensor allocations. Target wall time: <15 s with rdhc
present, <5 s without. Every capture function returns a fully-shaped dict
with ``None`` for missing values, so the schema is stable: keys never go
missing across environments.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


SCHEMA_VERSION = "1.1"
# 1.0 -> 1.1 (this commit):
#   - Renamed hipblaslt/rocblas/miopen.commit -> .rocm_release_tweak
#     (the field's value is the ROCm-release-shared identifier, not a
#     per-library upstream commit -- the old name misled consumers).
#   - Renamed hipblaslt/rocblas.tensile_yaml_revision -> .kernel_db_revision
#     (modern builds ship .dat not .yaml; matches miopen's naming).
#   - Removed USE_ROCM_CK_SDPA from env_vars (it's a build-time cmake
#     flag, not a runtime env var). Replaced with `pytorch_use_ck_sdpa`
#     and `pytorch_use_ck_gemm` booleans inside the composable_kernel
#     block, parsed from `torch.__config__.show()` (same pattern as the
#     existing `fbgemm.pytorch_use_fbgemm*` fields).
#   - Added top-level `host`, `miopen`, `rccl`, `gpu_arch`, `aotriton`,
#     `pytorch_build`, `composable_kernel`, `tensile`, `triton`,
#     `fbgemm`, `aiter`, `rocblas` blocks. All purely additive.
#   - Expanded CANONICAL_ENV_VARS across GPU scoping
#     (HIP_VISIBLE_DEVICES / ROCR_VISIBLE_DEVICES /
#     HSA_OVERRIDE_GFX_VERSION), launch (HIP_LAUNCH_BLOCKING), build
#     target (PYTORCH_ROCM_ARCH), MIOpen kernel-DB selection
#     (MIOPEN_SYSTEM_DB_PATH / MIOPEN_USER_DB_PATH /
#     MIOPEN_DEBUG_DISABLE_FIND_DB / MIOPEN_FIND_MODE), SDPA backend
#     (TORCH_ROCM_FA_PREFER_CK / TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL),
#     hipBLASLt autotune (TORCH_BLAS_PREFER_HIPBLASLT /
#     TORCH_HIPBLASLT_TUNING_FILE / TORCH_HIPBLASLT_TUNING_OVERRIDE_FILE),
#     and NCCL/RCCL extras (NCCL_P2P_LEVEL / NCCL_IB_HCA /
#     NCCL_SOCKET_IFNAME / RCCL_MSCCL_ENABLE). Removed USE_ROCM_CK_SDPA
#     (build-time cmake flag, see above). See CANONICAL_ENV_VARS for
#     the full set.
#   - host.glibc_version no longer carries the redundant "glibc "
#     prefix; the value is the bare version string (e.g. "2.35").

# RDHC subprocess budget. The issue caps at 30 s; we keep that to stay
# inside the 30 s worst-case env probe budget.
RDHC_TIMEOUT_SEC = 30.0

# Pointer appended to install-related rdhc partial_reasons so operators
# who hit ``system_health: null`` find the install + sudo recipe right
# away. Not appended to timeout / parse-failure reasons -- those are
# rdhc-side runtime issues, not install issues.
_RDHC_INSTALL_HINT = "see docs/env-probe.md#installing-rdhc"

# Generic per-subprocess budget for hipconfig, dpkg, etc. None of these
# should take more than a second on a healthy host.
SHORT_TIMEOUT_SEC = 5.0

# Larger budget for subprocesses that scan multi-hundred-MB binaries.
# `nm -D` over a ~400 MB libtorch_hip.so finishes in ~1 s on a warm
# page cache but can stretch past 5 s on cold cache or contended I/O;
# `c++filt` demangling the resulting symbol list is similarly bursty.
# Both falling back silently to "no CK in PyTorch" when they timeout
# would look identical to a CPU-only wheel -- so give them headroom.
NM_TIMEOUT_SEC = 30.0

# Canonical env var list -- explicit, NOT prefix matching. Workload
# config (AMP_DTYPE, MODEL_DTYPE, SHAMPOO_PRECONDITIONER_DTYPE) belongs
# in the trial result emitted by ``aorta run`` (Task B1), so it is
# deliberately absent here. Asserted by tests.
CANONICAL_ENV_VARS: tuple[str, ...] = (
    # GPU scoping (most common cause of "you see N GPUs, I see M")
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    # HSA / runtime
    "HSA_XNACK",
    "HSA_KERNARG_POOL_SIZE",
    "HSA_NO_SCRATCH_RECLAIM",
    "HSA_OVERRIDE_GFX_VERSION",  # forces a different gfx target than the silicon
    # GPU queue / codegen / build target
    "GPU_MAX_HW_QUEUES",
    "AMDGCN_USE_BUFFER_OPS",
    "DISABLE_TF32",
    "PYTORCH_ROCM_ARCH",
    "HIP_LAUNCH_BLOCKING",  # forces synchronous launches; trace-debug leftover
    # RCCL / NCCL
    "NCCL_MAX_NCHANNELS",
    "NCCL_P2P_LEVEL",
    "NCCL_IB_HCA",
    "NCCL_SOCKET_IFNAME",
    "RCCL_MSCCL_ENABLE",
    # FBGEMM
    "FBGEMM_NO_JK",
    "FBGEMM_TBE_V2",
    "FBGEMM_TBE_ROCM_HIP_BACKWARD_KERNEL",
    "FBGEMM_BOUNDS_CHECK_INDICES_V2",
    # MIOpen kernel DB + selection-mode
    "MIOPEN_SYSTEM_DB_PATH",
    "MIOPEN_USER_DB_PATH",
    "MIOPEN_DEBUG_DISABLE_FIND_DB",
    "MIOPEN_FIND_MODE",  # NORMAL/FAST/HYBRID/DYNAMIC -> different conv kernels
    # SDPA / Flash Attention backend selection (CK vs AOTriton).
    # Note: USE_ROCM_CK_SDPA / USE_ROCM_CK_GEMM are NOT here -- they're
    # build-time cmake flags consumed when the PyTorch wheel is built,
    # not runtime env vars. Captured under
    # composable_kernel.{pytorch_use_ck_sdpa,pytorch_use_ck_gemm}.
    "TORCH_ROCM_FA_PREFER_CK",
    "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL",
    # GEMM backend preference + hipBLASLt autotune pinning
    "TORCH_BLAS_PREFER_HIPBLASLT",
    "TORCH_HIPBLASLT_TUNING_FILE",
    "TORCH_HIPBLASLT_TUNING_OVERRIDE_FILE",
    # PyTorch / inductor
    "TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE",
    "PYTORCH_CUDA_ALLOC_CONF",
)

# Filesystem locations -- collected here so tests can monkeypatch them.
# Each constant is paired with a short note on the source so future
# editors don't have to rediscover where the data comes from.
#
# All paths verified against:
#   - host: ROCm 7.2.1 baremetal install
#   - rocm/pytorch:7.2.0 docker image
#   - rocm/pytorch:7.0.2.1 docker image (`version_dev` legitimately absent)
# Each path is absolute; the structural test in
# tests/instrumentation/test_environment.py::TestPathConstants
# enforces this so a future relative-path typo fails fast.

# Per #147 schema: "Explicit ROCm version files".
# /opt/rocm is conventionally a symlink to the active versioned install
# (e.g., /opt/rocm-7.2.1), so /opt/rocm/.info/version always points at the
# active release.
# Canonical ROCm bin dir used for fallback lookup of binaries (e.g.
# rocm_agent_enumerator) when the operator's PATH doesn't include
# /opt/rocm/bin (a common state when /etc/profile.d/rocm.sh hasn't
# been sourced -- happens in non-login shells).
ROCM_BIN_DIR = Path("/opt/rocm/bin")
ROCM_VERSION_FILE = Path("/opt/rocm/.info/version")            # release tag, e.g. "7.2.1"
ROCM_VERSION_DEV_FILE = Path("/opt/rocm/.info/version-dev")    # full build, e.g. "7.2.1-43"
# Linux kernel-side AMDGPU module version (KMD = kernel-mode driver).
# Provided by the kernel since the amdgpu module exposes a sysfs `version`.
KMD_VERSION_FILE = Path("/sys/module/amdgpu/version")          # e.g. "6.16.13"

# hipBLASLt build identity sources. The version header ships in the
# hipblaslt-dev package; on hosts without it, _capture_hipblaslt's commit
# / package_version fields fall back to None with a recorded reason.
HIPBLASLT_VERSION_HEADER = Path(
    "/opt/rocm/include/hipblaslt/hipblaslt-version.h"
)
HIPBLASLT_LIB_DIR = Path("/opt/rocm/lib")  # libhipblaslt.so* lives here
HIPBLASLT_TENSILE_DIR = Path(
    "/opt/rocm/lib/hipblaslt/library"
)  # Tensile kernel database (.dat on modern builds, .yaml on older)

# rocBLAS build identity sources. Mirrors the hipBLASLt layout exactly --
# header in the rocblas-dev package (note the `internal/` subdir, unlike
# hipblaslt), runtime lib in /opt/rocm/lib, and a Tensile kernel database
# at /opt/rocm/lib/rocblas/library/. The same fail-soft contract applies:
# missing -dev package -> commit/package_version fall back to None with a
# recorded reason; the runtime lib + kernel DB ship with the runtime
# rocblas package and are usually still present in stripped images.
ROCBLAS_VERSION_HEADER = Path(
    "/opt/rocm/include/rocblas/internal/rocblas-version.h"
)
ROCBLAS_LIB_DIR = Path("/opt/rocm/lib")  # librocblas.so* lives here
ROCBLAS_TENSILE_DIR = Path(
    "/opt/rocm/lib/rocblas/library"
)  # rocBLAS Tensile kernel database

# Composable Kernel (CK) is shipped header-only via the
# composablekernel-dev package -- there is no libck.so to hash. CK_TILE
# is a sub-component of CK with no separate version header; we record
# its presence as a boolean by checking for its core config header.
# When the -dev package is stripped from the container, both keys fall
# back to None / False with a recorded reason.
CK_VERSION_HEADER = Path("/opt/rocm/include/ck/version.h")
CK_TILE_CONFIG_HEADER = Path("/opt/rocm/include/ck_tile/core/config.hpp")

# Filename of the PyTorch-built HIP shared library, looked up relative to
# `torch.__file__` at runtime by the composable_kernel probe (we never
# initialise HIP context to find it -- pure path arithmetic).
PYTORCH_HIP_LIB_NAME = "libtorch_hip.so"

# AOTriton is the default Flash Attention backend on ROCm. Unlike CK
# / FBGEMM / AITER, AOTriton is NOT a PyTorch git submodule -- it's
# fetched at PyTorch build time via cmake/External/aotriton.cmake and
# bundled into the wheel as <torch>/lib/libaotriton_v2.so.MAJOR.MINOR.PATCH
# alongside an `aotriton.images/` directory of pre-compiled kernel images.
# Version is parsed from the filename (no header in the wheel install).
# The AOTRITON_INSTALLED_PREFIX env var lets operators point PyTorch at
# a system AOTriton install; we record its value so cross-env diffs
# surface the override.
AOTRITON_LIB_PREFIX = "libaotriton_v2.so"
AOTRITON_IMAGES_DIR_NAME = "aotriton.images"
AOTRITON_INSTALLED_PREFIX_ENV = "AOTRITON_INSTALLED_PREFIX"

# MIOpen build identity sources. Same shape as hipBLASLt: header in the
# -dev package, runtime lib in /opt/rocm/lib, kernel database under
# /opt/rocm/share/miopen/db/. Kernel DB files are .txt / .fdb.txt
# (gfx-target named) -- distinct suffix family from Tensile's .dat/.yaml.
MIOPEN_VERSION_HEADER = Path("/opt/rocm/include/miopen/version.h")
MIOPEN_LIB_DIR = Path("/opt/rocm/lib")  # libMIOpen.so* lives here (capital M, capital O)
MIOPEN_KERNEL_DB_DIR = Path("/opt/rocm/share/miopen/db")
MIOPEN_KERNEL_DB_SUFFIXES: tuple[str, ...] = (".txt",)  # .fdb.txt also matches via final suffix

# RCCL is AMD's NCCL-compatible collective comms library. Header is at
# /opt/rocm/include/rccl/rccl.h (same name as upstream NCCL's). Version
# is encoded as a single integer macro NCCL_VERSION_CODE -- decoded
# below via _decode_nccl_version_code.
RCCL_VERSION_HEADER = Path("/opt/rocm/include/rccl/rccl.h")
RCCL_LIB_DIR = Path("/opt/rocm/lib")  # librccl.so*

# `rocm_agent_enumerator` returns one gfx-target name per detected GPU
# (e.g. "gfx942\ngfx942\n..."). Works without /dev/kfd access on most
# hosts (the kernel module exposes the architecture via sysfs). Probed
# via subprocess; fail-soft to None when binary missing or returns empty.
ROCM_AGENT_ENUMERATOR_BIN = "rocm_agent_enumerator"

# Env var an operator sets to point at a PyTorch source checkout. When
# present and `<path>/third_party/` exists, the pytorch_build probe runs
# `git -C <src>/third_party/<sub> rev-parse HEAD` for each canonical
# submodule below, recording the actual bound commit SHAs. Without it,
# pip-installed wheels fall back to the GitHub-URL recovery hint.
AORTA_PYTORCH_SRC_ENV = "AORTA_PYTORCH_SRC"

# Canonical AMD-relevant submodules under PyTorch's third_party/.
# Verified against pytorch/.gitmodules on `main` (Dec 2025): these are
# the AMD/ROCm-relevant entries. Adding a new submodule is a deliberate
# schema-shape change -- mirror the CANONICAL_ENV_VARS workflow:
#   1. add to this tuple
#   2. update `test_canonical_pytorch_submodules_stable`
#   3. justify in PR description
# Cross-vendor (fbgemm) is included because its ROCm path is a major
# drift surface; the upstream commit can drift independently of PyTorch
# even when both are on the same release tag.
CANONICAL_PYTORCH_SUBMODULES: tuple[str, ...] = (
    "composable_kernel",
    "aiter",
    "fbgemm",
)

# GitHub URL template printed in `partial_reasons` when the PyTorch
# source tree is not on disk. The operator reading the partial entry
# can substitute the captured `git_commit` and resolve the bound
# submodule SHAs via the GitHub web tree without leaving the doc.
_PYTORCH_SUBMODULE_LOOKUP_HINT = (
    "github.com/pytorch/pytorch/tree/<git_commit>/third_party/<name>"
)

# Container runtime detection markers.
# /.dockerenv -- created by Docker since the early days.
# /run/.containerenv -- created by Podman.
# /proc/1/cgroup -- the init process's cgroup; last-resort cgroup token
#   sniff for the runtime *type* (docker / podman / singularity), per
#   OCI conventions.
# /proc/self/cgroup -- the current process's cgroup, parsed for the
#   container *ID* (a 12-64 hex SHA segment). Distinct file, distinct
#   purpose -- both go in the constants block so tests can monkeypatch
#   either without touching the real /proc.
DOCKERENV_MARKER = Path("/.dockerenv")
PODMAN_CONTAINERENV_MARKER = Path("/run/.containerenv")
CGROUP_FILE = Path("/proc/1/cgroup")
SELF_CGROUP_FILE = Path("/proc/self/cgroup")


# ---------------------------------------------------------------------------
# EnvSnapshot dataclass -- the public typed object B1/B2/CLI consume
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EnvSnapshot:
    """Wraps the env.json schema as a typed object.

    Attributes mirror the env.json keys 1-to-1; ``to_dict()`` / ``from_dict()``
    round-trip losslessly. The dataclass is ``frozen=True``, which prevents
    *attribute rebinding* on the snapshot itself (``snap.rocm = ...`` raises),
    but it does NOT deep-freeze the nested ``dict`` / ``list`` containers --
    callers can still mutate ``snap.rocm["version"] = ...`` or
    ``snap.partial_reasons.append(...)`` in place. Treat embedded snapshots
    as read-only; if you need to modify, deep-copy first via
    ``EnvSnapshot.from_dict(copy.deepcopy(snap.to_dict()))``.

    The two fail-soft fields make the snapshot honest about partial captures:

    * ``partial`` -- True if at least one probe fell back to None when it was
      expected to populate. False on a clean probe.
    * ``partial_reasons`` -- one human-readable string per fallback. The list
      is empty when ``partial`` is False. Each entry names the field plus a
      short cause (e.g. ``"system_health: rdhc not on PATH"``).

    "Documented absences" do NOT trigger partial:

    * ``docker == None`` on baremetal (no container, nothing to record)
    * ``env_vars[X] == None`` for an unset env var (the documented contract)
    * ``runtime_context.venv_path == None`` outside a venv

    "Probe fell back" cases that DO trigger partial:

    * ``system_health == None`` (rdhc unavailable / no sudo / timeout)
    * any field in ``rocm`` / ``hip`` / ``hipblaslt`` is None
    * ``pytorch_version == None`` (torch not installed)
    * ``docker.image`` / ``docker.digest`` is None when inside a container
    """

    schema_version: str
    captured_at: str
    system_health: dict | None
    rocm: dict
    hip: dict
    hipblaslt: dict
    rocblas: dict
    composable_kernel: dict
    tensile: dict
    triton: dict
    fbgemm: dict
    aiter: dict
    aotriton: dict
    miopen: dict
    rccl: dict
    gpu_arch: dict
    runtime_context: dict
    host: dict
    docker: dict | None
    env_vars: dict[str, str | None]
    python_version: str
    pytorch_version: str | None
    pytorch_build: dict
    partial: bool
    partial_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to the env.json shape. Round-trip pair with from_dict."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EnvSnapshot:
        """Reconstruct from a previously serialised env.json dict.

        Tolerates extra unknown keys (forward-compat) and missing optional
        ``partial_reasons`` (defaults to empty list).
        """
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in d.items() if k in known}
        kwargs.setdefault("partial_reasons", [])
        return cls(**kwargs)

    def summary(self) -> str:
        """Human-friendly multi-line summary for CLI / logs.

        Fixed-width labels, ~18 lines on a populated host (one cell
        per top-level block). Used by
        ``aorta env probe`` to print a brief after writing the JSON.

        Covers every top-level identity block so an operator running the
        probe doesn't have to ``jq env.json`` to see the new GEMM /
        kernel-library fingerprints. Hashes are truncated to 6 hex chars
        for line-width; the full values live in the JSON.
        """
        rt = self.runtime_context or {}
        rocm = self.rocm or {}
        hip = self.hip or {}
        hipblaslt = self.hipblaslt or {}
        rocblas = self.rocblas or {}
        ck = self.composable_kernel or {}
        ck_sys = ck.get("system") or {}
        ck_bundled = ck.get("pytorch_bundled") or {}
        tensile = self.tensile or {}
        triton = self.triton or {}
        fbgemm = self.fbgemm or {}
        aiter = self.aiter or {}
        aotriton = self.aotriton or {}
        miopen = self.miopen or {}
        rccl = self.rccl or {}
        gpu_arch = self.gpu_arch or {}
        host = self.host or {}
        # Use ``is not None`` -- RDHC may return an empty dict on a healthy
        # host with nothing to report, which is still a successful capture
        # and must NOT be summarised as "unavailable".
        sysh = (
            "present"
            if self.system_health is not None
            else "unavailable (system_health=null)"
        )
        # Partial state is signalled in two places already: the "Wrote
        # env probe to ... [PARTIAL]" header line printed by the CLI
        # before the summary, and the closing "[PARTIAL, N reason(s)]"
        # marker printed after the partial_reasons list. A third copy
        # on the runtime: line was redundant.

        def short_hash(value: str | None) -> str:
            """Truncate ``filenames-sha256:abcdef…``-style hashes for display."""
            if not value:
                return str(value)
            if ":" in value:
                kind, hex_part = value.split(":", 1)
                if len(hex_part) > 8:
                    return f"{kind}:{hex_part[:6]}…"
            return value

        def pkg_state(version: str | None) -> str:
            """Render a Python-package version cell self-explanatorily.

            `pip=None` was confusing -- the `None` is the JSON null
            value, but operators read it as "no info captured". Replace
            with a literal "(not installed)" marker so the meaning is
            obvious without needing to read the schema.
            """
            return version if version else "(not installed)"

        return "\n".join(
            (
                f"  runtime:   {rt.get('type', '?')} / python={rt.get('python_env', '?')}",
                f"  rocm:      {rocm.get('version', '?')} (dev: {rocm.get('version_dev', '?')})",
                f"  hip:       {hip.get('version', '?')} ({hip.get('platform', '?')})",
                # ROCm release tweak (HIPBLASLT_VERSION_TWEAK et al.)
                # is the same string across every library in a release,
                # not a per-library upstream commit. lib_hash is the
                # per-binary signal (in the JSON, not the brief).
                f"  hipblaslt: {hipblaslt.get('package_version', '?')} rocm_release_tweak={hipblaslt.get('rocm_release_tweak', '?')}",
                f"  rocblas:   {rocblas.get('package_version', '?')} rocm_release_tweak={rocblas.get('rocm_release_tweak', '?')}",
                f"  miopen:    {miopen.get('package_version', '?')} rocm_release_tweak={miopen.get('rocm_release_tweak', '?')}",
                f"  rccl:      {rccl.get('version', '?')} (code={rccl.get('version_code', '?')})",
                # gpu_arch: dedup'd targets are the meaningful diff
                # signal (homogeneous vs mixed-arch hosts); count tells
                # how many GPUs were detected. Both null when the
                # rocm_agent_enumerator probe failed.
                f"  gpu_arch:  {gpu_arch.get('gfx_targets') or '?'} "
                f"(counts={gpu_arch.get('agent_arch_counts') or '?'})",
                f"  host:      kernel={host.get('kernel_release') or '?'} "
                f"machine={host.get('machine') or '?'}  "
                f"glibc={host.get('glibc_version') or '?'}",
                # CK has two layers -- system headers (composablekernel-dev
                # apt pkg) and the copy compiled into libtorch_hip.so via
                # PyTorch's third_party/composable_kernel submodule. Both
                # can drift independently.
                f"  ck:        system={ck_sys.get('version', '?')}/{(ck_sys.get('commit') or '?')[:8]}  "
                f"ck_tile={'yes' if ck_sys.get('ck_tile_present') else 'no'}  "
                f"libtorch_hip={ck_bundled.get('symbol_count', '?')} ck:: syms",
                # Tensile generates kernels at rocBLAS / hipBLASLt build
                # time; we fingerprint the union of their kernel DBs.
                # The Tensile pip package itself is a build tool --
                # almost never on production hosts -- so the
                # not-installed state is annotated as expected.
                f"  tensile:   kernel_db={short_hash(tensile.get('kernel_db_combined_hash'))}  "
                f"[Tensile pip pkg: {pkg_state(tensile.get('package_version'))}; "
                f"build-time tool, normal]",
                f"  triton:    {pkg_state(triton.get('package_version'))}",
                # FBGEMM has TWO surfaces and they're different artifacts:
                #   1) FBGEMM the C++ lib is vendored inside the PyTorch
                #      wheel (third_party/fbgemm) -- the USE_FBGEMM /
                #      USE_FBGEMM_GENAI flags below confirm whether the
                #      build pulled it in.
                #   2) `fbgemm_gpu` the PyPI Python package is a
                #      separate distribution; rarely needed alongside
                #      PyTorch. Annotate the not-installed state so a
                #      reader doesn't think "FBGEMM is missing" when
                #      USE_FBGEMM=True (which means it IS here).
                f"  fbgemm:    in PyTorch: USE_FBGEMM={fbgemm.get('pytorch_use_fbgemm')} "
                f"USE_FBGEMM_GENAI={fbgemm.get('pytorch_use_fbgemm_genai')}  "
                f"[fbgemm_gpu pip pkg: {pkg_state(fbgemm.get('package_version'))}; "
                f"separate from torch's vendored copy]",
                # AITER (AMD's CK-based inference kernel lib) is
                # optional -- some inference stacks pull it in, training
                # / stock inference don't. Annotate to avoid alarming
                # readers who see "not installed".
                f"  aiter:     {pkg_state(aiter.get('package_version'))} "
                f"[aiter pip pkg; optional ROCm inference lib]",
                # AOTriton: default ROCm Flash Attention backend.
                # Bundled in the wheel; system override possible via
                # AOTRITON_INSTALLED_PREFIX (rare).
                f"  aotriton:  bundled={aotriton.get('bundled_version', '?')} "
                f"present={aotriton.get('bundled_present')} "
                f"images_dir={aotriton.get('bundled_images_dir_present')}  "
                f"[AOTRITON_INSTALLED_PREFIX={aotriton.get('installed_prefix') or '(unset)'}]",
                f"  rdhc:      {sysh}",
                f"  python:    {self.python_version} | pytorch: {self.pytorch_version}",
                f"  torch build: {self._summary_pytorch_build_line()}",
            )
        )

    def _summary_pytorch_build_line(self) -> str:
        """Single-line summary of the structured pytorch_build block."""
        pb = self.pytorch_build or {}
        commit = pb.get("git_commit")
        commit_short = commit[:8] if commit else "?"
        kind = pb.get("install_kind", "?")
        subs = pb.get("submodule_commits") or {}
        sub_source = subs.get("_source")
        if sub_source == "git":
            sub_pairs = []
            for name, sha in subs.items():
                if name == "_source" or not sha:
                    continue
                sub_pairs.append(f"{name}={sha[:8]}")
            sub_str = " ".join(sub_pairs) if sub_pairs else "(none)"
            return (
                f"git_commit={commit_short} install={kind} | "
                f"submodules({sub_source}): {sub_str}"
            )
        return (
            f"git_commit={commit_short} install={kind} "
            f"[third_party SHAs: set AORTA_PYTORCH_SRC=<src> or look up "
            f"github.com/pytorch/pytorch/tree/{commit_short}/third_party/<name>]"
        )


# ---------------------------------------------------------------------------
# collect_env -- the public entrypoint B1 / B2 / CLI all call
# ---------------------------------------------------------------------------


def collect_env() -> EnvSnapshot:
    """Capture the current process environment as an :class:`EnvSnapshot`.

    NEVER raises. The promise is enforced two ways:

    * Every probe is individually fail-soft (returns None or a shaped
      dict; appends a human-readable reason to a shared list on fallback).
    * The orchestrator body is wrapped in a top-level ``try / except
      Exception``. If anything genuinely unexpected raises (a stdlib call
      misbehaves, a probe is buggy, etc.), the disaster-recovery helper
      :func:`_disaster_snapshot` constructs a minimally-shaped
      :class:`EnvSnapshot` with ``partial=True`` and the original
      exception captured in ``partial_reasons``. Callers (B1 dispatcher,
      B2 matrix runner, CLI) always get back a valid object.

    No GPU compute. No tensor allocations. The optional ``import torch``
    for the version probe does NOT initialise CUDA / HIP context.
    """
    reasons: list[str] = []
    try:
        runtime_context = _detect_runtime_context()  # never partial; always populates
        system_health = _run_rdhc(reasons)
        rocm = _capture_rocm_version_files(reasons)
        hip = _capture_hip_toolchain(reasons)
        hipblaslt = _capture_hipblaslt(reasons)
        rocblas = _capture_rocblas(reasons)
        composable_kernel = _capture_composable_kernel(reasons)
        tensile = _capture_tensile(reasons)
        triton = _capture_triton(reasons)
        fbgemm = _capture_fbgemm(reasons)
        aiter = _capture_aiter(reasons)
        aotriton = _capture_aotriton(reasons)
        miopen = _capture_miopen(reasons)
        rccl = _capture_rccl(reasons)
        gpu_arch = _capture_gpu_arch(reasons)
        host = _capture_host(reasons)
        docker = _capture_docker_metadata(runtime_context, reasons)
        env_vars = _capture_env_vars()  # individual nulls are documented, not partial
        pytorch_version = _capture_pytorch_version(reasons)
        pytorch_build = _capture_pytorch_build(reasons)

        return EnvSnapshot(
            schema_version=SCHEMA_VERSION,
            captured_at=_utc_now_iso(),
            system_health=system_health,
            rocm=rocm,
            hip=hip,
            hipblaslt=hipblaslt,
            rocblas=rocblas,
            composable_kernel=composable_kernel,
            tensile=tensile,
            triton=triton,
            fbgemm=fbgemm,
            aiter=aiter,
            aotriton=aotriton,
            miopen=miopen,
            rccl=rccl,
            gpu_arch=gpu_arch,
            host=host,
            runtime_context=runtime_context,
            docker=docker,
            env_vars=env_vars,
            python_version=platform.python_version(),
            pytorch_version=pytorch_version,
            pytorch_build=pytorch_build,
            partial=bool(reasons),
            partial_reasons=reasons,
        )
    except Exception as exc:  # noqa: BLE001 -- this is the never-raises gate
        log.info("collect_env() hit unexpected exception", exc_info=True)
        return _disaster_snapshot(
            preceding_reasons=reasons,
            unexpected_reason=(
                f"collect_env: unexpected failure "
                f"({type(exc).__name__}: {exc})"
            ),
        )


def _disaster_snapshot(
    preceding_reasons: list[str], unexpected_reason: str
) -> EnvSnapshot:
    """Return a minimally-shaped EnvSnapshot when collect_env crashes.

    Used by the never-raises top-level guard. Every field gets a sane
    null/empty default so downstream consumers (B1, B2, jq pipelines)
    still see the schema they expect, with ``partial=True`` and the
    triggering exception in ``partial_reasons``.

    Even helpers used here are guarded -- if ``_utc_now_iso`` or
    ``platform.python_version`` themselves raise, we fall back to empty
    strings rather than re-throw.
    """
    try:
        captured_at = _utc_now_iso()
    except Exception:  # noqa: BLE001
        captured_at = ""
    try:
        python_version = platform.python_version()
    except Exception:  # noqa: BLE001
        python_version = ""

    return EnvSnapshot(
        schema_version=SCHEMA_VERSION,
        captured_at=captured_at,
        system_health=None,
        rocm={"version": None, "version_dev": None, "kmd_version": None},
        hip={
            "version": None,
            "platform": None,
            "compiler": None,
            "runtime": None,
            "cpp_config": None,
        },
        hipblaslt={
            "rocm_release_tweak": None,
            "package_version": None,
            "lib_hash": None,
            "kernel_db_revision": None,
            "applied_prs": {},
        },
        rocblas={
            "rocm_release_tweak": None,
            "package_version": None,
            "lib_hash": None,
            "kernel_db_revision": None,
            "applied_prs": {},
        },
        composable_kernel={
            "system": {
                "version": None,
                "commit": None,
                "ck_tile_present": False,
            },
            "pytorch_bundled": {"present": False, "symbol_count": None},
            "pytorch_use_ck_sdpa": None,
            "pytorch_use_ck_gemm": None,
        },
        tensile={"package_version": None, "kernel_db_combined_hash": None},
        triton={"package_version": None},
        fbgemm={
            "package_version": None,
            "pytorch_use_fbgemm": None,
            "pytorch_use_fbgemm_genai": None,
        },
        aiter={"package_version": None},
        aotriton={
            "bundled_present": False,
            "bundled_version": None,
            "bundled_lib_hash": None,
            "bundled_images_dir_present": False,
            "installed_prefix": None,
        },
        miopen={
            "rocm_release_tweak": None,
            "package_version": None,
            "lib_hash": None,
            "kernel_db_revision": None,
        },
        rccl={
            "version_code": None,
            "version": None,
            "lib_hash": None,
        },
        gpu_arch={
            "agent_count": None,
            "gfx_targets": None,
            "agent_arch_counts": None,
        },
        runtime_context={
            "type": "baremetal",
            "python_env": "system",
            "venv_path": None,
            "conda_env_name": None,
        },
        host={
            "kernel_release": None,
            "kernel_version": None,
            "machine": None,
            "glibc_version": None,
        },
        docker=None,
        env_vars=dict.fromkeys(CANONICAL_ENV_VARS),
        python_version=python_version,
        pytorch_version=None,
        pytorch_build={
            "git_commit": None,
            "hip_version": None,
            "cuda_version": None,
            "debug": None,
            "install_kind": "unknown",
            "source_path": None,
            "submodule_commits": {
                **{name: None for name in CANONICAL_PYTORCH_SUBMODULES},
                "_source": None,
            },
        },
        partial=True,
        partial_reasons=[*preceding_reasons, unexpected_reason],
    )


def _utc_now_iso() -> str:
    """ISO-8601 UTC timestamp with trailing 'Z' (per #147 schema example)."""
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


# ---------------------------------------------------------------------------
# RDHC wrapper
# ---------------------------------------------------------------------------


def _run_rdhc(reasons: list[str]) -> dict | None:
    """Run ``sudo -n -E rdhc --quick --json <tmp>`` and return parsed dict.

    Manages its own temp file via :mod:`tempfile` -- nothing leaks into
    the env probe's output directory.

    Returns ``None`` on any of:
    * RDHC not installed (``shutil.which`` returns nothing for ``rdhc.py``
      *and* ``rdhc``).
    * ``sudo -n`` would prompt for a password (return code != 0).
    * RDHC takes longer than ``RDHC_TIMEOUT_SEC``.
    * RDHC exits non-zero or produces malformed JSON.

    Each failure mode appends one human-readable entry to ``reasons`` AND
    logs a single INFO line. Never raises.
    """
    rdhc = shutil.which("rdhc.py") or shutil.which("rdhc")
    if rdhc is None:
        msg = f"system_health: rdhc not on PATH ({_RDHC_INSTALL_HINT})"
        log.info(msg)
        reasons.append(msg)
        return None

    # Tempfile creation is part of the never-raises surface: a read-only or
    # full /tmp would otherwise abort collect_env(). Treat as "rdhc
    # unavailable" with a clear reason.
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".json", prefix="rdhc_quick_", delete=False
        ) as tmp:
            tmp_path = Path(tmp.name)
    except OSError as exc:
        msg = f"system_health: failed to create rdhc temp file ({exc})"
        log.info(msg)
        reasons.append(msg)
        return None

    try:
        cmd = ["sudo", "-n", "-E", rdhc, "--quick", "--json", str(tmp_path)]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=RDHC_TIMEOUT_SEC,
                check=False,
            )
        except subprocess.TimeoutExpired:
            msg = f"system_health: rdhc exceeded {RDHC_TIMEOUT_SEC:.0f}s timeout"
            log.info(msg)
            reasons.append(msg)
            return None
        except (FileNotFoundError, OSError) as exc:
            # Same actionable hint as the not-on-PATH branch -- if exec
            # itself fails (e.g. broken interpreter shebang in a stripped
            # image), reinstalling rocm-systems is usually the fix.
            msg = (
                f"system_health: failed to invoke rdhc ({exc}) "
                f"({_RDHC_INSTALL_HINT})"
            )
            log.info(msg)
            reasons.append(msg)
            return None

        if result.returncode != 0:
            # Pull the last non-empty line of stderr; truncate to keep
            # partial_reasons readable in CLI output and JSON. Falls back
            # to "(no stderr; likely sudo-n unavailable)" when the child
            # printed nothing -- which is what the no-password case does.
            # The install hint is only appended for the no-stderr case
            # (where sudo config is the likely fix); when rdhc DOES print
            # to stderr the operator should debug from that, not from a
            # generic install link.
            stderr_lines = (result.stderr or "").splitlines()
            stderr_tail = next(
                (line.strip() for line in reversed(stderr_lines) if line.strip()),
                "",
            )
            if stderr_tail:
                detail = f"stderr: {stderr_tail[:200]}"
            else:
                detail = (
                    f"no stderr; likely sudo-n unavailable ({_RDHC_INSTALL_HINT})"
                )
            msg = (
                f"system_health: rdhc exited {result.returncode} ({detail})"
            )
            log.info(msg)
            reasons.append(msg)
            return None

        try:
            return json.loads(tmp_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
            msg = f"system_health: rdhc output not parseable ({exc})"
            log.info(msg)
            reasons.append(msg)
            return None
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# ROCm version files
# ---------------------------------------------------------------------------


def _capture_rocm_version_files(reasons: list[str]) -> dict[str, str | None]:
    """Read ROCm version markers directly from disk.

    These are explicit reads (not via RDHC) so that ``rocm.version`` is
    populated even when RDHC is unavailable. All three keys are always
    present; missing files yield ``None`` and append a reason.
    """
    block = {
        "version": _read_text_file(ROCM_VERSION_FILE),
        "version_dev": _read_text_file(ROCM_VERSION_DEV_FILE),
        "kmd_version": _read_text_file(KMD_VERSION_FILE),
    }
    paths = {
        "version": ROCM_VERSION_FILE,
        "version_dev": ROCM_VERSION_DEV_FILE,
        "kmd_version": KMD_VERSION_FILE,
    }
    # Note: _read_text_file returns None for missing, empty, permission
    # denied, AND non-utf8 cases. Reason wording covers all four so the
    # operator does not assume "missing" when the file is just empty.
    for key, value in block.items():
        if value is None:
            reasons.append(
                f"rocm.{key}: {paths[key]} missing, empty, or unreadable"
            )
    return block


def _read_text_file(path: Path) -> str | None:
    """Read a small text file; return its stripped contents or ``None``.

    Part of the ``never raises`` surface: catches everything that can come
    out of ``Path.read_text``, including ``UnicodeDecodeError`` from files
    with non-UTF8 bytes (e.g. a corrupt ``/sys/module/amdgpu/version``
    or a locale-mismatched ``/opt/rocm/.info/*``). Returns ``None`` for
    every error path so the caller can record a partial reason instead of
    crashing.
    """
    try:
        text = path.read_text(encoding="utf-8").strip()
        return text or None
    except (FileNotFoundError, PermissionError, IsADirectoryError):
        return None
    except UnicodeDecodeError as exc:
        log.debug("non-utf8 contents in %s: %s", path, exc)
        return None
    except OSError as exc:
        log.debug("read failed for %s: %s", path, exc)
        return None


# ---------------------------------------------------------------------------
# HIP toolchain
# ---------------------------------------------------------------------------


def _capture_hip_toolchain(reasons: list[str]) -> dict[str, str | None]:
    """Run ``hipconfig --<flag>`` for each toolchain field.

    Issued as separate invocations because hipconfig prints results
    concatenated when multiple flags are passed, with no delimiter -- a
    single ``hipconfig --version --platform`` produces ``"7.2.5amd"``,
    which is unparseable. Five short subprocesses still finish in <100 ms.
    """
    if shutil.which("hipconfig") is None:
        msg = "hip: hipconfig not on PATH; all hip.* fields = null"
        log.info(msg)
        reasons.append(msg)
        return {
            "version": None,
            "platform": None,
            "compiler": None,
            "runtime": None,
            "cpp_config": None,
        }

    block = {
        "version": _hipconfig("--version"),
        "platform": _hipconfig("--platform"),
        "compiler": _hipconfig("--compiler"),
        "runtime": _hipconfig("--runtime"),
        "cpp_config": _hipconfig("--cpp_config"),
    }
    for key, value in block.items():
        if value is None:
            reasons.append(f"hip.{key}: hipconfig --{key} returned no usable value")
    return block


def _hipconfig(flag: str) -> str | None:
    """Run ``hipconfig <flag>`` and return stripped stdout (or None)."""
    try:
        result = subprocess.run(
            ["hipconfig", flag],
            capture_output=True,
            text=True,
            timeout=SHORT_TIMEOUT_SEC,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0:
        return None
    out = (result.stdout or "").strip()
    return out or None


# ---------------------------------------------------------------------------
# hipBLASLt introspection
# ---------------------------------------------------------------------------


# The version-tweak field in hipblaslt-version.h is the canonical source
# for the build's git SHA (typically a 7-12 char short hash).
_HIPBLASLT_TWEAK_RE = re.compile(
    r"#define\s+HIPBLASLT_VERSION_TWEAK\s+([A-Za-z0-9_.+-]+)"
)
_HIPBLASLT_VERSION_RE = re.compile(
    r"#define\s+HIPBLASLT_VERSION_(MAJOR|MINOR|PATCH)\s+(\d+)"
)

# rocBLAS uses the same TWEAK/MAJOR/MINOR/PATCH layout in its header.
_ROCBLAS_TWEAK_RE = re.compile(
    r"#define\s+ROCBLAS_VERSION_TWEAK\s+([A-Za-z0-9_.+-]+)"
)
_ROCBLAS_VERSION_RE = re.compile(
    r"#define\s+ROCBLAS_VERSION_(MAJOR|MINOR|PATCH)\s+(\d+)"
)

# CK uses a different shape: full 40-char SHA in CK_COMMIT_ID (not a
# truncated TWEAK), and three MAJOR/MINOR/PATCH defines just like the
# others. Allow 7-40 hex chars so dev builds with abbreviated SHAs still
# parse.
_CK_COMMIT_RE = re.compile(r"#define\s+CK_COMMIT_ID\s+([A-Fa-f0-9]{7,40})")
_CK_VERSION_RE = re.compile(r"#define\s+CK_VERSION_(MAJOR|MINOR|PATCH)\s+(\d+)")

# MIOpen header has the same MAJOR/MINOR/PATCH/TWEAK shape as hipBLASLt.
_MIOPEN_TWEAK_RE = re.compile(
    r"#define\s+MIOPEN_VERSION_TWEAK\s+([A-Za-z0-9_.+-]+)"
)
_MIOPEN_VERSION_RE = re.compile(
    r"#define\s+MIOPEN_VERSION_(MAJOR|MINOR|PATCH)\s+(\d+)"
)

# RCCL/NCCL pack the version into a single int macro. Capture both the
# raw code and a decoded MAJOR.MINOR.PATCH string.
_NCCL_VERSION_CODE_RE = re.compile(r"#define\s+NCCL_VERSION_CODE\s+(\d+)")

# `-DUSE_FBGEMM` and `-DUSE_FBGEMM_GENAI` mean different things, so the
# plain-FBGEMM check needs a trailing word boundary (otherwise it would
# false-positive on `-DUSE_FBGEMM_GENAI`). The GENAI one is unique enough.
_FBGEMM_DEFINE_RE = re.compile(r"-DUSE_FBGEMM(?![A-Za-z0-9_])")
_FBGEMM_GENAI_DEFINE_RE = re.compile(r"-DUSE_FBGEMM_GENAI(?![A-Za-z0-9_])")

# CK SDPA / GEMM are build-time cmake flags (consumed when the PyTorch
# wheel is built, NOT runtime env vars). Detect from the same
# torch.__config__.show() text the FBGEMM probe scans.
_CK_SDPA_DEFINE_RE = re.compile(r"-DUSE_ROCM_CK_SDPA(?![A-Za-z0-9_])")
_CK_GEMM_DEFINE_RE = re.compile(r"-DUSE_ROCM_CK_GEMM(?![A-Za-z0-9_])")

# Match libaotriton_v2.so.<MAJOR>.<MINOR>.<PATCH>. Anchored so a stray
# debug-suffixed variant (e.g. .so.0.11.1.dbg) would NOT match -- we
# want a clean version cell.
_AOTRITON_VERSION_RE = re.compile(r"libaotriton_v2\.so\.(\d+\.\d+\.\d+)$")


def _capture_hipblaslt(reasons: list[str]) -> dict[str, Any]:
    """Capture hipBLASLt build identity.

    Goal: catch GEMM kernel library drift across docker images / conda
    envs / venvs. See issue #147 motivation.

    NOTE on ``rocm_release_tweak`` vs ``commit``: AMD sets
    ``HIPBLASLT_VERSION_TWEAK`` to the **ROCm release identifier**, not
    to hipBLASLt's upstream git SHA. In a given ROCm release every
    library shares the same TWEAK -- so this field is the right name
    for "which ROCm release built this library", not "which upstream
    hipBLASLt commit". For per-library upstream SHA tracking, see
    ``lib_hash`` (changes any time the binary changes) and the
    ``applied_prs`` slot (filled in when a specific PR detector lands).

    The ``applied_prs`` block is intentionally empty in this first cut.
    Adding ``pr_<id>_applied`` keys later is additive and does not bump
    ``schema_version``. Each PR detector needs a unique signature
    (symbol via ``nm``, string via ``strings``, or Tensile YAML revision
    bump) -- those will land in a follow-up alongside the first PR we
    care to track.
    """
    header_text = _read_text_file(HIPBLASLT_VERSION_HEADER)
    rocm_release_tweak, package_version = _parse_hipblaslt_header(header_text)
    lib_hash = _hash_hipblaslt_library()
    # Renamed from `tensile_yaml_revision` in schema 1.1: modern
    # hipBLASLt ships .dat (binary), not .yaml; the field shape is the
    # same kernel-DB filename fingerprint as miopen.kernel_db_revision.
    kernel_db_revision = _tensile_fingerprint()

    block: dict[str, Any] = {
        "rocm_release_tweak": rocm_release_tweak,
        "package_version": package_version,
        "lib_hash": lib_hash,
        "kernel_db_revision": kernel_db_revision,
        "applied_prs": {},
    }
    # Distinguish "header file unreadable" from "header readable but the
    # specific define is missing/unparseable" so partial_reasons points
    # callers at the right thing to investigate.
    header_unreadable = header_text is None
    if rocm_release_tweak is None:
        if header_unreadable:
            reasons.append(
                f"hipblaslt.rocm_release_tweak: {HIPBLASLT_VERSION_HEADER} not readable"
            )
        else:
            reasons.append(
                f"hipblaslt.rocm_release_tweak: {HIPBLASLT_VERSION_HEADER} did not "
                "contain a readable HIPBLASLT_VERSION_TWEAK define"
            )
    if package_version is None:
        if header_unreadable:
            reasons.append(
                f"hipblaslt.package_version: {HIPBLASLT_VERSION_HEADER} not readable"
            )
        else:
            reasons.append(
                f"hipblaslt.package_version: {HIPBLASLT_VERSION_HEADER} did not "
                "contain MAJOR/MINOR/PATCH defines"
            )
    if lib_hash is None:
        reasons.append(
            f"hipblaslt.lib_hash: {HIPBLASLT_LIB_DIR}/libhipblaslt.so "
            "missing or unreadable"
        )
    if kernel_db_revision is None:
        # _tensile_fingerprint returns None for both "directory missing /
        # unlistable" AND "directory present but no kernel files" --
        # the wording covers both so partial_reasons is honest.
        reasons.append(
            "hipblaslt.kernel_db_revision: directory missing/unreadable "
            f"or no kernel files under {HIPBLASLT_TENSILE_DIR}"
        )
    return block


def _parse_version_header(
    text: str | None,
    tweak_re: re.Pattern[str],
    version_re: re.Pattern[str],
) -> tuple[str | None, str | None]:
    """Extract (commit, MAJOR.MINOR.PATCH) from a ROCm version header.

    Generalised over the regex pair so the same logic serves hipblaslt,
    rocblas, and CK (which use TWEAK / VERSION_TWEAK / COMMIT_ID with
    slightly different naming but the same layout).

    Returns ``(commit, package_version)``, each ``None`` if the header
    was missing or did not contain the expected defines.
    """
    if not text:
        return (None, None)
    tweak_match = tweak_re.search(text)
    commit = tweak_match.group(1) if tweak_match else None
    parts: dict[str, str] = {}
    for match in version_re.finditer(text):
        parts[match.group(1)] = match.group(2)
    if {"MAJOR", "MINOR", "PATCH"}.issubset(parts):
        package_version = f"{parts['MAJOR']}.{parts['MINOR']}.{parts['PATCH']}"
    else:
        package_version = None
    return (commit, package_version)


def _parse_hipblaslt_header(text: str | None) -> tuple[str | None, str | None]:
    """Backward-compat wrapper. Kept so existing tests keep their call shape."""
    return _parse_version_header(text, _HIPBLASLT_TWEAK_RE, _HIPBLASLT_VERSION_RE)


def _hash_file_path(path: Path) -> str | None:
    """SHA-256 a specific file path, resolving symlinks first.

    Distinct from ``_hash_shared_library`` -- this one hashes a
    **caller-supplied** Path. Use it when the caller has already done
    a smarter selection than the glob+string-sort fallback in
    ``_hash_shared_library`` (e.g. AOTriton's version-tuple sort that
    correctly orders ``0.10.0`` after ``0.9.0``). Returns
    ``"sha256:<hex>"`` or ``None``.
    """
    try:
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, OSError):
        return None
    if not resolved.is_file():
        return None
    try:
        digest = hashlib.sha256()
        with resolved.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
        return f"sha256:{digest.hexdigest()}"
    except OSError as exc:
        log.debug("hash failed for %s: %s", resolved, exc)
        return None


def _parse_soname_version(filename: str, soname: str) -> tuple[int, ...] | None:
    """Parse the dotted-decimal suffix of a versioned soname.

    ``("libfoo.so.1.2.70201", "libfoo.so") -> (1, 2, 70201)``. Returns
    ``None`` when the suffix isn't pure dotted-decimal (e.g. a debug
    build with a trailing tag) so callers can fall back to lex sort.

    Used to sort versioned-soname siblings by integer-tuple instead of
    lexicographically -- otherwise ``libfoo.so.5.10.0`` ranks below
    ``libfoo.so.5.9.0`` (because ``"1" < "9"`` as strings) and a
    multi-version install would record the wrong file's hash.
    """
    prefix = soname + "."
    if not filename.startswith(prefix):
        return None
    suffix = filename[len(prefix):]
    if not suffix:
        return None
    try:
        return tuple(int(p) for p in suffix.split("."))
    except ValueError:
        return None


def _hash_shared_library(lib_dir: Path, soname: str) -> str | None:
    """SHA-256 a shared library, resolving symlinks first.

    Tries the unversioned ``soname`` (e.g. ``libfoo.so``) first; if
    absent, falls back to the highest-versioned matching filename
    (``libfoo.so.1.2.70201`` etc.). The unversioned ``.so`` symlink is
    typically shipped only with ``-dev`` packages (it's a build-time
    link target); stripped runtime-only images keep just the versioned
    files. Without this fallback, the probe would record
    ``lib_hash=None`` plus a misleading "missing or unreadable" reason
    on a host where the library is fully present and being used.

    Resolves through symlinks so e.g. ``libfoo.so`` -> ``libfoo.so.1`` ->
    ``libfoo.so.1.2.3`` collapses to one hash regardless of which name
    the consumer linked against. Returns ``"sha256:<hex>"`` or ``None``.
    """
    # Build candidate list: unversioned first, then versioned files
    # ranked highest-first by integer-tuple (NOT lexicographic) so a
    # mid-upgrade pair like ``.so.5.10.0`` vs ``.so.5.9.0`` resolves to
    # the actually-newer file. Any sibling whose suffix isn't pure
    # dotted-decimal falls into a second tier and is lex-sorted -- it's
    # an oddly-named file and we still want a deterministic pick.
    candidates: list[Path] = [lib_dir / soname]
    try:
        siblings = list(lib_dir.glob(f"{soname}.*"))
    except OSError as exc:
        log.debug("glob failed for %s/%s.*: %s", lib_dir, soname, exc)
        siblings = []
    parsed: list[tuple[tuple[int, ...], Path]] = []
    unparsed: list[Path] = []
    for path in siblings:
        version = _parse_soname_version(path.name, soname)
        if version is None:
            unparsed.append(path)
        else:
            parsed.append((version, path))
    parsed.sort(key=lambda x: x[0], reverse=True)
    unparsed.sort(reverse=True)
    candidates.extend(p for _, p in parsed)
    candidates.extend(unparsed)

    seen_resolved: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=True)
        except (FileNotFoundError, OSError):
            continue
        # Symlink chains may collapse to the same file; only hash once.
        if resolved in seen_resolved:
            continue
        seen_resolved.add(resolved)
        if not resolved.is_file():
            continue
        try:
            digest = hashlib.sha256()
            with resolved.open("rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    digest.update(chunk)
            return f"sha256:{digest.hexdigest()}"
        except OSError as exc:
            log.debug("hash failed for %s: %s", resolved, exc)
            continue
    return None


def _hash_hipblaslt_library() -> str | None:
    """Backward-compat wrapper. Kept so existing tests keep their call shape."""
    return _hash_shared_library(HIPBLASLT_LIB_DIR, "libhipblaslt.so")


def _kernel_db_filename_fingerprint(
    directory: Path,
    suffixes: tuple[str, ...] = (".yaml", ".dat", ".co"),
) -> str | None:
    """Fingerprint a Tensile-style kernel database by sorted filenames.

    Modern hipBLASLt / rocBLAS ship ``.dat`` files (binary), older builds
    shipped ``.yaml``; some toolchains also drop ``.co`` (code objects).
    We hash the **sorted filenames** -- a fast, deterministic fingerprint
    that changes whenever the kernel set changes (new gfx target, new
    operation layout, removed kernel). Hashing the contents would be GB
    of work and add seconds; the filename set already tracks the
    meaningful drift.

    Assumes a **flat layout** -- only the top-level directory is scanned
    (no recursion). Verified flat on ROCm 5.x / 6.x / 7.x for both
    ``/opt/rocm/lib/hipblaslt/library/`` and
    ``/opt/rocm/lib/rocblas/library/``. If a future release switches to
    per-gfx-target subdirectories, swap ``iterdir()`` for ``rglob("*")``
    -- but doing so unconditionally would also pull in unrelated cache
    files some packagers drop alongside the kernel DB.
    """
    if not directory.is_dir():
        return None
    try:
        names = sorted(
            p.name
            for p in directory.iterdir()
            if p.is_file() and p.suffix in suffixes
        )
    except OSError as exc:
        log.debug("kernel-db dir listing failed for %s: %s", directory, exc)
        return None
    if not names:
        return None
    digest = hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()
    return f"filenames-sha256:{digest}"


def _tensile_fingerprint() -> str | None:
    """Backward-compat wrapper. Kept so existing tests keep their call shape."""
    return _kernel_db_filename_fingerprint(HIPBLASLT_TENSILE_DIR)


# ---------------------------------------------------------------------------
# Runtime context: container + Python env detection
# ---------------------------------------------------------------------------


def _detect_runtime_context() -> dict[str, str | None]:
    """Detect container runtime + Python environment.

    The schema's allowed values for ``runtime_context.type`` are
    ``docker | podman | singularity | baremetal`` (per #147). To keep
    strict consumers safe, this function only ever returns one of those
    four; runtimes outside the documented set (e.g. containerd-managed
    Kubernetes pods) currently fall through to ``baremetal``. Adding new
    values is a schema change and would bump ``schema_version``.

    Container precedence (first match wins):
        1. ``/.dockerenv`` -> docker
        2. ``/run/.containerenv`` -> podman
        3. ``SINGULARITY_NAME`` env var or ``singularity`` in
           ``/proc/1/cgroup`` -> singularity
        4. ``docker`` / ``podman`` token in ``/proc/1/cgroup``
           -> matched runtime (cgroup fallback for stripped containers)
        5. otherwise -> baremetal

    Python env precedence:
        1. ``$CONDA_DEFAULT_ENV`` -> conda
        2. ``sys.prefix != sys.base_prefix`` -> venv
        3. otherwise -> system

    Never partial: every field is either populated or has a documented
    null reason (e.g. ``venv_path`` is null outside a venv -- that is the
    contract, not a fallback).
    """
    container_type = _detect_container_type()
    python_env = _detect_python_env()
    return {
        "type": container_type,
        "python_env": python_env,
        "venv_path": str(sys.prefix) if python_env == "venv" else None,
        "conda_env_name": (
            os.environ.get("CONDA_DEFAULT_ENV") if python_env == "conda" else None
        ),
    }


def _detect_container_type() -> str:
    """Resolve the container runtime; ``"baremetal"`` if none matched."""
    if DOCKERENV_MARKER.exists():
        return "docker"
    if PODMAN_CONTAINERENV_MARKER.exists():
        return "podman"
    if os.environ.get("SINGULARITY_NAME"):
        return "singularity"

    cgroup = _read_text_file(CGROUP_FILE)
    if cgroup:
        # cgroup lines look like '12:freezer:/docker/<id>' or
        # '0::/system.slice/docker-<id>.scope'. Apply the documented
        # precedence: singularity wins over docker/podman so a
        # Singularity environment that happens to inherit a docker-shaped
        # cgroup path is not misclassified. Limited to schema-documented
        # values.
        for runtime in ("singularity", "docker", "podman"):
            if runtime in cgroup:
                return runtime
    return "baremetal"


def _detect_python_env() -> str:
    """Return ``"conda"``, ``"venv"``, or ``"system"``."""
    if os.environ.get("CONDA_DEFAULT_ENV"):
        return "conda"
    # sys.base_prefix differs from sys.prefix inside a venv (PEP 405).
    if getattr(sys, "base_prefix", sys.prefix) != sys.prefix:
        return "venv"
    return "system"


# ---------------------------------------------------------------------------
# Docker metadata
# ---------------------------------------------------------------------------


def _capture_docker_metadata(
    runtime_context: dict[str, str | None],
    reasons: list[str],
) -> dict[str, str | None] | None:
    """Capture image + digest when running inside a container.

    Returns ``None`` for baremetal -- there is no image to record. This
    documented absence does NOT trigger ``partial=True``.

    For containerised runs we emit the block with best-effort values; the
    aorta-side launcher can populate them via the env vars below before
    invoking ``aorta env probe`` (which is the only reliable way to know
    image+digest from inside a container without privileged docker access):

    * ``AORTA_DOCKER_IMAGE``  -> ``docker.image``
    * ``AORTA_DOCKER_DIGEST`` -> ``docker.digest``

    Always also emits ``container_id`` parsed from ``/proc/self/cgroup``,
    which is recoverable from inside the container.

    When inside a container but the launcher did not set the env vars,
    appends a reason -- the snapshot can still be useful but cross-image
    comparison loses fidelity.
    """
    if runtime_context.get("type") == "baremetal":
        return None

    block = {
        "image": os.environ.get("AORTA_DOCKER_IMAGE"),
        "digest": os.environ.get("AORTA_DOCKER_DIGEST"),
        "container_id": _read_container_id(),
    }
    if block["image"] is None:
        reasons.append(
            "docker.image: AORTA_DOCKER_IMAGE env var not set by the launcher"
        )
    if block["digest"] is None:
        reasons.append(
            "docker.digest: AORTA_DOCKER_DIGEST env var not set by the launcher"
        )
    return block


_CONTAINER_ID_RE = re.compile(r"[0-9a-f]{12,64}")


def _read_container_id() -> str | None:
    """Pull the container ID out of ``SELF_CGROUP_FILE`` if present.

    Reads the *current* process's cgroup (not init's) -- the container
    ID lives there as a 12-64 hex segment (e.g. ``/docker/<id>/`` or
    ``/system.slice/docker-<id>.scope``).
    """
    text = _read_text_file(SELF_CGROUP_FILE)
    if not text:
        return None
    for line in text.splitlines():
        match = _CONTAINER_ID_RE.search(line)
        if match:
            return match.group(0)
    return None


# ---------------------------------------------------------------------------
# Env vars + Python/PyTorch
# ---------------------------------------------------------------------------


def _capture_env_vars() -> dict[str, str | None]:
    """Capture canonical env vars (explicit list, not prefix matching).

    Individual ``None`` values are the documented contract (env var unset)
    and DO NOT trigger ``partial=True``.
    """
    return {name: os.environ.get(name) for name in CANONICAL_ENV_VARS}


def _capture_python_package_version(
    package_name: str,
    reasons: list[str],
    *,
    reason_prefix: str | None = None,
    suppress_missing: bool = False,
) -> str | None:
    """Best-effort import of a Python package to read its ``__version__``.

    The same fail-soft contract as ``_capture_pytorch_version``: catches
    ImportError + any other unexpected import-time exception, never
    initialises GPU context (uses ``__import__`` directly so caller
    behaviour is identical to ``import package_name``), and never returns
    the string ``"None"``.

    Args:
        package_name: The module name to import (e.g. ``"triton"``).
        reasons: Shared partial-reasons list to append to on any fallback.
        reason_prefix: The prefix used when appending to ``reasons``.
            Defaults to ``f"{package_name}_version"`` (mirrors the existing
            ``pytorch_version: ...`` shape). Pass an explicit string when
            the snapshot field is named differently from the package
            (e.g. ``"fbgemm.package_version"``).
        suppress_missing: When True, don't append a partial reason for a
            plain ``ImportError`` -- the absence of this package is the
            documented common case (used for fbgemm_gpu / aiter, which
            most stock installs lack). Other failures (broken __version__,
            unexpected exception) still record a reason.
    """
    prefix = reason_prefix or f"{package_name}_version"
    # ``__import__`` returns the *top-level* module for dotted names
    # (so ``__import__("a.b.c")`` returns ``a``, not ``a.b.c``). Every
    # caller passes a top-level package name (torch, triton, fbgemm_gpu,
    # aiter, Tensile), so this is fine. Switch to ``importlib`` if you
    # ever need to probe a leaf module's ``__version__``.
    try:
        mod = __import__(package_name)
    except ImportError:
        if not suppress_missing:
            reasons.append(f"{prefix}: {package_name} not importable")
        return None
    except Exception as exc:  # noqa: BLE001 -- defensive; never let env probe fail
        log.debug("%s import for version probe failed: %s", package_name, exc)
        reasons.append(
            f"{prefix}: {package_name} import raised ({type(exc).__name__})"
        )
        return None

    version = getattr(mod, "__version__", None)
    if version is None:
        reasons.append(
            f"{prefix}: {package_name} lacks __version__ attribute"
        )
        return None
    return str(version)


def _safe_import_torch(reasons: list[str], probe_name: str) -> Any | None:
    """Try ``import torch``; return the module or ``None``.

    Centralised so every probe that needs torch (composable_kernel,
    aotriton, fbgemm flag scan, CK flag scan, pytorch_build) can use a
    single fail-soft import path. Three outcomes:

    * ``ImportError`` -> return ``None`` **silently**. The
      ``pytorch_version`` probe already records torch absence; every
      other probe shares that signal and shouldn't double-count.
    * Any other ``Exception`` (broken install, partial wheel,
      C-extension load failure) -> log + record one
      ``"<probe_name>: torch import raised (<exc-type>)"`` reason and
      return ``None``.
    * Success -> return the imported module.

    The probe_name string is the partial-reason prefix (e.g.
    ``"composable_kernel.pytorch_bundled"``,
    ``"aotriton"``). Use the same prefix the rest of that probe uses
    so partial_reasons stay grep-consistent.
    """
    try:
        return __import__("torch")
    except ImportError:
        return None
    except Exception as exc:  # noqa: BLE001 -- defensive; never let env probe fail
        log.debug("torch import for %s probe failed: %s", probe_name, exc)
        reasons.append(f"{probe_name}: torch import raised ({type(exc).__name__})")
        return None


def _capture_pytorch_version(reasons: list[str]) -> str | None:
    """Best-effort import of torch to read its version. No GPU work.

    ``import torch`` will ``dlopen`` HIP runtime libraries (so the
    process's loaded-libraries list grows), but it does NOT allocate
    device memory, launch kernels, or otherwise produce HIP API calls
    that ``rocprofv3 --hip-trace`` would record. The acceptance
    criterion "no GPU compute" -- meaning zero kernel dispatches and
    zero device allocations -- is preserved.

    Returns the version as a string when available, or ``None`` when torch
    is not installed OR is installed without a ``__version__`` attribute.
    Either fallback path appends a reason. Never returns the string
    ``"None"`` -- that would break consumers doing strict null checks
    against the JSON.
    """
    return _capture_python_package_version(
        "torch", reasons, reason_prefix="pytorch_version"
    )


# ---------------------------------------------------------------------------
# rocBLAS introspection -- mirrors the hipBLASLt block 1:1
# ---------------------------------------------------------------------------


def _capture_rocblas(reasons: list[str]) -> dict[str, Any]:
    """Capture rocBLAS build identity.

    Same shape and contract as ``_capture_hipblaslt`` -- two trials with
    different rocBLAS Tensile databases or different librocblas.so
    contents become trivially diffable. Reuses the generic
    ``_parse_version_header`` / ``_hash_shared_library`` /
    ``_kernel_db_filename_fingerprint`` helpers.

    See ``_capture_hipblaslt`` for the ``rocm_release_tweak`` vs
    upstream-commit distinction -- ``ROCBLAS_VERSION_TWEAK`` is the
    same ROCm-release-shared identifier, NOT a per-rocBLAS commit.

    The header lives at ``/opt/rocm/include/rocblas/internal/rocblas-version.h``
    (note the ``internal/`` subdir, unlike hipblaslt). It ships with
    ``rocblas-dev``; the runtime lib + Tensile DB ship with the runtime
    ``rocblas`` package and are usually present even in stripped images.

    ``applied_prs`` is intentionally empty in this first cut, mirroring
    the hipblaslt convention.
    """
    header_text = _read_text_file(ROCBLAS_VERSION_HEADER)
    rocm_release_tweak, package_version = _parse_version_header(
        header_text, _ROCBLAS_TWEAK_RE, _ROCBLAS_VERSION_RE
    )
    lib_hash = _hash_shared_library(ROCBLAS_LIB_DIR, "librocblas.so")
    # Renamed from `tensile_yaml_revision` in schema 1.1 -- see hipblaslt
    # block for the rationale.
    kernel_db_revision = _kernel_db_filename_fingerprint(ROCBLAS_TENSILE_DIR)

    block: dict[str, Any] = {
        "rocm_release_tweak": rocm_release_tweak,
        "package_version": package_version,
        "lib_hash": lib_hash,
        "kernel_db_revision": kernel_db_revision,
        "applied_prs": {},
    }
    header_unreadable = header_text is None
    if rocm_release_tweak is None:
        if header_unreadable:
            reasons.append(
                f"rocblas.rocm_release_tweak: {ROCBLAS_VERSION_HEADER} not readable"
            )
        else:
            reasons.append(
                f"rocblas.rocm_release_tweak: {ROCBLAS_VERSION_HEADER} did not "
                "contain a readable ROCBLAS_VERSION_TWEAK define"
            )
    if package_version is None:
        if header_unreadable:
            reasons.append(
                f"rocblas.package_version: {ROCBLAS_VERSION_HEADER} not readable"
            )
        else:
            reasons.append(
                f"rocblas.package_version: {ROCBLAS_VERSION_HEADER} did not "
                "contain MAJOR/MINOR/PATCH defines"
            )
    if lib_hash is None:
        reasons.append(
            f"rocblas.lib_hash: {ROCBLAS_LIB_DIR}/librocblas.so "
            "missing or unreadable"
        )
    if kernel_db_revision is None:
        reasons.append(
            "rocblas.kernel_db_revision: directory missing/unreadable "
            f"or no kernel files under {ROCBLAS_TENSILE_DIR}"
        )
    return block


# ---------------------------------------------------------------------------
# Composable Kernel (CK) -- two layers: system headers + PyTorch-bundled
# ---------------------------------------------------------------------------


def _capture_composable_kernel(reasons: list[str]) -> dict[str, Any]:
    """Capture Composable Kernel identity at both layers.

    CK ships in two places that can drift independently:

    * **system**: header-only install at ``/opt/rocm/include/ck/`` from
      the ``composablekernel-dev`` apt package. Other ROCm libs
      (rocBLAS, hipBLASLt, MIOpen) statically link CK kernels built
      against this version at *their* build time.
    * **pytorch_bundled**: vendored as a git submodule at
      ``third_party/composable_kernel/`` inside the PyTorch source tree;
      compiled into ``libtorch_hip.so`` at PyTorch wheel build time.
      Distinct from the system version -- often a different commit.

    Both are knowable from the installed environment without running any
    HIP code; both are recorded so cross-env diffs surface either drift.

    The two sub-blocks fail independently: a host with the
    ``composablekernel-dev`` package stripped will see ``system.*`` go
    null + reason while ``pytorch_bundled.*`` populates from the wheel
    on disk, and vice versa for a CPU-only PyTorch install with the
    system CK package present.
    """
    # ------- system sub-block -------
    header_text = _read_text_file(CK_VERSION_HEADER)
    sys_version, sys_commit = _parse_ck_header(header_text)
    ck_tile_present = CK_TILE_CONFIG_HEADER.exists()
    system_block: dict[str, Any] = {
        "version": sys_version,
        "commit": sys_commit,
        "ck_tile_present": ck_tile_present,
    }
    header_unreadable = header_text is None
    if sys_version is None:
        if header_unreadable:
            reasons.append(
                f"composable_kernel.system.version: {CK_VERSION_HEADER} not readable"
            )
        else:
            reasons.append(
                f"composable_kernel.system.version: {CK_VERSION_HEADER} did not "
                "contain MAJOR/MINOR/PATCH defines"
            )
    if sys_commit is None:
        if header_unreadable:
            reasons.append(
                f"composable_kernel.system.commit: {CK_VERSION_HEADER} not readable"
            )
        else:
            reasons.append(
                f"composable_kernel.system.commit: {CK_VERSION_HEADER} did not "
                "contain a readable CK_COMMIT_ID define"
            )

    # ------- pytorch_bundled sub-block -------
    bundled_block = _probe_pytorch_bundled_ck(reasons)

    # ------- build-time PyTorch CK dispatch flags -------
    # These are -DUSE_ROCM_CK_{SDPA,GEMM} cmake flags baked into the
    # wheel, NOT runtime env vars (a common misconception -- setting
    # USE_ROCM_CK_SDPA in the workload's env does nothing). Mirrors
    # the fbgemm.pytorch_use_fbgemm{,_genai} pattern.
    use_ck_sdpa, use_ck_gemm = _read_pytorch_ck_flags(reasons)

    return {
        "system": system_block,
        "pytorch_bundled": bundled_block,
        "pytorch_use_ck_sdpa": use_ck_sdpa,
        "pytorch_use_ck_gemm": use_ck_gemm,
    }


def _read_pytorch_ck_flags(reasons: list[str]) -> tuple[bool | None, bool | None]:
    """Parse ``torch.__config__.show()`` for the CK build-time flags.

    Returns ``(use_ck_sdpa, use_ck_gemm)``. Both ``None`` when torch is
    absent or ``__config__`` raises (rare). When torch is present, the
    booleans reflect whether ``-DUSE_ROCM_CK_SDPA`` / ``-DUSE_ROCM_CK_GEMM``
    appear in CXX_FLAGS. ``False`` is meaningful (a wheel deliberately
    built with the CK SDPA/GEMM path disabled, dispatching to AOTriton
    or non-CK rocBLAS instead); distinct from ``None`` (couldn't ask).
    """
    torch_mod = _safe_import_torch(reasons, "composable_kernel.pytorch_use_ck_sdpa")
    if torch_mod is None:
        return (None, None)
    config = getattr(torch_mod, "__config__", None)
    show = getattr(config, "show", None)
    if show is None:
        reasons.append(
            "composable_kernel.pytorch_use_ck_sdpa: torch.__config__.show unavailable"
        )
        return (None, None)
    try:
        config_text = show()
    except Exception as exc:  # noqa: BLE001
        log.debug("torch.__config__.show() raised: %s", exc)
        reasons.append(
            f"composable_kernel.pytorch_use_ck_sdpa: torch.__config__.show() raised "
            f"({type(exc).__name__})"
        )
        return (None, None)
    use_ck_sdpa = bool(_CK_SDPA_DEFINE_RE.search(config_text))
    use_ck_gemm = bool(_CK_GEMM_DEFINE_RE.search(config_text))
    return (use_ck_sdpa, use_ck_gemm)


def _parse_ck_header(text: str | None) -> tuple[str | None, str | None]:
    """Extract (version_str, commit) from CK's version.h.

    Note the ordering vs. ``_parse_version_header``: CK records the
    *version* as MAJOR.MINOR.PATCH and the *commit* as a separate full-SHA
    define, so the helper returns version first to match how the
    ``composable_kernel.system`` sub-block is shaped.
    """
    if not text:
        return (None, None)
    parts: dict[str, str] = {}
    for match in _CK_VERSION_RE.finditer(text):
        parts[match.group(1)] = match.group(2)
    if {"MAJOR", "MINOR", "PATCH"}.issubset(parts):
        version = f"{parts['MAJOR']}.{parts['MINOR']}.{parts['PATCH']}"
    else:
        version = None
    commit_match = _CK_COMMIT_RE.search(text)
    commit = commit_match.group(1) if commit_match else None
    return (version, commit)


def _probe_pytorch_bundled_ck(reasons: list[str]) -> dict[str, Any]:
    """Look for ``ck::`` symbols inside ``libtorch_hip.so``.

    Strategy: locate the lib via ``torch.__file__`` (no HIP context init),
    then run ``nm -D <lib>`` piped through ``c++filt`` and count
    occurrences of the ``ck::`` namespace prefix. Falls back to
    ``present=False, symbol_count=None`` plus a partial reason whenever
    any step is unavailable (torch broken, binutils stripped from the
    container, subprocess timeout).

    Documented absences that **do not** trigger ``partial=True``:

    * ``import torch`` raises ImportError (already captured by the
      ``pytorch_version`` probe -- no need to duplicate the reason).
    * ``torch.version.hip is None`` (this is a CPU-only PyTorch wheel,
      so ``libtorch_hip.so`` is legitimately absent by design).

    ``-D`` (dynamic symbols only) is much faster than the full symbol
    table on a multi-hundred-MB lib while still covering every CK kernel
    that's actually exposed for dispatch.
    """
    default = {"present": False, "symbol_count": None}

    # Locate libtorch_hip.so via torch.__file__. The helper shares the
    # same no-HIP-context guarantee as _capture_python_package_version
    # (only does __import__, never touches torch.cuda).
    torch_mod = _safe_import_torch(reasons, "composable_kernel.pytorch_bundled")
    if torch_mod is None:
        return default

    # CPU-only PyTorch wheel -- there is no HIP code to fingerprint.
    # ``torch.version.hip`` is the canonical "was this wheel built with
    # HIP?" signal (None when not). Treat as a documented absence so a
    # CPU-only wheel doesn't flip ``partial=True`` for the bundled-CK
    # probe -- but the consumer can still see it from the
    # ``pytorch_bundled.present=False`` field.
    torch_version = getattr(torch_mod, "version", None)
    if torch_version is not None and getattr(torch_version, "hip", None) is None:
        return default

    torch_file = getattr(torch_mod, "__file__", None)
    if not torch_file:
        reasons.append(
            "composable_kernel.pytorch_bundled: torch.__file__ unavailable"
        )
        return default

    lib_path = Path(torch_file).parent / "lib" / PYTORCH_HIP_LIB_NAME
    if not lib_path.exists():
        # torch.version.hip claimed HIP support but the lib is missing.
        # That's not a CPU-only wheel (we caught those above) -- it's a
        # broken / incomplete install worth flagging.
        reasons.append(
            f"composable_kernel.pytorch_bundled: {lib_path} not found "
            "(torch.version.hip claims HIP but the runtime lib is missing)"
        )
        return default

    nm = shutil.which("nm")
    cxxfilt = shutil.which("c++filt")
    if nm is None or cxxfilt is None:
        reasons.append(
            "composable_kernel.pytorch_bundled: nm/c++filt not on PATH "
            "(install binutils for symbol-count detection)"
        )
        return default

    try:
        nm_proc = subprocess.run(
            [nm, "-D", "--defined-only", str(lib_path)],
            capture_output=True,
            text=True,
            timeout=NM_TIMEOUT_SEC,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        reasons.append(
            f"composable_kernel.pytorch_bundled: nm invocation failed ({exc})"
        )
        return default
    if nm_proc.returncode != 0:
        reasons.append(
            f"composable_kernel.pytorch_bundled: nm exited "
            f"{nm_proc.returncode} on {lib_path}"
        )
        return default

    try:
        cxxfilt_proc = subprocess.run(
            [cxxfilt],
            input=nm_proc.stdout,
            capture_output=True,
            text=True,
            timeout=NM_TIMEOUT_SEC,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        reasons.append(
            f"composable_kernel.pytorch_bundled: c++filt invocation failed ({exc})"
        )
        return default
    if cxxfilt_proc.returncode != 0:
        reasons.append(
            f"composable_kernel.pytorch_bundled: c++filt exited "
            f"{cxxfilt_proc.returncode}"
        )
        return default

    # Count exported symbols in the `ck::` namespace. We use a substring
    # check (not a regex) for speed; demangled symbols routinely look
    # like "ck::tensor_operation::..." so any line containing "ck::" is a
    # CK symbol. This will also include symbols from other namespaces
    # whose template parameters contain "ck::", but those are themselves
    # CK-related by construction (they reference CK types), so they count.
    symbol_count = sum(
        1 for line in cxxfilt_proc.stdout.splitlines() if "ck::" in line
    )
    return {
        "present": symbol_count > 0,
        "symbol_count": symbol_count,
    }


# ---------------------------------------------------------------------------
# Tensile -- pip-probe + cross-library kernel-DB fingerprint
# ---------------------------------------------------------------------------


def _capture_tensile(reasons: list[str]) -> dict[str, Any]:
    """Capture Tensile build-time identity.

    Tensile is a build-time Python tool that generates GEMM kernels for
    rocBLAS and hipBLASLt; it is not a runtime artifact. Two surfaces:

    * ``package_version``: pip-probe ``Tensile`` if it happens to be
      installed (rare on production hosts; common on rocBLAS builders).
      Suppressed from partial_reasons because absence is the norm.
    * ``kernel_db_combined_hash``: a single fingerprint over the union
      of the hipBLASLt + rocBLAS kernel databases, so any drift in the
      Tensile *output* is captured even when Tensile itself isn't on
      disk. Marked partial only when *both* dirs are missing.
    """
    package_version = _capture_python_package_version(
        "Tensile",
        reasons,
        reason_prefix="tensile.package_version",
        suppress_missing=True,
    )
    combined_hash = _combined_kernel_db_fingerprint(
        [HIPBLASLT_TENSILE_DIR, ROCBLAS_TENSILE_DIR]
    )
    if combined_hash is None:
        reasons.append(
            "tensile.kernel_db_combined_hash: no kernel files under "
            f"{HIPBLASLT_TENSILE_DIR} or {ROCBLAS_TENSILE_DIR}"
        )
    return {
        "package_version": package_version,
        "kernel_db_combined_hash": combined_hash,
    }


def _combined_kernel_db_fingerprint(
    directories: list[Path],
    suffixes: tuple[str, ...] = (".yaml", ".dat", ".co"),
) -> str | None:
    """SHA-256 the sorted union of ``(library_name, filename)`` pairs.

    Tagging each filename with its **library's** name (the parent dir's
    basename, since each library puts its kernel DB under
    ``<library>/library/``) keeps the fingerprint distinguishable when
    the same kernel name appears in multiple libraries' databases.
    Using ``d.name`` directly would collapse to ``"library"`` for both
    hipBLASLt and rocBLAS -- ``d.parent.name`` is what makes this work.

    Returns ``None`` only when *every* input directory is missing or
    empty.
    """
    pairs: list[str] = []
    for d in directories:
        if not d.is_dir():
            continue
        # `d` is e.g. /opt/rocm/lib/hipblaslt/library; the meaningful
        # tag is the library name one level up.
        tag = d.parent.name or d.name
        try:
            for p in d.iterdir():
                if p.is_file() and p.suffix in suffixes:
                    pairs.append(f"{tag}/{p.name}")
        except OSError as exc:
            log.debug("combined kernel-db listing failed for %s: %s", d, exc)
            continue
    if not pairs:
        return None
    pairs.sort()
    digest = hashlib.sha256("\n".join(pairs).encode("utf-8")).hexdigest()
    return f"filenames-sha256:{digest}"


# ---------------------------------------------------------------------------
# Triton / FBGEMM / AITER -- pure Python-package version probes
# ---------------------------------------------------------------------------


def _capture_triton(reasons: list[str]) -> dict[str, str | None]:
    """Capture Triton package version.

    The ROCm Triton fork puts the source commit into ``__version__``
    (e.g. ``"3.5.1+rocm7.2.1.gita272dfa8"``), so this single field
    captures both the upstream version and the fork commit. No header
    parsing needed.
    """
    return {
        "package_version": _capture_python_package_version(
            "triton", reasons, reason_prefix="triton.package_version"
        ),
    }


def _capture_fbgemm(reasons: list[str]) -> dict[str, Any]:
    """Capture FBGEMM identity.

    FBGEMM is usually vendored inside the PyTorch wheel rather than
    installed as a separate ``fbgemm_gpu`` pip package. So we capture
    both surfaces:

    * ``package_version``: pip-probe ``fbgemm_gpu``; absence is
      suppressed from partial_reasons (the common case for stock
      ``pip install torch`` is no separate fbgemm_gpu).
    * ``pytorch_use_fbgemm`` / ``pytorch_use_fbgemm_genai``: parsed
      from ``torch.__config__.show()`` -- whether the PyTorch wheel was
      built with ``-DUSE_FBGEMM`` / ``-DUSE_FBGEMM_GENAI``. These reflect
      the actual code paths compiled into ``libtorch_cpu.so`` /
      ``libtorch_hip.so``, regardless of whether fbgemm_gpu is
      separately installed.
    """
    package_version = _capture_python_package_version(
        "fbgemm_gpu",
        reasons,
        reason_prefix="fbgemm.package_version",
        suppress_missing=True,
    )
    use_fbgemm, use_fbgemm_genai = _read_pytorch_fbgemm_flags(reasons)
    return {
        "package_version": package_version,
        "pytorch_use_fbgemm": use_fbgemm,
        "pytorch_use_fbgemm_genai": use_fbgemm_genai,
    }


def _read_pytorch_fbgemm_flags(
    reasons: list[str],
) -> tuple[bool | None, bool | None]:
    """Parse ``torch.__config__.show()`` for the FBGEMM compile-time flags.

    Returns ``(use_fbgemm, use_fbgemm_genai)``. Both are ``None`` when
    torch is absent or ``__config__`` raises (rare). When torch is
    present, the booleans reflect whether ``-DUSE_FBGEMM`` /
    ``-DUSE_FBGEMM_GENAI`` appear in the build's CXX_FLAGS. ``False``
    is a meaningful answer (a ROCm wheel deliberately built without
    FBGEMM-GENAI), distinct from ``None`` (couldn't ask).
    """
    torch_mod = _safe_import_torch(reasons, "fbgemm.pytorch_use_fbgemm")
    if torch_mod is None:
        return (None, None)
    config = getattr(torch_mod, "__config__", None)
    show = getattr(config, "show", None)
    if show is None:
        reasons.append(
            "fbgemm.pytorch_use_fbgemm: torch.__config__.show unavailable"
        )
        return (None, None)
    try:
        config_text = show()
    except Exception as exc:  # noqa: BLE001
        log.debug("torch.__config__.show() raised: %s", exc)
        reasons.append(
            f"fbgemm.pytorch_use_fbgemm: torch.__config__.show() raised "
            f"({type(exc).__name__})"
        )
        return (None, None)
    use_fbgemm = bool(_FBGEMM_DEFINE_RE.search(config_text))
    use_fbgemm_genai = bool(_FBGEMM_GENAI_DEFINE_RE.search(config_text))
    return (use_fbgemm, use_fbgemm_genai)


def _capture_aiter(reasons: list[str]) -> dict[str, str | None]:
    """Capture AITER (AMD Iterative kernel library) version.

    AITER is a PyPI-distributed ROCm inference kernel library built on
    top of CK. Most environments don't have it installed; absence is
    suppressed from partial_reasons.

    Note: upstream PyTorch now vendors AITER as a third_party/ submodule
    too -- see :func:`_capture_pytorch_build` for the bundled-commit
    probe. This block only covers the standalone pip distribution.
    """
    return {
        "package_version": _capture_python_package_version(
            "aiter",
            reasons,
            reason_prefix="aiter.package_version",
            suppress_missing=True,
        ),
    }


# ---------------------------------------------------------------------------
# PyTorch build identity -- structured complement to ``pytorch_version``
# ---------------------------------------------------------------------------


def _capture_pytorch_build(reasons: list[str]) -> dict[str, Any]:
    """Capture structured PyTorch build identity.

    Complements the flat ``pytorch_version`` field with the build
    metadata that ``torch.version`` exposes (always available on a
    PyTorch install) plus per-submodule SHAs from the source tree (when
    available). Reproducibility-critical for two reasons:

    * ``git_commit`` is the linchpin: it deterministically pins every
      vendored submodule. An operator with only ``env.json`` can resolve
      ``third_party/composable_kernel``, ``third_party/aiter``, and
      ``third_party/fbgemm`` to specific commits via the GitHub tree at
      that SHA -- no source tree required.
    * ``submodule_commits.*`` give the answer directly when a source
      tree is available (set ``AORTA_PYTORCH_SRC=/path/to/pytorch``, or
      auto-detected for editable / source installs).

    No GPU work. ``import torch`` populates Python objects; this probe
    only reads attributes off the imported module and runs ``git``
    subprocesses against a filesystem path.
    """
    install_kind, source_path = _detect_pytorch_install_kind()

    # torch.version.* fields. Defaults if torch is absent or the
    # `version` attribute is missing.
    git_commit: str | None = None
    hip_version: str | None = None
    cuda_version: str | None = None
    debug: bool | None = None

    # pytorch_version probe already records ImportError; helper stays
    # silent on that path. Records a reason only for unexpected
    # import-time exceptions.
    torch_mod = _safe_import_torch(reasons, "pytorch_build")
    if torch_mod is not None:
        version = getattr(torch_mod, "version", None)
        if version is None:
            reasons.append(
                "pytorch_build: torch.version unavailable (unexpectedly old build?)"
            )
        else:
            git_commit = getattr(version, "git_version", None) or None
            hip_version = getattr(version, "hip", None) or None
            cuda_version = getattr(version, "cuda", None) or None
            debug = getattr(version, "debug", None)
            if git_commit is None:
                reasons.append(
                    "pytorch_build.git_commit: torch.version.git_version is null"
                )

    submodule_commits = _capture_pytorch_submodules(
        install_kind, source_path, git_commit, reasons
    )

    return {
        "git_commit": git_commit,
        "hip_version": hip_version,
        "cuda_version": cuda_version,
        "debug": debug,
        "install_kind": install_kind,
        "source_path": str(source_path) if source_path else None,
        "submodule_commits": submodule_commits,
    }


def _detect_pytorch_install_kind() -> tuple[str, Path | None]:
    """Determine how PyTorch is installed.

    Returns ``(install_kind, source_path)``:

    * ``"source"`` -- explicit ``$AORTA_PYTORCH_SRC`` points at a tree
      with ``third_party/``; OR walking up from ``torch.__file__`` finds
      a ``.git`` + ``third_party/`` combo (common for `python -c "import
      torch"` from inside a checkout where the local torch dir shadows
      the installed wheel).
    * ``"editable"`` -- the install metadata's ``direct_url.json`` (PEP
      660) marks the install as editable AND the URL points at a
      directory with ``third_party/``.
    * ``"wheel"`` -- default. The ``third_party/`` tree is not on disk;
      submodule SHAs are recoverable only via the wheel's
      ``git_commit`` + GitHub lookup.
    * ``"unknown"`` -- torch import failed; we can't introspect.

    Cheap and side-effect-free: zero subprocesses, only stat() + a
    single small JSON read for the editable-install marker.
    """
    src_env = os.environ.get(AORTA_PYTORCH_SRC_ENV)
    if src_env:
        candidate = Path(src_env).expanduser()
        if (candidate / "third_party").is_dir():
            return ("source", candidate.resolve())
        # Honour the env-var intent but fall through if the directory
        # doesn't actually have third_party/. The reasons list will
        # catch it from _capture_pytorch_submodules.

    try:
        torch_mod = __import__("torch")
    except Exception:  # noqa: BLE001 -- ImportError + any unexpected failure both yield "unknown"
        return ("unknown", None)

    torch_file = getattr(torch_mod, "__file__", None)
    if not torch_file:
        return ("unknown", None)
    torch_dir = Path(torch_file).parent

    # Editable install marker (PEP 660). The .dist-info layout puts a
    # `direct_url.json` next to METADATA when `pip install -e` was used.
    torch_version = getattr(torch_mod, "__version__", "") or ""
    if torch_version:
        direct_url = (
            torch_dir.parent / f"torch-{torch_version}.dist-info" / "direct_url.json"
        )
        if direct_url.exists():
            try:
                data = json.loads(direct_url.read_text(encoding="utf-8"))
                if data.get("dir_info", {}).get("editable") and "url" in data:
                    src = Path(data["url"].removeprefix("file://"))
                    if (src / "third_party").is_dir():
                        return ("editable", src.resolve())
            except (OSError, json.JSONDecodeError) as exc:
                log.debug("editable-install detection: %s", exc)

    # Walk up from torch.__file__ looking for a .git + third_party
    # combo (catches imports from inside a source checkout).
    for parent in (torch_dir.parent, torch_dir.parent.parent):
        if (parent / ".git").exists() and (parent / "third_party").is_dir():
            return ("source", parent.resolve())

    return ("wheel", None)


def _capture_pytorch_submodules(
    install_kind: str,
    source_path: Path | None,
    git_commit: str | None,
    reasons: list[str],
) -> dict[str, str | None | dict | None]:
    """Resolve third_party submodule SHAs.

    Layered:

    * source / editable: ``git -C <src>/third_party/<name> rev-parse
      HEAD`` per submodule. ``_source = "git"`` records the provenance.
    * wheel / unknown: every submodule is ``None`` and a single
      partial_reasons line tells the operator how to look the SHAs up
      via the GitHub tree at the captured ``git_commit``. The reason
      uses the literal URL template so the operator never has to leave
      the env.json to find the recovery path.
    """
    result: dict[str, Any] = {name: None for name in CANONICAL_PYTORCH_SUBMODULES}
    result["_source"] = None

    if install_kind in ("source", "editable") and source_path is not None:
        third_party = source_path / "third_party"
        if not third_party.is_dir():
            reasons.append(
                f"pytorch_build.submodule_commits: {third_party} missing "
                f"on detected {install_kind} tree at {source_path}"
            )
            return result

        ok_count = 0
        missing: list[str] = []
        for name in CANONICAL_PYTORCH_SUBMODULES:
            sub_dir = third_party / name
            if not sub_dir.exists():
                # Submodule pin may post-date this commit; record but
                # don't make it noisy.
                missing.append(name)
                continue
            sha = _git_rev_parse_head(sub_dir)
            if sha:
                result[name] = sha
                ok_count += 1
            else:
                missing.append(name)
        if ok_count > 0:
            result["_source"] = "git"
        if missing:
            reasons.append(
                "pytorch_build.submodule_commits: source tree at "
                f"{source_path} has no readable git checkout for: "
                + ", ".join(missing)
            )
        return result

    # Wheel / unknown: print the recovery hint with the captured commit
    # substituted in (when known). Operators reading partial_reasons
    # get a copy-pasteable URL.
    if install_kind == "wheel":
        commit = git_commit or "<git_commit>"
        url_template = _PYTORCH_SUBMODULE_LOOKUP_HINT.replace(
            "<git_commit>", commit
        )
        reasons.append(
            "pytorch_build.submodule_commits: wheel install -- direct "
            "SHAs not recoverable; resolve via "
            f"{url_template} (set {AORTA_PYTORCH_SRC_ENV}=<src> to enable "
            "in-process probing)"
        )
    elif install_kind == "unknown":
        reasons.append(
            "pytorch_build.submodule_commits: torch import failed -- "
            "submodule SHAs unrecoverable"
        )
    return result


def _capture_aotriton(reasons: list[str]) -> dict[str, Any]:
    """Capture AOTriton identity (default ROCm Flash Attention backend).

    AOTriton is fetched at PyTorch build time via
    ``cmake/External/aotriton.cmake`` and bundled into the wheel as
    ``<torch>/lib/libaotriton_v2.so.MAJOR.MINOR.PATCH`` plus an
    ``aotriton.images/`` directory of pre-compiled kernel images.
    Operators can override the bundled copy by setting
    ``AOTRITON_INSTALLED_PREFIX`` to a system install root.

    Captured fields (all always present; absent values become null +
    partial reason where partial):

    * ``bundled_present`` -- ``libaotriton_v2.so*`` exists in the
      torch wheel's lib dir.
    * ``bundled_version`` -- parsed from the filename
      (``libaotriton_v2.so.0.11.1`` -> ``"0.11.1"``).
    * ``bundled_lib_hash`` -- sha256 of the resolved file (changes
      whenever AOTriton is rebuilt, even at the same version string).
    * ``bundled_images_dir_present`` -- whether
      ``<torch>/lib/aotriton.images/`` is shipped (it always should be;
      absence indicates a non-default packaging).
    * ``installed_prefix`` -- value of ``$AOTRITON_INSTALLED_PREFIX``,
      or ``null`` when unset (the common case -- bundled wins).
    """
    default = {
        "bundled_present": False,
        "bundled_version": None,
        "bundled_lib_hash": None,
        "bundled_images_dir_present": False,
        "installed_prefix": os.environ.get(AOTRITON_INSTALLED_PREFIX_ENV),
    }

    # AOTriton has no presence without torch loaded; documented absence
    # -- the helper silently returns None on ImportError (the
    # pytorch_version probe already records that), only records a
    # reason for unexpected import-time exceptions.
    torch_mod = _safe_import_torch(reasons, "aotriton")
    if torch_mod is None:
        return default

    # CPU-only torch wheel won't ship AOTriton. Treat the same way as
    # the bundled-CK probe: torch.version.hip is None -> skip silently.
    torch_version = getattr(torch_mod, "version", None)
    if torch_version is not None and getattr(torch_version, "hip", None) is None:
        return default

    torch_file = getattr(torch_mod, "__file__", None)
    if not torch_file:
        reasons.append("aotriton: torch.__file__ unavailable")
        return default

    lib_dir = Path(torch_file).parent / "lib"

    # Find every libaotriton_v2.so* file. Pick the best-versioned for
    # the version+hash; record presence/absence of the images dir.
    try:
        candidates = sorted(lib_dir.glob(f"{AOTRITON_LIB_PREFIX}*"))
    except OSError as exc:
        log.debug("aotriton glob failed in %s: %s", lib_dir, exc)
        reasons.append(f"aotriton: failed to scan {lib_dir} ({exc})")
        return default

    images_dir = lib_dir / AOTRITON_IMAGES_DIR_NAME
    images_present = images_dir.is_dir()

    if not candidates:
        # libtorch_hip.so was present (the CK probe verified that) but
        # AOTriton isn't bundled -- unusual but possible (custom build,
        # AOTriton disabled). Worth flagging but not catastrophic.
        reasons.append(
            f"aotriton.bundled: no {AOTRITON_LIB_PREFIX}* in {lib_dir} "
            "(custom PyTorch build with AOTriton disabled?)"
        )
        block = dict(default)
        block["bundled_images_dir_present"] = images_present
        return block

    # Pick the highest-versioned file by parsing the suffix (so
    # `libaotriton_v2.so.0.11.1` wins over `libaotriton_v2.so.0.10.0`
    # if both exist, regardless of mtime).
    versioned: list[tuple[tuple[int, int, int], Path]] = []
    for cand in candidates:
        m = _AOTRITON_VERSION_RE.search(cand.name)
        if m:
            try:
                parts = tuple(int(p) for p in m.group(1).split("."))
                if len(parts) == 3:
                    versioned.append((parts, cand))  # type: ignore[arg-type]
            except ValueError:
                continue

    if versioned:
        versioned.sort(key=lambda x: x[0], reverse=True)
        best_path = versioned[0][1]
        bundled_version = ".".join(str(p) for p in versioned[0][0])
    else:
        # Only an unversioned `libaotriton_v2.so` symlink exists; we
        # can still hash it but won't get a version string.
        best_path = candidates[0]
        bundled_version = None
        reasons.append(
            f"aotriton.bundled_version: no versioned filename in {lib_dir} "
            "matching libaotriton_v2.so.MAJOR.MINOR.PATCH"
        )

    # Hash the **specific** ``best_path`` we just chose by version-tuple
    # sort. Don't fall back to ``_hash_shared_library``'s string-sort
    # glob -- that picks the wrong file for any pair like
    # ``libaotriton_v2.so.0.10.0`` vs ``libaotriton_v2.so.0.9.0``
    # (lexically "0.9.0" > "0.10.0" because '9' > '1'), which would
    # leave bundled_version and bundled_lib_hash describing different
    # files for the same record.
    lib_hash = _hash_file_path(best_path)
    if lib_hash is None:
        reasons.append(
            f"aotriton.bundled_lib_hash: {best_path} unreadable"
        )

    return {
        "bundled_present": True,
        "bundled_version": bundled_version,
        "bundled_lib_hash": lib_hash,
        "bundled_images_dir_present": images_present,
        "installed_prefix": os.environ.get(AOTRITON_INSTALLED_PREFIX_ENV),
    }


def _capture_miopen(reasons: list[str]) -> dict[str, Any]:
    """Capture MIOpen build identity.

    Same shape as hipBLASLt -- header parse + lib hash + kernel-DB
    fingerprint. MIOpen is the GPU primitives library backing PyTorch's
    convolution (and certain fused) kernels on ROCm; its version drift
    is a major confound when training-loss numerics differ between
    environments.

    The kernel database lives under ``/opt/rocm/share/miopen/db/`` as
    ``.txt`` and ``.fdb.txt`` files keyed by gfx target -- the
    ``MIOPEN_SYSTEM_DB_PATH`` env var (in CANONICAL_ENV_VARS) overrides
    this path at runtime, which is captured separately so a cross-env
    diff catches the override.
    """
    header_text = _read_text_file(MIOPEN_VERSION_HEADER)
    rocm_release_tweak, package_version = _parse_version_header(
        header_text, _MIOPEN_TWEAK_RE, _MIOPEN_VERSION_RE
    )
    lib_hash = _hash_shared_library(MIOPEN_LIB_DIR, "libMIOpen.so")
    kernel_db_revision = _kernel_db_filename_fingerprint(
        MIOPEN_KERNEL_DB_DIR, suffixes=MIOPEN_KERNEL_DB_SUFFIXES
    )

    block: dict[str, Any] = {
        # Like hipblaslt and rocblas, this is the ROCm release identifier
        # that's shared across the whole release's library set -- NOT a
        # per-MIOpen upstream commit. lib_hash is the per-binary signal.
        "rocm_release_tweak": rocm_release_tweak,
        "package_version": package_version,
        "lib_hash": lib_hash,
        "kernel_db_revision": kernel_db_revision,
    }
    header_unreadable = header_text is None
    if rocm_release_tweak is None:
        if header_unreadable:
            reasons.append(
                f"miopen.rocm_release_tweak: {MIOPEN_VERSION_HEADER} not readable"
            )
        else:
            reasons.append(
                f"miopen.rocm_release_tweak: {MIOPEN_VERSION_HEADER} did not "
                "contain a readable MIOPEN_VERSION_TWEAK define"
            )
    if package_version is None:
        if header_unreadable:
            reasons.append(
                f"miopen.package_version: {MIOPEN_VERSION_HEADER} not readable"
            )
        else:
            reasons.append(
                f"miopen.package_version: {MIOPEN_VERSION_HEADER} did not "
                "contain MAJOR/MINOR/PATCH defines"
            )
    if lib_hash is None:
        reasons.append(
            f"miopen.lib_hash: {MIOPEN_LIB_DIR}/libMIOpen.so missing or unreadable"
        )
    if kernel_db_revision is None:
        reasons.append(
            "miopen.kernel_db_revision: directory missing/unreadable or no "
            f".txt files under {MIOPEN_KERNEL_DB_DIR}"
        )
    return block


def _capture_rccl(reasons: list[str]) -> dict[str, Any]:
    """Capture RCCL identity (AMD's NCCL-compatible collectives lib).

    The header packs the version into a single ``NCCL_VERSION_CODE``
    integer (e.g. ``22707`` -> ``2.27.7``). We capture both the raw
    integer (machine-comparable) and the decoded string (human-readable)
    plus the runtime lib hash. No kernel DB.
    """
    header_text = _read_text_file(RCCL_VERSION_HEADER)
    version_code, version_str = _parse_rccl_header(header_text)
    lib_hash = _hash_shared_library(RCCL_LIB_DIR, "librccl.so")

    block: dict[str, Any] = {
        "version_code": version_code,
        "version": version_str,
        "lib_hash": lib_hash,
    }
    if version_code is None:
        if header_text is None:
            reasons.append(
                f"rccl.version_code: {RCCL_VERSION_HEADER} not readable"
            )
        else:
            reasons.append(
                f"rccl.version_code: {RCCL_VERSION_HEADER} did not contain "
                "a readable NCCL_VERSION_CODE define"
            )
    if lib_hash is None:
        reasons.append(
            f"rccl.lib_hash: {RCCL_LIB_DIR}/librccl.so missing or unreadable"
        )
    return block


def _parse_rccl_header(text: str | None) -> tuple[int | None, str | None]:
    """Extract (NCCL_VERSION_CODE int, decoded MAJOR.MINOR.PATCH).

    NCCL/RCCL version encoding (per the upstream NCCL_VERSION macro):

    * X <= 2 AND Y <= 8:  code = X * 1000  + Y * 100  + Z   (legacy)
    * else:               code = X * 10000 + Y * 100  + Z   (modern)

    Detect which scheme by inspecting the magnitude of the trailing
    digits -- modern codes for X=2 are >= 20000 (since Y >= 9 implies
    Y*100 >= 900, so code >= 20900). Legacy maxes out around 2899
    (X=2, Y=8, Z=99).
    """
    if not text:
        return (None, None)
    m = _NCCL_VERSION_CODE_RE.search(text)
    if not m:
        return (None, None)
    try:
        code = int(m.group(1))
    except ValueError:
        return (None, None)

    # Disambiguate scheme by code magnitude.
    if code >= 10000:
        major = code // 10000
        minor = (code % 10000) // 100
        patch = code % 100
    else:
        major = code // 1000
        minor = (code % 1000) // 100
        patch = code % 100
    return (code, f"{major}.{minor}.{patch}")


def _capture_gpu_arch(reasons: list[str]) -> dict[str, Any]:
    """Capture detected GPU architecture(s) via rocm_agent_enumerator.

    The binary prints one gfx-target per detected GPU on stdout (e.g.
    ``gfx942\\ngfx942\\n...``). On most hosts this works without
    ``/dev/kfd`` access since the kernel module exposes the architecture
    via sysfs -- on hosts where it does require kfd, the probe records
    the failure and falls through to None.

    Always returns a fully-shaped dict; populated values are None
    plus a partial reason on failure.
    """
    default = {
        "agent_count": None,
        "gfx_targets": None,
        "agent_arch_counts": None,
    }

    bin_path = shutil.which(ROCM_AGENT_ENUMERATOR_BIN)
    if bin_path is None:
        # Try the canonical ROCM_BIN_DIR explicitly -- not always on
        # PATH for users who haven't sourced /etc/profile.d/rocm.sh.
        fallback = ROCM_BIN_DIR / ROCM_AGENT_ENUMERATOR_BIN
        if fallback.exists():
            bin_path = str(fallback)
        else:
            reasons.append(
                f"gpu_arch: {ROCM_AGENT_ENUMERATOR_BIN} not on PATH "
                f"(install rocminfo / rocm-core, or add {ROCM_BIN_DIR} to PATH)"
            )
            return default

    try:
        proc = subprocess.run(
            [bin_path],
            capture_output=True,
            text=True,
            timeout=SHORT_TIMEOUT_SEC,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        reasons.append(f"gpu_arch: {ROCM_AGENT_ENUMERATOR_BIN} invocation failed ({exc})")
        return default

    if proc.returncode != 0:
        # Common failure: not in render group / no /dev/kfd access.
        # Surface the stderr tail so the operator can act on it.
        stderr_tail = (proc.stderr or "").strip().splitlines()
        tail = stderr_tail[-1] if stderr_tail else "(no stderr)"
        reasons.append(
            f"gpu_arch: {ROCM_AGENT_ENUMERATOR_BIN} exited "
            f"{proc.returncode} ({tail[:200]})"
        )
        return default

    # Parse: one gfx-target per line. Filter empty lines and the
    # "gfx000" placeholder (sometimes printed for the host CPU agent).
    raw = [line.strip() for line in (proc.stdout or "").splitlines()]
    targets = [t for t in raw if t and t != "gfx000"]
    if not targets:
        reasons.append(
            f"gpu_arch: {ROCM_AGENT_ENUMERATOR_BIN} returned no GPU targets"
        )
        return default

    # `gfx_targets` is the sorted unique set for quick equality checks
    # ("did both hosts have the same arch lineup?"). `agent_arch_counts`
    # captures the distribution -- on a homogeneous box this is a
    # one-key dict like {"gfx942": 8}; on a mixed box it surfaces the
    # exact mix ({"gfx1100": 1, "gfx942": 6}). Strictly more compact
    # than carrying around an N-element flat list when N is large.
    counts: dict[str, int] = {}
    for t in targets:
        counts[t] = counts.get(t, 0) + 1
    return {
        "agent_count": len(targets),
        "gfx_targets": sorted(set(targets)),
        "agent_arch_counts": dict(sorted(counts.items())),
    }


def _capture_host(reasons: list[str]) -> dict[str, Any]:
    """Capture host-system identity (kernel + glibc + machine arch).

    Trivially derived from stdlib, but materially useful for cross-env
    debugging:

    * ``kernel_release`` (e.g. ``"5.15.0-174-generic"``): some ROCm
      releases break against older kernels, especially around amdgpu
      module changes. Today this only appears inside the ``rdhc``
      ``dkms_status`` blob, which is null on hosts without rdhc set
      up -- a fallback here is essential.
    * ``kernel_version`` (e.g. ``"#184-Ubuntu SMP Fri Mar 13 ..."``):
      the build-id flavour, distinguishes patched kernels at the same
      release tag.
    * ``machine`` (``"x86_64"`` / ``"aarch64"``): basic but matters
      when the same source tree gets built for multiple targets.
    * ``glibc_version`` (e.g. ``"glibc 2.35"``): C++ extensions
      compiled against a newer glibc fail to load on older hosts. The
      most common "compiled-against vs runtime drift" confound after
      HIP/ROCm versions themselves.
    """
    block: dict[str, Any] = {
        "kernel_release": None,
        "kernel_version": None,
        "machine": None,
        "glibc_version": None,
    }
    try:
        uname = os.uname()
    except (AttributeError, OSError) as exc:
        log.debug("os.uname() failed: %s", exc)
        reasons.append(f"host.kernel_release: os.uname() failed ({exc})")
    else:
        block["kernel_release"] = uname.release
        block["kernel_version"] = uname.version
        block["machine"] = uname.machine

    # os.confstr is POSIX-only; on Linux it returns the GNU libc
    # version string (e.g. "glibc 2.35"). Catch broad exceptions:
    # confstr can raise OSError on unsupported names, AttributeError
    # on Windows, ValueError on bad input.
    try:
        glibc = os.confstr("CS_GNU_LIBC_VERSION")
    except (AttributeError, OSError, ValueError) as exc:
        log.debug("os.confstr(CS_GNU_LIBC_VERSION) failed: %s", exc)
        reasons.append(f"host.glibc_version: os.confstr failed ({exc})")
    else:
        # Strip the redundant ``glibc `` prefix (the field name already
        # says glibc -- the value should be the bare version string,
        # e.g. "2.35"). Falls back to the raw return when prefix
        # absent (e.g. on a non-glibc libc that confstr happens to
        # answer for).
        if glibc:
            stripped = glibc.removeprefix("glibc ").strip()
            block["glibc_version"] = stripped or None
        else:
            reasons.append(
                "host.glibc_version: os.confstr(CS_GNU_LIBC_VERSION) returned empty"
            )
    return block


def _git_rev_parse_head(repo_or_submodule_dir: Path) -> str | None:
    """Run ``git -C <dir> rev-parse HEAD``. Fail-soft.

    Works for both ``.git`` directories AND submodules (which have a
    ``.git`` *file* pointing at the parent's ``.git/modules/...``).
    """
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_or_submodule_dir), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=SHORT_TIMEOUT_SEC,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        log.debug("git rev-parse failed for %s: %s", repo_or_submodule_dir, exc)
        return None
    if proc.returncode != 0:
        return None
    sha = (proc.stdout or "").strip()
    # Hard-validate to a hex SHA so a misconfigured `git` aliasing
    # rev-parse to something else can't poison the snapshot.
    if not sha or not all(c in "0123456789abcdefABCDEF" for c in sha):
        return None
    return sha
