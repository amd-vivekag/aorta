#!/usr/bin/env python3
"""
RCCL Fence Stress Test — Concurrent Collectives from Multiple Process Groups

Reproduces silent data corruption caused by RCCL not synchronizing between
concurrent collectives on different streams when using multiple process groups.
This is documented RCCL behavior (unlike NCCL which handles it correctly).

Root cause: on AMD gfx942/gfx950, concurrent collectives from different PGs
dispatched to different non-default CUDA streams can corrupt each other's data
when GPU_MAX_HW_QUEUES >= 4.

Known mitigations:
  - NCCL_LAUNCH_ORDER_IMPLICIT=1 + RCCL_ENABLE_CONTEXT_TRACKING=1
    (serializes all RCCL ops through a device-wide context tracking stream)
  - GPU_MAX_HW_QUEUES=2 (reduces HW parallelism so collisions don't happen)

Usage (single node, 8 GPUs — expose bug):
  GPU_MAX_HW_QUEUES=4 torchrun --nproc_per_node=8 scripts/rccl_fence_stress.py \
      --num-iters 5000 --buf-size 500000 --mode burst

Usage (single node, 8 GPUs — control, should pass):
  torchrun --nproc_per_node=8 scripts/rccl_fence_stress.py \
      --num-iters 5000 --buf-size 500000 --mode burst --implicit-order

Usage (single node, 2 GPUs — quick smoke test):
  GPU_MAX_HW_QUEUES=4 torchrun --nproc_per_node=2 scripts/rccl_fence_stress.py \
      --num-iters 1000 --buf-size 100000

Usage (3 nodes, 24 GPUs — full stress):
  torchrun --nnodes=3 --nproc_per_node=8 \
      --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
      scripts/rccl_fence_stress.py \
      --num-iters 10000 --buf-size 1000000 --mode overlapped \
      --compute-overlap --compute-size 2048

Usage (control — implicit order serialization, should pass):
  torchrun --nproc_per_node=8 scripts/rccl_fence_stress.py \
      --num-iters 10000 --implicit-order

Usage (control — reduced HW queues, should pass):
  GPU_MAX_HW_QUEUES=2 torchrun --nproc_per_node=8 scripts/rccl_fence_stress.py \
      --num-iters 10000

Usage (sustained mode — many iters without sync, then verify):
  GPU_MAX_HW_QUEUES=4 torchrun --nproc_per_node=8 scripts/rccl_fence_stress.py \
      --num-iters 10000 --mode sustained --check-interval 500

Usage (adjacent buffers — stress cache line sharing):
  GPU_MAX_HW_QUEUES=4 torchrun --nproc_per_node=8 scripts/rccl_fence_stress.py \
      --num-iters 5000 --adjacent-buffers --buf-size 250000

Usage (bfloat16 — test narrower data width):
  GPU_MAX_HW_QUEUES=4 torchrun --nproc_per_node=8 scripts/rccl_fence_stress.py \
      --num-iters 5000 --dtype bfloat16 --mode burst

Usage (firehose — minimum CPU overhead, maximum GPU concurrency):
  GPU_MAX_HW_QUEUES=4 torchrun --nproc_per_node=8 scripts/rccl_fence_stress.py \
      --num-iters 50000 --mode firehose --check-interval 1000

Usage (3 nodes, firehose + adjacent buffers — maximum stress):
  torchrun --nnodes=3 --nproc_per_node=8 \
      --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
      scripts/rccl_fence_stress.py \
      --num-iters 50000 --mode firehose --check-interval 1000 \
      --adjacent-buffers --buf-size 500000 --num-pgs 4
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist

# ---------------------------------------------------------------------------
# Environment setup — must happen before any HIP/CUDA call
# ---------------------------------------------------------------------------

_ENV_DEFAULTS = {
    "GPU_MAX_HW_QUEUES": "4",
}


def _apply_env_defaults() -> None:
    for k, v in _ENV_DEFAULTS.items():
        if k not in os.environ:
            os.environ[k] = v


def _apply_implicit_order() -> None:
    os.environ["NCCL_LAUNCH_ORDER_IMPLICIT"] = "1"
    os.environ["RCCL_ENABLE_CONTEXT_TRACKING"] = "1"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

log = logging.getLogger("rccl_fence_stress")


def _setup_logging(rank: int) -> None:
    level = logging.INFO if rank == 0 else logging.WARNING
    fmt = f"[rank {rank}] %(levelname)s %(message)s"
    logging.basicConfig(level=level, format=fmt, stream=sys.stderr)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    num_iters: int = 10000
    buf_size: int = 1_000_000
    check_interval: int = 1
    dtype_str: str = "float32"
    num_concurrent_collectives: int = 3
    compute_overlap: bool = False
    compute_size: int = 1024
    implicit_order: bool = False
    mode: str = "burst"
    adjacent_buffers: bool = False
    num_pgs: int = 3

    @property
    def dtype(self) -> torch.dtype:
        return {"float32": torch.float32, "bfloat16": torch.bfloat16}[self.dtype_str]


def parse_args() -> Config:
    p = argparse.ArgumentParser(
        description="RCCL fence stress test — concurrent collectives from multiple PGs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--num-iters", type=int, default=10000)
    p.add_argument("--buf-size", type=int, default=1_000_000)
    p.add_argument("--check-interval", type=int, default=1,
                    help="Verify every N iterations (1=every iter)")
    p.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    p.add_argument("--num-concurrent-collectives", type=int, default=3,
                    help="Concurrent collectives per PG per iteration")
    p.add_argument("--compute-overlap", action="store_true",
                    help="Interleave GEMM compute between collective launches")
    p.add_argument("--compute-size", type=int, default=1024,
                    help="Matrix dimension for overlap GEMMs")
    p.add_argument("--implicit-order", action="store_true",
                    help="Enable NCCL_LAUNCH_ORDER_IMPLICIT=1 + RCCL_ENABLE_CONTEXT_TRACKING=1 (control)")
    p.add_argument("--mode", choices=["burst", "overlapped", "sustained", "firehose"],
                    default="burst",
                    help="Stress pattern: burst | overlapped | sustained | firehose")
    p.add_argument("--adjacent-buffers", action="store_true",
                    help="Allocate all buffers from a single contiguous tensor (stress cache lines)")
    p.add_argument("--num-pgs", type=int, default=3,
                    choices=[1, 2, 3, 4],
                    help="Number of process groups (1=single PG, all collectives "
                         "serialized on one NCCL stream; 2-4=concurrent PGs)")

    # Diagnostic env var overrides — applied BEFORE any CUDA call
    p.add_argument("--hwq", type=int, default=None,
                    help="Set GPU_MAX_HW_QUEUES (default: 4 if not in env)")
    p.add_argument("--max-nchannels", type=int, default=None,
                    help="Set NCCL_MAX_NCHANNELS (reduces RCCL internal parallelism)")
    p.add_argument("--nccl-proto", type=str, default=None,
                    choices=["Simple", "LL", "LL128"],
                    help="Set NCCL_PROTO (LL avoids threadfence entirely)")
    p.add_argument("--no-cca", action="store_true",
                    help="Set PYTORCH_NO_CUDA_MEMORY_CACHING=1 (bypass CCA entirely)")
    args = p.parse_args()
    return Config(
        num_iters=args.num_iters,
        buf_size=args.buf_size,
        check_interval=args.check_interval,
        dtype_str=args.dtype,
        num_concurrent_collectives=args.num_concurrent_collectives,
        compute_overlap=args.compute_overlap,
        compute_size=args.compute_size,
        implicit_order=args.implicit_order,
        mode=args.mode,
        adjacent_buffers=args.adjacent_buffers,
        num_pgs=args.num_pgs,
    ), args


# ---------------------------------------------------------------------------
# Diagnostics — log environment, GPU info, RCCL version at startup
# ---------------------------------------------------------------------------

_RCCL_ENV_VARS = [
    "GPU_MAX_HW_QUEUES",
    "HSA_ENABLE_SDMA",
    "NCCL_LAUNCH_ORDER_IMPLICIT",
    "RCCL_ENABLE_CONTEXT_TRACKING",
    "RCCL_GFX9_CHEAP_FENCE_OFF",
    "RCCL_GFX942_CHEAP_FENCE_OFF",
    "ROC_AQL_QUEUE_SIZE",
    "ROC_SIGNAL_POOL_SIZE",
    "NCCL_DEBUG",
    "NCCL_PROTO",
    "NCCL_ALGO",
    "NCCL_MAX_NCHANNELS",
    "NCCL_MIN_NCHANNELS",
    "PYTORCH_NO_CUDA_MEMORY_CACHING",
    "PYTORCH_CUDA_ALLOC_CONF",
    "HIP_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
]


def log_diagnostics(rank: int, world_size: int, cfg: Config) -> None:
    if rank != 0:
        return

    log.info("=" * 72)
    log.info("RCCL FENCE STRESS TEST")
    log.info("=" * 72)

    # GPU info
    dev = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(dev)
    log.info("GPU: %s  (gcnArchName: %s)", props.name, getattr(props, "gcnArchName", "N/A"))
    log.info("GPU count (this node): %d", torch.cuda.device_count())
    log.info("World size: %d", world_size)

    # PyTorch / NCCL version
    log.info("PyTorch: %s", torch.__version__)
    log.info("CUDA/HIP: %s", torch.version.cuda or torch.version.hip or "unknown")
    nccl_ver = "unknown"
    try:
        nccl_ver = ".".join(str(x) for x in torch.cuda.nccl.version())
    except Exception:
        pass
    log.info("NCCL/RCCL version: %s", nccl_ver)

    # Env vars
    log.info("--- Environment Variables ---")
    for var in _RCCL_ENV_VARS:
        val = os.environ.get(var, "<not set>")
        log.info("  %s = %s", var, val)

    # Config
    log.info("--- Test Configuration ---")
    log.info("  mode              = %s", cfg.mode)
    log.info("  num_iters         = %d", cfg.num_iters)
    log.info("  buf_size          = %d elements", cfg.buf_size)
    log.info("  dtype             = %s", cfg.dtype_str)
    log.info("  check_interval    = %d", cfg.check_interval)
    log.info("  num_pgs           = %d", cfg.num_pgs)
    log.info("  concurrent_colls  = %d", cfg.num_concurrent_collectives)
    log.info("  compute_overlap   = %s", cfg.compute_overlap)
    log.info("  compute_size      = %d", cfg.compute_size)
    log.info("  adjacent_buffers  = %s", cfg.adjacent_buffers)
    log.info("  implicit_order    = %s", cfg.implicit_order)
    log.info("=" * 72)


# ---------------------------------------------------------------------------
# Corruption record
# ---------------------------------------------------------------------------

@dataclass
class CorruptionRecord:
    iteration: int
    collective: str
    pg_index: int
    expected_sample: list[float]
    actual_sample: list[float]
    num_mismatched: int
    total_elements: int
    has_nan: bool
    has_inf: bool

    def __str__(self) -> str:
        pct = 100.0 * self.num_mismatched / max(self.total_elements, 1)
        parts = [
            f"iter={self.iteration} pg={self.pg_index} op={self.collective}",
            f"mismatched={self.num_mismatched}/{self.total_elements} ({pct:.2f}%)",
        ]
        if self.has_nan:
            parts.append("HAS_NaN")
        if self.has_inf:
            parts.append("HAS_Inf")
        parts.append(f"expected[:8]={self.expected_sample}")
        parts.append(f"actual[:8]={self.actual_sample}")
        return "  ".join(parts)


# ---------------------------------------------------------------------------
# Buffer manager — handles allocation (adjacent or separate) and fill patterns
# ---------------------------------------------------------------------------

class BufferManager:
    """Manages GPU buffers for all process groups and collectives."""

    def __init__(self, rank: int, world_size: int, cfg: Config, num_pgs: int):
        self.rank = rank
        self.world_size = world_size
        self.cfg = cfg
        self.num_pgs = num_pgs
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.dtype = cfg.dtype

        # Each PG gets buffers for: all_reduce, reduce_scatter, all_gather, all_to_all
        # We allocate `num_concurrent_collectives` sets per PG.
        # Collective types assigned round-robin to create variety.
        # Only use collectives with naturally separate input/output buffers.
        # all_reduce is excluded because it's in-place — the output overwrites
        # the input, causing accumulation across iterations when there's no
        # inter-iteration sync.  The remaining 3 types all have separate
        # input and output buffers, so the input is never modified by the
        # collective and can be safely refilled without synchronization.
        self.collective_types = ["reduce_scatter", "all_gather", "all_to_all"]

        self._backing: torch.Tensor | None = None
        self.bufs: dict[tuple[int, int], dict[str, torch.Tensor]] = {}
        self._allocate()

    def _buf_size_for(self, pg_idx: int, coll_idx: int) -> int:
        """Variable buffer sizes to create timing asymmetry."""
        base = self.cfg.buf_size
        scale_factor = 1.0 + 0.3 * ((pg_idx + coll_idx) % 4)
        return int(base * scale_factor)

    def _coll_type_for(self, pg_idx: int, coll_idx: int) -> str:
        return self.collective_types[(pg_idx + coll_idx) % len(self.collective_types)]

    def _allocate(self) -> None:
        specs: list[tuple[tuple[int, int], str, int]] = []
        for pg_idx in range(self.num_pgs):
            for coll_idx in range(self.cfg.num_concurrent_collectives):
                ctype = self._coll_type_for(pg_idx, coll_idx)
                raw_size = self._buf_size_for(pg_idx, coll_idx)
                # all_to_all and all_gather need world_size-aligned buffers
                size = self._aligned_size(raw_size, ctype)
                specs.append(((pg_idx, coll_idx), ctype, size))

        if self.cfg.adjacent_buffers:
            # Each collective needs send + recv buffers
            total_elems = sum(self._total_buf_elems(s, ct) for _, ct, s in specs)
            self._backing = torch.zeros(total_elems, dtype=self.dtype, device=self.device)
            offset = 0
            for key, ctype, size in specs:
                bufs, consumed = self._slice_bufs(self._backing, offset, size, ctype)
                self.bufs[key] = bufs
                offset += consumed
        else:
            for key, ctype, size in specs:
                self.bufs[key] = self._alloc_bufs(size, ctype)

    def _aligned_size(self, size: int, ctype: str) -> int:
        ws = self.world_size
        if ctype in ("all_to_all", "all_gather", "reduce_scatter"):
            return ((size + ws - 1) // ws) * ws
        return size

    def _total_buf_elems(self, size: int, ctype: str) -> int:
        if ctype == "all_to_all":
            return size * 2  # send + recv
        if ctype == "all_gather":
            return size // self.world_size + size  # input shard + output
        if ctype == "reduce_scatter":
            return size + size // self.world_size  # input + output shard
        return size  # fallback (should not be reached)

    def _slice_bufs(self, backing: torch.Tensor, offset: int, size: int,
                    ctype: str) -> tuple[dict[str, torch.Tensor], int]:
        bufs: dict[str, torch.Tensor] = {}
        ws = self.world_size
        consumed = 0
        if ctype == "all_to_all":
            bufs["send"] = backing[offset:offset + size]
            bufs["recv"] = backing[offset + size:offset + 2 * size]
            consumed = 2 * size
        elif ctype == "all_gather":
            shard = size // ws
            bufs["input"] = backing[offset:offset + shard]
            bufs["output"] = backing[offset + shard:offset + shard + size]
            consumed = shard + size
        elif ctype == "reduce_scatter":
            shard = size // ws
            bufs["input"] = backing[offset:offset + size]
            bufs["output"] = backing[offset + size:offset + size + shard]
            consumed = size + shard
        else:
            bufs["send"] = backing[offset:offset + size]
            consumed = size
        bufs["_size"] = torch.tensor(size)  # metadata
        bufs["_ctype"] = torch.tensor(0)    # placeholder
        return bufs, consumed

    def _alloc_bufs(self, size: int, ctype: str) -> dict[str, torch.Tensor]:
        ws = self.world_size
        bufs: dict[str, torch.Tensor] = {}
        if ctype == "all_to_all":
            bufs["send"] = torch.zeros(size, dtype=self.dtype, device=self.device)
            bufs["recv"] = torch.zeros(size, dtype=self.dtype, device=self.device)
        elif ctype == "all_gather":
            shard = size // ws
            bufs["input"] = torch.zeros(shard, dtype=self.dtype, device=self.device)
            bufs["output"] = torch.zeros(size, dtype=self.dtype, device=self.device)
        elif ctype == "reduce_scatter":
            shard = size // ws
            bufs["input"] = torch.zeros(size, dtype=self.dtype, device=self.device)
            bufs["output"] = torch.zeros(shard, dtype=self.dtype, device=self.device)
        else:
            bufs["send"] = torch.zeros(size, dtype=self.dtype, device=self.device)
        return bufs

    def fill_pattern(self, iteration: int, pg_idx: int, coll_idx: int) -> None:
        """Fill buffers with iteration-dependent known pattern.

        Pattern encodes (iteration, rank, pg_idx, coll_idx) so stale data
        from previous iterations or wrong ranks can be detected.
        """
        ctype = self._coll_type_for(pg_idx, coll_idx)
        bufs = self.bufs[(pg_idx, coll_idx)]
        base_val = float((iteration % 1000) * 100 + self.rank * 10 + pg_idx + coll_idx)

        if ctype == "all_to_all":
            bufs["send"].fill_(base_val)
            bufs["recv"].fill_(-1.0)
        elif ctype == "all_gather":
            bufs["input"].fill_(float(self.rank + 1 + iteration % 1000))
            bufs["output"].fill_(-1.0)
        elif ctype == "reduce_scatter":
            bufs["input"].fill_(float(self.rank + 1 + iteration % 1000))
            bufs["output"].fill_(-1.0)

    def expected_result(self, iteration: int, pg_idx: int, coll_idx: int
                        ) -> dict[str, float | torch.Tensor]:
        """Compute expected output values after the collective completes."""
        ctype = self._coll_type_for(pg_idx, coll_idx)
        ws = self.world_size
        iter_mod = iteration % 1000

        if ctype == "all_to_all":
            # After all_to_all_single with equal splits, each rank's recv chunk i
            # came from rank i. Rank i's send value was:
            #   (iter_mod * 100 + i * 10 + pg_idx + coll_idx)
            bufs = self.bufs[(pg_idx, coll_idx)]
            chunk_size = bufs["recv"].numel() // ws
            expected = torch.zeros_like(bufs["recv"])
            for src_rank in range(ws):
                val = float(iter_mod * 100 + src_rank * 10 + pg_idx + coll_idx)
                start = src_rank * chunk_size
                expected[start:start + chunk_size].fill_(val)
            return {"recv": expected}
        elif ctype == "all_gather":
            bufs = self.bufs[(pg_idx, coll_idx)]
            shard = bufs["input"].numel()
            expected = torch.zeros_like(bufs["output"])
            for r in range(ws):
                val = float(r + 1 + iter_mod)
                expected[r * shard:(r + 1) * shard].fill_(val)
            return {"output": expected}
        elif ctype == "reduce_scatter":
            # reduce_scatter: sum across ranks, then scatter shard to each rank.
            # Input on every rank is (rank + 1 + iter_mod).
            # Sum = ws*(ws+1)/2 + ws*iter_mod.
            expected_val = float(ws * (ws + 1) / 2 + ws * iter_mod)
            return {"output": expected_val}
        return {}


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_collective(
    buf_mgr: BufferManager,
    iteration: int,
    pg_idx: int,
    coll_idx: int,
) -> CorruptionRecord | None:
    """Check a single collective's output against expected values."""
    ctype = buf_mgr._coll_type_for(pg_idx, coll_idx)
    bufs = buf_mgr.bufs[(pg_idx, coll_idx)]
    expected_map = buf_mgr.expected_result(iteration, pg_idx, coll_idx)

    for buf_key, expected in expected_map.items():
        actual = bufs[buf_key]
        if isinstance(expected, (int, float)):
            expected_t = torch.full_like(actual, expected)
        else:
            expected_t = expected.to(actual.device)

        # Use tolerance for bfloat16
        atol = 1.0 if buf_mgr.dtype == torch.bfloat16 else 0.5
        mismatch = ~torch.isclose(actual, expected_t, atol=atol, rtol=1e-3)
        nan_mask = torch.isnan(actual)
        inf_mask = torch.isinf(actual)
        bad = mismatch | nan_mask | inf_mask
        num_bad = bad.sum().item()

        if num_bad > 0:
            bad_indices = bad.nonzero(as_tuple=True)[0][:8]
            return CorruptionRecord(
                iteration=iteration,
                collective=f"{ctype}.{buf_key}",
                pg_index=pg_idx,
                expected_sample=expected_t[bad_indices].tolist(),
                actual_sample=actual[bad_indices].tolist(),
                num_mismatched=int(num_bad),
                total_elements=actual.numel(),
                has_nan=bool(nan_mask.any().item()),
                has_inf=bool(inf_mask.any().item()),
            )
    return None


# ---------------------------------------------------------------------------
# Collective launcher
# ---------------------------------------------------------------------------

def launch_collective(
    buf_mgr: BufferManager,
    pg: dist.ProcessGroup,
    pg_idx: int,
    coll_idx: int,
    stream: torch.cuda.Stream,
) -> dist.Work | None:
    """Launch a single async collective on the given stream."""
    ctype = buf_mgr._coll_type_for(pg_idx, coll_idx)
    bufs = buf_mgr.bufs[(pg_idx, coll_idx)]
    ws = buf_mgr.world_size

    with torch.cuda.stream(stream):
        if ctype == "all_to_all":
            return dist.all_to_all_single(
                bufs["recv"], bufs["send"],
                output_split_sizes=[bufs["recv"].numel() // ws] * ws,
                input_split_sizes=[bufs["send"].numel() // ws] * ws,
                group=pg, async_op=True,
            )
        elif ctype == "all_gather":
            shard = bufs["input"]
            out_list = list(bufs["output"].chunk(ws))
            return dist.all_gather(out_list, shard, group=pg, async_op=True)
        elif ctype == "reduce_scatter":
            in_list = list(bufs["input"].chunk(ws))
            return dist.reduce_scatter(bufs["output"], in_list,
                                       op=dist.ReduceOp.SUM, group=pg, async_op=True)
    return None


# ---------------------------------------------------------------------------
# Compute overlap (GEMM) helper
# ---------------------------------------------------------------------------

def do_overlap_gemm(stream: torch.cuda.Stream, size: int, dtype: torch.dtype,
                    device: torch.device) -> None:
    """Run a GEMM on the given stream to simulate real compute overlap."""
    with torch.cuda.stream(stream):
        a = torch.randn(size, size, dtype=dtype, device=device)
        b = torch.randn(size, size, dtype=dtype, device=device)
        torch.mm(a, b)


# ---------------------------------------------------------------------------
# Stress modes
# ---------------------------------------------------------------------------

def _fill_on_stream(
    buf_mgr: BufferManager,
    streams: list[torch.cuda.Stream],
    cfg: Config,
    iteration: int,
) -> None:
    """Fill buffers ON the same stream the collective will use.

    Critical: fill_pattern() must run on the PG's stream, NOT the default
    stream.  Otherwise:
      1. Default-stream fill_() adds accidental serialization (GPU work on
         default stream separates collective rounds, reducing concurrency).
      2. The fill_() on the default stream is NOT ordered relative to the
         collective on the non-default PG stream — ProcessGroupNCCL records
         an event on the CURRENT stream (the PG stream) when launching,
         which does NOT capture default-stream work.  The collective could
         read stale data, causing false-positive verification failures.
    """
    for pg_idx in range(cfg.num_pgs):
        with torch.cuda.stream(streams[pg_idx]):
            for coll_idx in range(cfg.num_concurrent_collectives):
                buf_mgr.fill_pattern(iteration, pg_idx, coll_idx)


def run_burst(
    buf_mgr: BufferManager,
    pgs: list[dist.ProcessGroup],
    streams: list[torch.cuda.Stream],
    compute_stream: torch.cuda.Stream,
    cfg: Config,
    iteration: int,
) -> list[dist.Work]:
    """Burst: launch all collectives from all PGs rapidly, no sync between them."""
    _fill_on_stream(buf_mgr, streams, cfg, iteration)

    works: list[dist.Work] = []
    for pg_idx in range(cfg.num_pgs):
        stream = streams[pg_idx]
        for coll_idx in range(cfg.num_concurrent_collectives):
            w = launch_collective(buf_mgr, pgs[pg_idx], pg_idx, coll_idx, stream)
            if w is not None:
                works.append(w)
    return works


def run_overlapped(
    buf_mgr: BufferManager,
    pgs: list[dist.ProcessGroup],
    streams: list[torch.cuda.Stream],
    compute_stream: torch.cuda.Stream,
    cfg: Config,
    iteration: int,
) -> list[dist.Work]:
    """Overlapped: interleave collectives from different PGs with compute kernels."""
    _fill_on_stream(buf_mgr, streams, cfg, iteration)

    works: list[dist.Work] = []
    device = buf_mgr.device

    for coll_idx in range(cfg.num_concurrent_collectives):
        for pg_idx in range(cfg.num_pgs):
            w = launch_collective(buf_mgr, pgs[pg_idx], pg_idx, coll_idx, streams[pg_idx])
            if w is not None:
                works.append(w)

        if cfg.compute_overlap:
            do_overlap_gemm(compute_stream, cfg.compute_size, cfg.dtype, device)

    return works


def run_sustained(
    buf_mgr: BufferManager,
    pgs: list[dist.ProcessGroup],
    streams: list[torch.cuda.Stream],
    compute_stream: torch.cuda.Stream,
    cfg: Config,
    iteration: int,
) -> list[dist.Work]:
    """Sustained: launch collectives without any sync — caller decides when to sync."""
    _fill_on_stream(buf_mgr, streams, cfg, iteration)

    works: list[dist.Work] = []
    for pg_idx in range(cfg.num_pgs):
        stream = streams[pg_idx]
        for coll_idx in range(cfg.num_concurrent_collectives):
            w = launch_collective(buf_mgr, pgs[pg_idx], pg_idx, coll_idx, stream)
            if w is not None:
                works.append(w)

    if cfg.compute_overlap:
        do_overlap_gemm(compute_stream, cfg.compute_size, cfg.dtype, buf_mgr.device)

    return works


def run_firehose(
    buf_mgr: BufferManager,
    pgs: list[dist.ProcessGroup],
    streams: list[torch.cuda.Stream],
    compute_stream: torch.cuda.Stream,
    cfg: Config,
    iteration: int,
) -> list[dist.Work]:
    """Firehose: absolute minimum CPU overhead between collective launches.

    Fills buffers only on the FIRST iteration.  Subsequent iterations reuse
    the same data (same fill values) and just re-launch collectives as fast
    as possible.  Verification uses the iteration-0 expected values.

    This minimizes Python/CPU overhead to maximize GPU-side concurrency
    and the chance of RCCL internal resource aliasing.  The trade-off is
    that stale-data-from-previous-iteration detection is impossible (since
    we don't change patterns).  But cross-collective corruption and NaN/Inf
    are still detectable.
    """
    if iteration <= 1:
        _fill_on_stream(buf_mgr, streams, cfg, iteration=1)

    works: list[dist.Work] = []
    for pg_idx in range(cfg.num_pgs):
        stream = streams[pg_idx]
        for coll_idx in range(cfg.num_concurrent_collectives):
            w = launch_collective(buf_mgr, pgs[pg_idx], pg_idx, coll_idx, stream)
            if w is not None:
                works.append(w)
    return works


_MODE_FNS = {
    "burst": run_burst,
    "overlapped": run_overlapped,
    "sustained": run_sustained,
    "firehose": run_firehose,
}


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

@dataclass
class Stats:
    total_iters: int = 0
    checked_iters: int = 0
    corruptions: list[CorruptionRecord] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def elapsed(self) -> float:
        return max(self.end_time - self.start_time, 1e-9)

    @property
    def iters_per_sec(self) -> float:
        return self.total_iters / self.elapsed


def run(cfg: Config) -> Stats:
    # ---- Init dist ----
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    _setup_logging(rank)
    log_diagnostics(rank, world_size, cfg)

    if world_size < 2:
        log.error("Need at least 2 ranks. Got %d.", world_size)
        dist.destroy_process_group()
        sys.exit(1)

    # ---- Create process groups ----
    all_ranks = list(range(world_size))
    pgs: list[dist.ProcessGroup] = []
    for i in range(cfg.num_pgs):
        pg = dist.new_group(ranks=all_ranks, backend="nccl")
        pgs.append(pg)
        if rank == 0:
            log.info("Created process group %d (all %d ranks)", i, world_size)

    if rank == 0 and cfg.num_pgs == 1:
        log.info("** SINGLE PG MODE: all collectives share one NCCL communicator **")
        log.info("** Collectives are serialized on one NCCL stream (no cross-PG concurrency) **")
        log.info("** If this is clean but --num-pgs>=2 is not, cross-PG concurrency is the trigger **")

    # ---- Create non-default streams (one per PG + one for compute) ----
    streams = [torch.cuda.Stream(device=device) for _ in range(cfg.num_pgs)]
    compute_stream = torch.cuda.Stream(device=device)

    if rank == 0:
        log.info("Created %d non-default streams for PGs + 1 compute stream", cfg.num_pgs)

    # ---- Allocate buffers ----
    buf_mgr = BufferManager(rank, world_size, cfg, cfg.num_pgs)
    if rank == 0:
        total_bytes = 0
        elem_size = 2 if cfg.dtype == torch.bfloat16 else 4
        for key, bufs in buf_mgr.bufs.items():
            for bk, bv in bufs.items():
                if isinstance(bv, torch.Tensor) and bv.device.type == "cuda" and bk[0] != "_":
                    total_bytes += bv.numel() * elem_size
        log.info("Total GPU buffer memory: %.2f MB  (adjacent=%s)",
                 total_bytes / 1e6, cfg.adjacent_buffers)

    # ---- Warmup (one iteration with sync) ----
    if rank == 0:
        log.info("Running warmup iteration...")

    mode_fn = _MODE_FNS[cfg.mode]
    works = mode_fn(buf_mgr, pgs, streams, compute_stream, cfg, iteration=0)
    for w in works:
        w.wait()
    torch.cuda.synchronize()
    dist.barrier()

    if rank == 0:
        log.info("Warmup complete. Starting %d stress iterations (mode=%s)...",
                 cfg.num_iters, cfg.mode)

    # ---- Main stress loop ----
    stats = Stats(start_time=time.monotonic())
    first_corruption_iter: int | None = None

    # In firehose mode, verification uses iteration=1 expected values since
    # fill only happens once.  In other modes, use the current iteration.
    is_no_sync_mode = cfg.mode in ("sustained", "firehose")

    for it in range(1, cfg.num_iters + 1):
        works = mode_fn(buf_mgr, pgs, streams, compute_stream, cfg, iteration=it)

        should_check = (it % cfg.check_interval == 0) or (it == cfg.num_iters)

        if should_check:
            for w in works:
                if w is not None:
                    w.wait()
            torch.cuda.synchronize()

            stats.checked_iters += 1
            verify_iter = 1 if cfg.mode == "firehose" else it
            for pg_idx in range(cfg.num_pgs):
                for coll_idx in range(cfg.num_concurrent_collectives):
                    rec = verify_collective(buf_mgr, verify_iter, pg_idx, coll_idx)
                    if rec is not None:
                        stats.corruptions.append(rec)
                        if first_corruption_iter is None:
                            first_corruption_iter = it
                        log.warning("CORRUPTION: %s", rec)
        elif is_no_sync_mode:
            # sustained/firehose: do NOT wait — let GPU pile up work
            # across iterations for maximum concurrency.  Work objects
            # are intentionally dropped; ProcessGroupNCCL's destructor
            # will NOT block (the GPU streams keep running).
            pass
        else:
            # burst/overlapped: NO sync between iterations.  All
            # collectives use separate input/output buffers — the input
            # is never modified by the collective.  The fill only writes
            # to input buffers on the PG stream; the collective reads
            # inputs on the NCCL stream (ordered after the PG stream
            # fill via the event ProcessGroupNCCL records at launch).
            # The output buffer is only read at check time after
            # torch.cuda.synchronize().  This allows sustained GPU-side
            # concurrency across iteration boundaries — matching the
            # a zero-sync pipeline.
            pass

        stats.total_iters = it

        if rank == 0 and it % 1000 == 0:
            elapsed = time.monotonic() - stats.start_time
            log.info("  iter %d/%d  (%.1f it/s)  corruptions=%d",
                     it, cfg.num_iters, it / elapsed, len(stats.corruptions))

    stats.end_time = time.monotonic()
    torch.cuda.synchronize()
    dist.barrier()

    # ---- Aggregate results across ranks ----
    local_corruption_count = torch.tensor([len(stats.corruptions)], device=device)
    dist.all_reduce(local_corruption_count, op=dist.ReduceOp.SUM)
    global_corruptions = int(local_corruption_count.item())

    # ---- Report ----
    if rank == 0:
        log.info("=" * 72)
        log.info("RESULTS")
        log.info("=" * 72)
        log.info("  Total iterations:    %d", stats.total_iters)
        log.info("  Checked iterations:  %d", stats.checked_iters)
        log.info("  Throughput:          %.1f iter/s", stats.iters_per_sec)
        log.info("  Elapsed:             %.1f s", stats.elapsed)

        # Per-element throughput estimate
        elem_size = 2 if cfg.dtype == torch.bfloat16 else 4
        total_colls = cfg.num_pgs * cfg.num_concurrent_collectives
        bytes_per_iter = total_colls * cfg.buf_size * elem_size
        gbps = (bytes_per_iter * stats.iters_per_sec) / 1e9
        log.info("  Collective throughput: ~%.2f GB/s (aggregate)", gbps)

        if global_corruptions == 0:
            log.info("")
            log.info("PASSED: No corruption detected in %d iterations", stats.total_iters)
            log.info("VERDICT: No RCCL fence bug detected with current settings.")
        else:
            log.info("")
            local_count = len(stats.corruptions)
            log.info("FAILED: %d corruption events on rank 0 (%d global across all ranks)",
                     local_count, global_corruptions)
            if first_corruption_iter is not None:
                log.info("First corruption at iteration %d", first_corruption_iter)

            # Print up to 20 corruption records from this rank
            for i, rec in enumerate(stats.corruptions[:20]):
                log.info("  [%d] %s", i, rec)
            if len(stats.corruptions) > 20:
                log.info("  ... and %d more", len(stats.corruptions) - 20)

            log.info("")
            log.info("VERDICT: RCCL DATA CORRUPTION DETECTED!")
            log.info("  Concurrent collectives from different process groups on different")
            log.info("  non-default streams produce corrupted results.")
            log.info("  Try: --implicit-order  (sets NCCL_LAUNCH_ORDER_IMPLICIT=1 +")
            log.info("        RCCL_ENABLE_CONTEXT_TRACKING=1) to serialize and verify fix.")
        log.info("=" * 72)

    dist.destroy_process_group()
    return stats


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    cfg, raw_args = parse_args()

    # Diagnostic env vars — must be set BEFORE any CUDA/HIP call
    if raw_args.hwq is not None:
        os.environ["GPU_MAX_HW_QUEUES"] = str(raw_args.hwq)
    if raw_args.max_nchannels is not None:
        os.environ["NCCL_MAX_NCHANNELS"] = str(raw_args.max_nchannels)
        os.environ["NCCL_MIN_NCHANNELS"] = str(raw_args.max_nchannels)
    if raw_args.nccl_proto is not None:
        os.environ["NCCL_PROTO"] = raw_args.nccl_proto
    if raw_args.no_cca:
        os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"

    if cfg.implicit_order:
        _apply_implicit_order()

    _apply_env_defaults()

    stats = run(cfg)

    if stats.corruptions:
        sys.exit(1)


if __name__ == "__main__":
    main()
