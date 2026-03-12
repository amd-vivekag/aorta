"""
CLI entry point for the RCCL race condition reproducer.

Supports multiple workload modes via --mode:
  - default:        TorchRec-like pattern (H2D + all_to_all + all_reduce)
  - ddp:            DDP pattern (H2D prefetch + gradient all_reduce)
  - fsdp:           FSDP pattern (per-layer all_gather + reduce_scatter)
  - eval_pipelined: Pipelined eval loop for NaN investigation (Experiments A/B)

Usage:
    # Default mode (TorchRec-like)
    GPU_MAX_HW_QUEUES=4 torchrun --nproc_per_node=8 -m aorta.race \\
        --warmup 100 --verify 10000

    # Eval pipelined mode (Experiment A)
    torchrun --nproc_per_node=2 -m aorta.race --mode eval_pipelined \\
        --batch-size 512 --use-compile --sync-policy none --verify 500

    # Eval pipelined mode via YAML config
    torchrun --nproc_per_node=2 -m aorta.race \\
        --config config/race/eval_exp_a_reproduce.yaml

    # Same-stream mode (definitive runtime bug test)
    torchrun --nproc_per_node=8 -m aorta.race --same-stream

Environment variables to test:
    GPU_MAX_HW_QUEUES=4          # Full parallelism (use 2 to reduce)
    ROC_AQL_QUEUE_SIZE=1024      # AQL queue depth (16384 default on AMD)
    ROC_SIGNAL_POOL_SIZE=16384   # HSA signal pool size
    HSA_ENABLE_SDMA=0            # Disable SDMA engine
    GPU_FORCE_BLIT_COPY_SIZE=128 # Force blit copy threshold
    NCCL_LAUNCH_ORDER_IMPLICIT=1 # Serialize NCCL operations (pair with RCCL_ENABLE_CONTEXT_TRACKING=1)
    RCCL_GFX9_CHEAP_FENCE_OFF=1  # Disable fence optimization
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

from aorta.race.config import ReproducerConfig, ReproducerResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="RCCL race condition reproducer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Config file
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file. CLI args override YAML values."
    )

    # Mode selection
    parser.add_argument(
        "--mode", type=str, default="default",
        choices=["default", "ddp", "fsdp", "eval_pipelined"],
        help=(
            "Workload mode. "
            "default: TorchRec-like (H2D + all_to_all + all_reduce). "
            "ddp: DDP (H2D prefetch + gradient all_reduce). "
            "fsdp: FSDP (per-layer all_gather + reduce_scatter). "
            "eval_pipelined: Pipelined eval loop (Experiments A/B). "
            "Default: default"
        ),
    )

    # Iteration settings
    parser.add_argument(
        "--warmup", type=int, default=None,
        help="Number of warmup iterations (no verification). Default: 100"
    )
    parser.add_argument(
        "--verify", type=int, default=None,
        help="Number of verification iterations. Default: 10000"
    )
    parser.add_argument(
        "--no-stop-on-first", action="store_true",
        help="Don't stop on first corruption (continue counting)"
    )
    parser.add_argument(
        "--log-interval", type=int, default=None,
        help="Log progress every N iterations. Default: 100"
    )

    # Tensor sizes
    parser.add_argument(
        "--h2d-size", type=int, default=None,
        help="H2D tensor size (elements). Default: 1M"
    )
    parser.add_argument(
        "--a2a-size", type=int, default=None,
        help="all_to_all tensor size per rank (elements). Default: 100K"
    )
    parser.add_argument(
        "--ar-size", type=int, default=None,
        help="all_reduce tensor size (elements). Default: 100K"
    )
    parser.add_argument(
        "--dtype", type=str, default=None,
        choices=["bfloat16", "float16", "float32"],
        help="Data type. Default: bfloat16"
    )

    # Compute simulation (enabled by default, use --no-compute to disable)
    parser.add_argument(
        "--no-compute", action="store_true",
        help="Disable compute simulation (fast but may not trigger bug)"
    )
    parser.add_argument(
        "--gemm-size", type=int, default=None,
        help="GEMM matrix size. Default: 5120 (~500ms/step on MI300X)"
    )
    parser.add_argument(
        "--gemm-layers", type=int, default=None,
        help="Number of GEMM layers. Default: 26 (~500ms/step on MI300X)"
    )
    parser.add_argument(
        "--no-backward", action="store_true",
        help="Skip backward pass simulation"
    )

    # FSDP-specific
    parser.add_argument(
        "--fsdp-shard-size", type=int, default=None,
        help="FSDP shard size per rank (elements). Default: 100K"
    )

    # H2D buffering strategy
    parser.add_argument(
        "--prefetch", action="store_true",
        help="Use double-buffered H2D prefetch (overlap next batch with backward)"
    )

    # Stream configuration
    parser.add_argument(
        "--same-stream", action="store_true",
        help="Put H2D and datadist on same stream (definitive runtime bug test)"
    )

    # Optimizer (for modes that support it)
    parser.add_argument(
        "--optimizer", type=str, default=None,
        choices=["none", "adamw", "sgd", "shampoo"],
        help="Optimizer for weight updates (used by DDP mode). Default: none"
    )

    # DDP-specific
    parser.add_argument(
        "--deterministic", action="store_true",
        help="Enable deterministic mode (required for DDP gradient verification)"
    )
    parser.add_argument(
        "--bucketed", action="store_true",
        help="Use bucketed per-layer gradient all_reduce overlapping with backward (DDP mode)"
    )

    # Hardware settings (GPU_MAX_HW_QUEUES)
    parser.add_argument(
        "--hw-queues", type=int, default=None,
        help="Set GPU_MAX_HW_QUEUES (4 exposes bug, 2 masks it)"
    )

    # ==========================================================================
    # Environment variable flags
    # ==========================================================================

    parser.add_argument(
        "--signal-pool-size", type=int, default=None,
        help="Set ROC_SIGNAL_POOL_SIZE (default 64)"
    )
    parser.add_argument(
        "--disable-sdma", action="store_true",
        help="Set HSA_ENABLE_SDMA=0 (disable SDMA engine)"
    )
    parser.add_argument(
        "--blit-copy-size", type=int, default=None,
        help="Set GPU_FORCE_BLIT_COPY_SIZE threshold"
    )
    parser.add_argument(
        "--nccl-implicit-order", action="store_true",
        help="Set NCCL_LAUNCH_ORDER_IMPLICIT=1 + RCCL_ENABLE_CONTEXT_TRACKING=1 (serializes NCCL ops)"
    )
    parser.add_argument(
        "--disable-cheap-fence", action="store_true",
        help="Set RCCL_GFX9_CHEAP_FENCE_OFF=1 and RCCL_GFX942_CHEAP_FENCE_OFF=1"
    )
    parser.add_argument(
        "--disable-clr-batch", action="store_true",
        help="Set DEBUG_CLR_BATCH_CPU_SYNC_SIZE=0 (disable CLR batching)"
    )
    parser.add_argument(
        "--aql-queue-size", type=int, default=None,
        help="Set ROC_AQL_QUEUE_SIZE (default 16384 on AMD, 1024 mitigates Experiment A)"
    )

    # ==========================================================================
    # Eval pipelined mode arguments
    # ==========================================================================

    eval_group = parser.add_argument_group(
        "eval_pipelined mode",
        "Arguments for --mode eval_pipelined (pipelined eval NaN investigation)"
    )
    eval_group.add_argument(
        "--batch-size", type=int, default=None,
        help="Batch size for eval model input. Default: 512"
    )
    eval_group.add_argument(
        "--feature-dim", type=int, default=None,
        help="Input feature dimension. Default: 256"
    )
    eval_group.add_argument(
        "--hidden-dim", type=int, default=None,
        help="Hidden dimension for MLP layers. Default: 1024"
    )
    eval_group.add_argument(
        "--model-layers", type=int, default=None,
        help="Number of hidden MLP layers. Default: 4"
    )
    eval_group.add_argument(
        "--model-type", type=str, default=None,
        choices=["mlp", "dlrm", "dlrm_v3"],
        help="Model type: mlp (simple MLP), dlrm (TorchRec-like DLRM), or dlrm_v3 (HSTU attention model). Default: mlp"
    )
    eval_group.add_argument(
        "--num-embedding-tables", type=int, default=None,
        help="Number of embedding tables (DLRM). Default: 64"
    )
    eval_group.add_argument(
        "--embedding-rows", type=int, default=None,
        help="Rows per embedding table (DLRM). Default: 100000"
    )
    eval_group.add_argument(
        "--embedding-dim", type=int, default=None,
        help="Embedding dimension (DLRM). Default: 128"
    )
    eval_group.add_argument(
        "--sparse-pooling-factor", type=int, default=None,
        help="Sparse lookups per sample per table (DLRM). Default: 20"
    )
    eval_group.add_argument(
        "--over-arch-layers", type=int, default=None,
        help="Over-arch MLP layers (DLRM). Default: 5"
    )
    eval_group.add_argument(
        "--use-compile", action="store_true", default=None,
        help="Apply torch.compile to the eval model"
    )
    eval_group.add_argument(
        "--no-compile", action="store_true",
        help="Disable torch.compile (for hypothesis B1 testing)"
    )
    eval_group.add_argument(
        "--disable-pipelining", action="store_true",
        help="Disable pipelined prefetch (each iteration independent)"
    )
    eval_group.add_argument(
        "--disable-datadist", action="store_true",
        help="Disable datadist stream (all work on default stream)"
    )
    eval_group.add_argument(
        "--disable-metrics", action="store_true",
        help="Disable metric simulation"
    )
    eval_group.add_argument(
        "--sync-policy", type=str, default=None,
        choices=["none", "end_only", "periodic", "every_iter", "all_pipeline_points"],
        help=(
            "Inter-iteration sync policy. "
            "none: zero sync (Experiment A). "
            "end_only: sync only at end. "
            "periodic: sync every --nan-check-interval iters. "
            "every_iter: sync each iteration. "
            "all_pipeline_points: sync at every stream point (Experiment B). "
            "Default: end_only"
        ),
    )
    eval_group.add_argument(
        "--nan-check-interval", type=int, default=None,
        help="NaN check interval for periodic sync policy. Default: 50"
    )
    eval_group.add_argument(
        "--embed-tensor-size", type=int, default=None,
        help="Size of embedding tensors for datadist. Default: 500000"
    )
    eval_group.add_argument(
        "--p2p-tensor-size", type=int, default=None,
        help="Size of point-to-point tensors for datadist. Default: 100000"
    )
    eval_group.add_argument(
        "--fresh-buffers", action="store_true",
        help="Allocate fresh GPU buffers each iteration (hypothesis B2/B3 test)"
    )
    eval_group.add_argument(
        "--gpu-padding-dispatches", type=int, default=None,
        help="Extra dispatches per iteration to inflate AQL queue fill rate. Default: 0"
    )
    eval_group.add_argument(
        "--no-pre-generate", action="store_true",
        help="Disable pre-generation of CPU data (generate on-the-fly)"
    )
    eval_group.add_argument(
        "--seq-len", type=int, default=None,
        help="Sequence length for dlrm_v3 attention. Controls GPU work. Default: 200"
    )
    eval_group.add_argument(
        "--use-bfloat16", action="store_true", default=None,
        help="Run dense forward in bfloat16 autocast (matches production precision)"
    )
    eval_group.add_argument(
        "--pre-generate-pool-size", type=int, default=None,
        help="Number of CPU batches to pre-generate and cycle through. Default: auto"
    )
    eval_group.add_argument(
        "--skip-lag-diagnostics", action="store_true",
        help="Skip post-loop CPU-GPU lag diagnostics (avoids NCCL timeouts with sync_policy=none)"
    )
    eval_group.add_argument(
        "--profile", action="store_true",
        help="Enable torch.profiler tracing (generates Chrome trace JSON)"
    )
    eval_group.add_argument(
        "--profile-iterations", type=int, default=None,
        help="Number of iterations to profile. Default: 5"
    )
    eval_group.add_argument(
        "--profile-output-dir", type=str, default=None,
        help="Directory for profiler trace files. Default: traces"
    )

    # CCA cross-stream allocation (CSAN race reproduction)
    cca_group = parser.add_argument_group(
        "CCA cross-stream allocation",
        "Reproduce the CCA event race detected by CSAN in TorchRec pipelines"
    )
    cca_group.add_argument(
        "--cca-cross-stream-alloc", action="store_true", default=None,
        help="Enable dynamic cross-stream tensor allocation (triggers CCA recycling race)"
    )
    cca_group.add_argument(
        "--no-cca-record-stream", action="store_true",
        help="Skip record_stream() for cross-stream tensors (reproduces missing-record_stream bug)"
    )
    cca_group.add_argument(
        "--cca-record-stream", action="store_true",
        help="Call record_stream() for cross-stream tensors (mitigation test)"
    )
    cca_group.add_argument(
        "--cca-num-pressure-tensors", type=int, default=None,
        help="Extra tensors to create/free on default stream each iter for CCA pressure. Default: 0"
    )
    cca_group.add_argument(
        "--cca-integrity-check", action="store_true",
        help="Enable GPU-side checksum verification of cross-stream tensors (detects corruption without NaN)"
    )
    cca_group.add_argument(
        "--cca-event-sync", action="store_true",
        help="Call event.synchronize() after event.query() success before CCA recycle (tests hipEventQuery ordering)"
    )

    # HSTU attention arguments (for dlrm_v3 model type)
    hstu_group = parser.add_argument_group(
        "dlrm_v3 / HSTU model",
        "Arguments for --model-type dlrm_v3 (HSTU attention model)"
    )
    hstu_group.add_argument(
        "--hstu-num-layers", type=int, default=None,
        help="Number of HSTU attention layers. Default: 5"
    )
    hstu_group.add_argument(
        "--hstu-num-heads", type=int, default=None,
        help="Number of attention heads. Default: 4"
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


def apply_env_vars(args: argparse.Namespace, yaml_cfg: dict) -> dict[str, str]:
    """Apply environment variables from CLI flags and YAML config.

    CLI flags take precedence over YAML values.  Must be called BEFORE
    any CUDA/HIP initialization so that driver-level knobs like
    ROC_AQL_QUEUE_SIZE take effect.

    Returns dict of variables that were set for logging.
    """
    applied = {}

    hwq = args.hw_queues
    if hwq is None:
        hwq = yaml_cfg.get("gpu_max_hw_queues")
    if hwq is not None:
        os.environ["GPU_MAX_HW_QUEUES"] = str(hwq)
        applied["GPU_MAX_HW_QUEUES"] = str(hwq)

    if args.signal_pool_size is not None:
        os.environ["ROC_SIGNAL_POOL_SIZE"] = str(args.signal_pool_size)
        applied["ROC_SIGNAL_POOL_SIZE"] = str(args.signal_pool_size)

    if args.disable_sdma:
        os.environ["HSA_ENABLE_SDMA"] = "0"
        applied["HSA_ENABLE_SDMA"] = "0"

    if args.blit_copy_size is not None:
        os.environ["GPU_FORCE_BLIT_COPY_SIZE"] = str(args.blit_copy_size)
        applied["GPU_FORCE_BLIT_COPY_SIZE"] = str(args.blit_copy_size)

    if args.nccl_implicit_order:
        os.environ["NCCL_LAUNCH_ORDER_IMPLICIT"] = "1"
        os.environ["RCCL_ENABLE_CONTEXT_TRACKING"] = "1"
        applied["NCCL_LAUNCH_ORDER_IMPLICIT"] = "1"
        applied["RCCL_ENABLE_CONTEXT_TRACKING"] = "1"

    if args.disable_cheap_fence:
        os.environ["RCCL_GFX9_CHEAP_FENCE_OFF"] = "1"
        os.environ["RCCL_GFX942_CHEAP_FENCE_OFF"] = "1"
        applied["RCCL_GFX9_CHEAP_FENCE_OFF"] = "1"
        applied["RCCL_GFX942_CHEAP_FENCE_OFF"] = "1"

    if args.disable_clr_batch:
        os.environ["DEBUG_CLR_BATCH_CPU_SYNC_SIZE"] = "0"
        applied["DEBUG_CLR_BATCH_CPU_SYNC_SIZE"] = "0"

    aql = getattr(args, "aql_queue_size", None)
    if aql is None:
        aql = yaml_cfg.get("aql_queue_size")
    if aql is not None:
        os.environ["ROC_AQL_QUEUE_SIZE"] = str(aql)
        applied["ROC_AQL_QUEUE_SIZE"] = str(aql)

    return applied


def run_with_mode(config: ReproducerConfig, rank: int, world_size: int) -> ReproducerResult:
    """
    Dispatch to the appropriate reproducer based on config.mode.

    Uses the modular system (base.py + modes/) via create_reproducer().
    """
    from aorta.race.modes import create_reproducer

    reproducer = create_reproducer(config, rank, world_size)
    return reproducer.run()


def _load_yaml_config(path: str) -> dict:
    """Load YAML config file."""
    from aorta.utils.config import load_config
    return load_config(Path(path))


def _build_config(args: argparse.Namespace, yaml_cfg: dict) -> ReproducerConfig:
    """Build ReproducerConfig from pre-loaded YAML dict + CLI args.

    YAML provides defaults; CLI args override when explicitly provided.
    The yaml_cfg dict is loaded early in main() (before CUDA init) so that
    env-var fields like aql_queue_size can be applied in time.
    """
    def _get(cli_val, yaml_key, default):
        if cli_val is not None:
            return cli_val
        return yaml_cfg.get(yaml_key, default)

    # CLI --mode wins over YAML "mode" wins over "default".
    # argparse defaults --mode to "default", so treat that as "not explicitly set".
    if args.mode != "default":
        mode = args.mode
    else:
        mode = yaml_cfg.get("mode", "default")

    config = ReproducerConfig(
        mode=mode,
        warmup_iterations=_get(args.warmup, "warmup_iterations", 100),
        verify_iterations=_get(args.verify, "verify_iterations", 10000),
        stop_on_first_corruption=(
            not args.no_stop_on_first
            if args.no_stop_on_first
            else yaml_cfg.get("stop_on_first_corruption", True)
        ),
        log_interval=_get(args.log_interval, "log_interval", 100),
        h2d_tensor_size=_get(args.h2d_size, "h2d_tensor_size", 1_000_000),
        alltoall_tensor_size=_get(args.a2a_size, "alltoall_tensor_size", 100_000),
        allreduce_tensor_size=_get(args.ar_size, "allreduce_tensor_size", 100_000),
        fsdp_shard_size=_get(args.fsdp_shard_size, "fsdp_shard_size", 100_000),
        dtype=_get(args.dtype, "dtype", "bfloat16"),
        simulate_compute=not args.no_compute if args.no_compute else yaml_cfg.get("simulate_compute", True),
        h2d_prefetch=args.prefetch or yaml_cfg.get("h2d_prefetch", False),
        gemm_size=_get(args.gemm_size, "gemm_size", 5120),
        gemm_layers=_get(args.gemm_layers, "gemm_layers", 26),
        include_backward_compute=not args.no_backward if args.no_backward else yaml_cfg.get("include_backward_compute", True),
        same_stream_mode=args.same_stream or yaml_cfg.get("same_stream_mode", False),
        gpu_max_hw_queues=_get(args.hw_queues, "gpu_max_hw_queues", 4),
        optimizer=_get(args.optimizer, "optimizer", "none"),
        deterministic=args.deterministic or yaml_cfg.get("deterministic", False),
        ddp_bucketed=args.bucketed or yaml_cfg.get("ddp_bucketed", False),
        # Eval pipelined mode fields
        batch_size=_get(args.batch_size, "batch_size", 512),
        feature_dim=_get(args.feature_dim, "feature_dim", 256),
        hidden_dim=_get(args.hidden_dim, "hidden_dim", 1024),
        model_layers=_get(args.model_layers, "model_layers", 4),
        use_compile=_resolve_compile(args, yaml_cfg),
        model_type=_get(getattr(args, "model_type", None), "model_type", "mlp"),
        num_embedding_tables=_get(getattr(args, "num_embedding_tables", None), "num_embedding_tables", 64),
        embedding_rows=_get(getattr(args, "embedding_rows", None), "embedding_rows", 100_000),
        embedding_dim=_get(getattr(args, "embedding_dim", None), "embedding_dim", 128),
        sparse_pooling_factor=_get(getattr(args, "sparse_pooling_factor", None), "sparse_pooling_factor", 20),
        over_arch_layers=_get(getattr(args, "over_arch_layers", None), "over_arch_layers", 5),
        enable_pipelining=not args.disable_pipelining if args.disable_pipelining else yaml_cfg.get("enable_pipelining", True),
        use_datadist_stream=not args.disable_datadist if args.disable_datadist else yaml_cfg.get("use_datadist_stream", True),
        simulate_metrics=not args.disable_metrics if args.disable_metrics else yaml_cfg.get("simulate_metrics", True),
        sync_policy=_get(args.sync_policy, "sync_policy", "end_only"),
        nan_check_interval=_get(args.nan_check_interval, "nan_check_interval", 50),
        embed_tensor_size=_get(args.embed_tensor_size, "embed_tensor_size", 500_000),
        p2p_tensor_size=_get(getattr(args, "p2p_tensor_size", None), "p2p_tensor_size", 100_000),
        fresh_buffers_each_iter=args.fresh_buffers or yaml_cfg.get("fresh_buffers_each_iter", False),
        gpu_padding_dispatches=_get(args.gpu_padding_dispatches, "gpu_padding_dispatches", 0),
        pre_generate_data=not args.no_pre_generate if args.no_pre_generate else yaml_cfg.get("pre_generate_data", True),
        seq_len=_get(getattr(args, "seq_len", None), "seq_len", 200),
        use_bfloat16=(
            args.use_bfloat16 if getattr(args, "use_bfloat16", None)
            else yaml_cfg.get("use_bfloat16", False)
        ),
        pre_generate_pool_size=_get(getattr(args, "pre_generate_pool_size", None), "pre_generate_pool_size", None),
        skip_lag_diagnostics=getattr(args, "skip_lag_diagnostics", False) or yaml_cfg.get("skip_lag_diagnostics", False),
        profile=getattr(args, "profile", False) or yaml_cfg.get("profile", False),
        profile_iterations=_get(getattr(args, "profile_iterations", None), "profile_iterations", 5),
        profile_output_dir=_get(getattr(args, "profile_output_dir", None), "profile_output_dir", "traces"),
        aql_queue_size=_get(args.aql_queue_size, "aql_queue_size", None),
        # HSTU attention fields (dlrm_v3 model type)
        hstu_attn_num_layers=_get(getattr(args, "hstu_num_layers", None), "hstu_attn_num_layers", 5),
        hstu_num_heads=_get(getattr(args, "hstu_num_heads", None), "hstu_num_heads", 4),
        # CCA cross-stream allocation
        cca_cross_stream_alloc=_resolve_cca_alloc(args, yaml_cfg),
        cca_record_stream=_resolve_cca_record_stream(args, yaml_cfg),
        cca_num_pressure_tensors=_get(
            getattr(args, "cca_num_pressure_tensors", None),
            "cca_num_pressure_tensors", 0,
        ),
        cca_integrity_check=getattr(args, "cca_integrity_check", False) or yaml_cfg.get("cca_integrity_check", False),
        cca_event_sync=getattr(args, "cca_event_sync", False) or yaml_cfg.get("cca_event_sync", False),
    )
    return config


def _resolve_compile(args: argparse.Namespace, yaml_cfg: dict) -> bool:
    """Resolve --use-compile / --no-compile / YAML use_compile."""
    if args.no_compile:
        return False
    if args.use_compile:
        return True
    return yaml_cfg.get("use_compile", False)


def _resolve_cca_alloc(args: argparse.Namespace, yaml_cfg: dict) -> bool:
    """Resolve --cca-cross-stream-alloc / YAML cca_cross_stream_alloc."""
    if getattr(args, "cca_cross_stream_alloc", None):
        return True
    return yaml_cfg.get("cca_cross_stream_alloc", False)


def _resolve_cca_record_stream(args: argparse.Namespace, yaml_cfg: dict) -> bool:
    """Resolve --cca-record-stream / --no-cca-record-stream / YAML."""
    if getattr(args, "no_cca_record_stream", False):
        return False
    if getattr(args, "cca_record_stream", False):
        return True
    return yaml_cfg.get("cca_record_stream", True)


def main():
    """Main entry point."""
    args = parse_args()

    # Load YAML config EARLY -- before CUDA init so that env-var fields
    # (aql_queue_size, gpu_max_hw_queues, etc.) can be applied in time.
    yaml_cfg: dict = {}
    if args.config:
        yaml_cfg = _load_yaml_config(args.config)

    # Apply environment variables from CLI flags + YAML (before any CUDA init)
    applied_env = apply_env_vars(args, yaml_cfg)

    # Initialize distributed (triggers CUDA/HIP init)
    rank, world_size = init_distributed()

    # Build full config from pre-loaded YAML + CLI
    config = _build_config(args, yaml_cfg)

    if rank == 0:
        log.info("=" * 70)
        log.info("RCCL RACE CONDITION REPRODUCER")
        log.info("=" * 70)
        log.info("")
        if config.mode == "eval_pipelined":
            log.info("Mode: eval_pipelined (pipelined eval NaN investigation)")
            log.info(f"  model_type={config.model_type}")
            log.info(f"  batch_size={config.batch_size}, feature_dim={config.feature_dim}")
            log.info(f"  hidden_dim={config.hidden_dim}, model_layers={config.model_layers}")
            if config.model_type == "dlrm":
                log.info(f"  DLRM: tables={config.num_embedding_tables}, rows={config.embedding_rows}, "
                         f"emb_dim={config.embedding_dim}, pool={config.sparse_pooling_factor}, "
                         f"over_arch={config.over_arch_layers}")
            elif config.model_type == "dlrm_v3":
                log.info(f"  HSTU: seq_len={config.seq_len}, "
                         f"attn_layers={config.hstu_attn_num_layers}, "
                         f"heads={config.hstu_num_heads}, "
                         f"emb_dim={config.embedding_dim}, "
                         f"bfloat16={config.use_bfloat16}")
            log.info(f"  use_compile={config.use_compile}, pipelining={config.enable_pipelining}")
            log.info(f"  sync_policy={config.sync_policy}, metrics={config.simulate_metrics}")
            log.info(f"  datadist_stream={config.use_datadist_stream}")
            log.info(f"  fresh_buffers={config.fresh_buffers_each_iter}")
            log.info(f"  gpu_padding_dispatches={config.gpu_padding_dispatches}")
            if config.cca_cross_stream_alloc:
                log.info(f"  cca_cross_stream_alloc=True (dynamic side-stream allocation)")
                log.info(f"  cca_record_stream={config.cca_record_stream}")
                log.info(f"  cca_num_pressure_tensors={config.cca_num_pressure_tensors}")
        else:
            log.info("This reproducer uses PROPER SYNCHRONIZATION everywhere.")
            log.info("If corruption occurs, it indicates a RUNTIME BUG in RCCL/HIP,")
            log.info("not an application-level issue.")
        log.info("")
        log.info(f"Mode: {config.mode}")
        log.info(f"World size: {world_size}")
        log.info(f"Warmup iterations: {config.warmup_iterations}")
        log.info(f"Verify iterations: {config.verify_iterations}")
        if args.config:
            log.info(f"Config file: {args.config}")
        log.info("")

        env_vars = [
            "GPU_MAX_HW_QUEUES",
            "ROC_AQL_QUEUE_SIZE",
            "ROC_SIGNAL_POOL_SIZE",
            "HSA_ENABLE_SDMA",
            "GPU_FORCE_BLIT_COPY_SIZE",
            "NCCL_LAUNCH_ORDER_IMPLICIT",
            "RCCL_ENABLE_CONTEXT_TRACKING",
            "RCCL_GFX9_CHEAP_FENCE_OFF",
            "RCCL_GFX942_CHEAP_FENCE_OFF",
            "DEBUG_CLR_BATCH_CPU_SYNC_SIZE",
            "PYTORCH_NO_CUDA_MEMORY_CACHING",
        ]
        log.info("Environment variables:")
        for var in env_vars:
            value = os.environ.get(var, "(not set)")
            is_cli = var in applied_env
            from_yaml = is_cli and (
                (var == "ROC_AQL_QUEUE_SIZE" and getattr(args, "aql_queue_size", None) is None)
                or (var == "GPU_MAX_HW_QUEUES" and args.hw_queues is None)
            )
            source = " (via config)" if from_yaml else (" (via CLI)" if is_cli else "")
            log.info(f"  {var}={value}{source}")
        log.info("")
        # Version information
        log.info("Version information:")
        log.info(f"  PyTorch: {torch.__version__}")
        log.info(f"  HIP runtime: {getattr(torch.version, 'hip', 'N/A')}")
        log.info(f"  CUDA/HIP: {torch.version.cuda}")
        try:
            log.info(f"  GPU: {torch.cuda.get_device_name(0)}")
            log.info(f"  GPU arch: {torch.cuda.get_device_capability(0)}")
        except Exception:
            log.info("  GPU: (not available)")
        try:
            nccl_version = torch.cuda.nccl.version()
            log.info(f"  NCCL/RCCL: {nccl_version}")
        except Exception:
            log.info("  NCCL/RCCL: (not available)")
        log.info(f"  PYTORCH_CUDA_ALLOC_CONF: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '(not set)')}")
        log.info(f"  NCCL_MAX_NCHANNELS: {os.environ.get('NCCL_MAX_NCHANNELS', '(not set)')}")
        log.info(f"  HSA_KERNARG_POOL_SIZE: {os.environ.get('HSA_KERNARG_POOL_SIZE', '(not set)')}")
        log.info("")
        log.info("=" * 70)

    # Run reproducer via mode dispatch
    result = run_with_mode(config, rank, world_size)

    # Sync all ranks -- skip if the run may have corrupted NCCL state
    # (e.g., CCA cross-stream race with no record_stream).  A corrupted
    # NCCL communicator will deadlock on any collective, including barrier.
    if not (config.cca_cross_stream_alloc and not config.cca_record_stream):
        try:
            dist.barrier()
        except Exception:
            pass

    # Report results (rank 0 only)
    if rank == 0:
        log.info("")
        log.info("=" * 70)
        log.info("RESULTS")
        log.info("=" * 70)
        log.info(f"Mode: {config.mode}")
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
            log.info("If corruption still occurs in real workloads, check for")
            log.info("application-level synchronization issues.")
        else:
            log.info("")
            log.info("VERDICT: RUNTIME BUG DETECTED!")
            log.info("Corruption occurred DESPITE proper synchronization.")
            log.info("This is a bug in RCCL/HIP runtime - report to AMD.")

        log.info("=" * 70)

    # Tear down process group before exit to prevent the NCCL destructor
    # from hanging during Python shutdown (it attempts collective cleanup
    # that deadlocks if any rank has already exited).
    try:
        dist.destroy_process_group()
    except Exception:
        pass

    # Exit with appropriate code
    sys.exit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
