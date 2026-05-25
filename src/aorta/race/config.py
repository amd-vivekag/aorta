"""
Race experiment configuration dataclasses.

This module defines:
- ReproducerConfig: Settings for the standalone RCCL race condition reproducer
- ReproducerResult: Result from running the reproducer
- RaceConfig: Settings for race condition injection experiments (broader aorta system)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# =============================================================================
# Standalone Reproducer Config
# =============================================================================


@dataclass
class ReproducerConfig:
    """
    Configuration for the standalone RCCL race condition reproducer.

    This reproducer tests for RUNTIME bugs (not application bugs) by:
    - Using proper synchronization everywhere
    - Using known-pattern data to detect any corruption
    - Simulating training timing profile

    If corruption occurs with proper syncs, it's a RUNTIME BUG in RCCL/HIP.
    """

    # =========================================================================
    # Mode selection
    # =========================================================================
    mode: str = "default"
    """
    Reproducer mode. Determines the communication pattern tested.

    Available modes:
    - "default": TorchRec-like pattern (H2D + all_to_all + all_reduce)
    - "ddp": DDP pattern (H2D with double-buffered prefetch + gradient all_reduce)
    - "fsdp": FSDP pattern (per-layer all_gather + reduce_scatter)
    - "eval_pipelined": Pipelined eval loop for NaN investigation (Experiments A/B)
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
    Used by default mode only.
    """

    allreduce_tensor_size: int = 100_000
    """
    Size of all_reduce tensor (number of elements).

    Simulates gradient synchronization (e.g., FSDP-style).
    Used by default mode only; DDP mode all-reduces actual gradients.
    """

    fsdp_shard_size: int = 100_000
    """
    Size of each FSDP parameter shard per rank (number of elements).

    Used by FSDP mode for per-layer all_gather/reduce_scatter.
    Full parameter size = fsdp_shard_size * world_size.
    Larger = longer collectives = more overlap opportunity.
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

    compute_type: str = "gemm"
    """Compute pattern type. Options: gemm (more can be registered)."""

    gemm_size: int = 5120
    """
    Matrix size for GEMM operations.

    5120x5120 GEMM takes ~14ms on MI300X.
    Adjust based on GPU to achieve ~500ms/step target.
    """

    gemm_layers: int = 26
    """
    Number of GEMM layers to simulate forward/backward.

    26 layers x 14ms = ~364ms per forward/backward pass.
    """

    include_backward_compute: bool = True
    """Also simulate backward pass GEMMs (doubles compute time)."""

    # =========================================================================
    # Optimizer (used by modes that support it, e.g., DDP)
    # =========================================================================
    optimizer: str = "none"
    """
    Optimizer for weight updates. Options: none, adamw, sgd, shampoo.

    When set to 'none', no optimizer step is performed.
    DDP mode uses this to test gradient all_reduce + optimizer interaction.
    """

    optimizer_lr: float = 1e-4
    """Learning rate."""

    optimizer_weight_decay: float = 0.01
    """Weight decay."""

    optimizer_betas: Tuple[float, float] = (0.9, 0.999)
    """Adam betas."""

    optimizer_eps: float = 1e-8
    """Adam epsilon."""

    # =========================================================================
    # Deterministic mode (used by DDP for cross-rank gradient verification)
    # =========================================================================
    deterministic: bool = False
    """
    Enable deterministic mode with fixed seeds.

    Required for DDP gradient consistency verification across ranks.
    """

    deterministic_seed: int = 42
    """Seed for deterministic mode."""

    ddp_bucketed: bool = False
    """
    Use bucketed gradient all_reduce overlapping with backward.

    When enabled, per-layer gradient all_reduce runs concurrently with
    backward computation of earlier layers (real DDP behavior).
    When disabled, one bulk all_reduce runs after all of backward finishes.

    Only used by DDP mode.
    """

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

    h2d_prefetch: bool = False
    """
    Use double-buffered H2D with prefetch.

    When enabled, the next batch is copied to GPU during the current iteration's
    backward pass, overlapping H2D with compute. When disabled, H2D is a blocking
    transfer at the start of each iteration.

    Double-buffered is the standard pattern in real DDP/FSDP training pipelines.
    Single-buffered is simpler and tests a different timing profile.
    """

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

    # =========================================================================
    # Eval pipelined mode (eval_pipelined)
    # =========================================================================
    batch_size: int = 512
    """Batch size (first dimension of model input). Affects kernel parameters."""

    feature_dim: int = 256
    """Input feature dimension for the eval model."""

    hidden_dim: int = 1024
    """Hidden dimension for the eval model MLP layers."""

    model_layers: int = 4
    """Number of hidden layers in the eval model."""

    use_compile: bool = False
    """
    Apply torch.compile to the eval model.

    When enabled, the forward pass is compiled (CompiledFullGraph pattern).
    Different batch sizes produce different compiled kernels.
    """

    # TorchRec-like DLRM model settings
    model_type: str = "mlp"
    """
    Model type for eval_pipelined mode.

    - "mlp": Simple MLP (fast, for quick tests). Memory-BW bound, ~0.1ms/iter.
    - "dlrm": TorchRec-style DLRM (embedding tables + bottom/over-arch MLPs).
              Memory-BW bound, ~0.5ms/iter.
    - "dlrm_v3": DLRMv3-inspired model with HSTU-style causal attention on
                 configurable-length sequences. Compute-bound via O(seq_len^2)
                 attention -- the only model heavy enough to create CPU-GPU lag
                 for Experiment A NaN reproduction. No GPU-side embedding tables;
                 sparse lookups happen on CPU and results are transferred.
    - "ig3_rec_proxy": IG3 TraceLens proxy -- full DLRM-style recommendation
                 model from pruned_trace_ig3. Dense bottom MLP (512->256->128),
                 50 EmbeddingBag tables (128 dim), multi-head BMM interaction
                 (32 heads), top MLP (1024->2048->4096->2048->1024), and
                 multi-task output heads (36 BCE + 10 MSE). Architecture
                 constants are trace-derived; num_embedding_tables (capped at
                 50), embedding_rows, embedding_dim, and sparse_pooling_factor
                 are configurable.
    """

    num_embedding_tables: int = 64
    """Number of embedding tables (DLRM model). Production TorchRec uses 50-200."""

    embedding_rows: int = 100_000
    """Rows per embedding table. Production uses 1K-10M per table."""

    embedding_dim: int = 128
    """Embedding dimension per table. Production uses 32-256."""

    sparse_pooling_factor: int = 20
    """Average number of sparse feature lookups per sample per table."""

    over_arch_layers: int = 5
    """Number of over-arch MLP layers (DLRM model)."""

    # HSTU attention settings (dlrm_v3 model type)
    hstu_num_heads: int = 4
    """Number of attention heads in HSTU layers (dlrm_v3 model type)."""

    hstu_attn_num_layers: int = 5
    """Number of HSTU attention layers (dlrm_v3 model type)."""

    seq_len: int = 200
    """
    Sequence length for attention in dlrm_v3 model.

    Controls GPU forward pass duration via O(seq_len^2) attention cost.
    Longer sequences = heavier GPU work = more CPU-GPU lag.
    User interaction history sequences in production are 200-16K tokens.

    Approximate GPU time per forward (MI300X, 5 attn layers, embed_dim=512, bs=512):
      seq_len=17:   ~0.5ms  (GPU keeps up, no NaN)
      seq_len=100:  ~5ms    (GPU starts falling behind)
      seq_len=200:  ~15ms   (CPU 2-3 iters ahead -- Experiment A range)
      seq_len=500:  ~80ms   (deep AQL fill)
    """

    use_bfloat16: bool = False
    """
    Run dense forward pass in bfloat16 autocast.

    Matches production precision. bfloat16 has 7 mantissa bits
    vs 23 in fp32, making NaN more likely from small data corruptions.
    """

    pre_generate_pool_size: Optional[int] = None
    """
    Number of CPU batches to pre-generate and cycle through.

    None = auto: pre-generates all iterations for mlp/dlrm (small data),
    or 20 batches for dlrm_v3 (large seq_embeddings data).
    For dlrm_v3 at seq_len=200, bs=512, feature_dim=256: each batch is
    ~26MB, so 20 batches = ~520MB pinned CPU memory.
    """

    enable_pipelining: bool = True
    """
    Enable pipelined prefetch (double-buffered).

    When enabled, iteration N+1's data is prefetched on side streams while
    iteration N's compute runs on the default stream. Disabling makes each
    iteration fully independent (no cross-iteration buffer sharing).
    """

    use_datadist_stream: bool = True
    """
    Use a separate datadist stream for data distribution collectives.

    When disabled, all work runs on the default stream (serialized).
    Automatically disabled for single-GPU runs.
    """

    simulate_metrics: bool = True
    """
    Simulate metric computation (NE, MAE, calibration) on the default stream
    after the forward pass, matching the eval pipeline's update_metrics and
    update_reg_metrics stages.
    """

    use_ddp_wrapper: bool = True
    """Wrap the model in DistributedDataParallel (when world_size > 1)."""

    sync_policy: str = "end_only"
    """
    Inter-iteration CPU-GPU synchronization policy.

    - "none": Zero sync in the loop. CPU races ahead freely.
    - "end_only": Sync only after all iterations complete.
    - "periodic": Sync every nan_check_interval iterations.
    - "every_iter": Sync after each iteration.
    - "all_pipeline_points": Sync at every stream interaction point
      (drains AQL queue to zero; for Experiment B).
    """

    nan_check_interval: int = 50
    """
    Check for NaN every N iterations (when sync_policy is 'periodic').

    Lower values detect NaN iteration more precisely but add more sync
    points that may reduce the CPU-GPU lag needed for Experiment A.
    """

    embed_tensor_size: int = 500_000
    """Size of embedding tensors for datadist reduce_scatter."""

    p2p_tensor_size: int = 100_000
    """Size of tensors for point-to-point send/recv in datadist."""

    fresh_buffers_each_iter: bool = False
    """
    Allocate fresh GPU buffers every iteration (no address reuse).

    For Experiment B hypothesis testing: if NaN disappears with fresh
    buffers, the bug is related to buffer address reuse (cache staleness
    or allocator recycling).
    """

    gpu_padding_dispatches: int = 0
    """
    Extra no-op kernel dispatches per iteration to inflate the AQL queue
    fill rate. Helps ensure the CPU races ahead of the GPU.
    """

    pre_generate_data: bool = True
    """
    Pre-generate all CPU batch data before the run loop.

    Minimizes CPU-side overhead per iteration so the CPU can submit
    dispatches as fast as possible (maximizes CPU-GPU lag).
    """

    profile: bool = False
    """
    Enable torch.profiler tracing. Generates a Chrome trace JSON file
    that can be viewed in chrome://tracing or Perfetto UI.
    Profiles a window of iterations after warmup.
    """

    profile_iterations: int = 5
    """Number of iterations to profile (after warmup)."""

    profile_output_dir: str = "traces"
    """Directory to write profiler trace files."""

    aql_queue_size: Optional[int] = None
    """
    Set ROC_AQL_QUEUE_SIZE environment variable (before CUDA init).

    Controls the AQL hardware queue depth on AMD GPUs:
    - None: use system default (16384 on AMD)
    - 1024: matches NVIDIA queue depth, mitigates Experiment A
    - 512: aggressive backpressure

    Stored in config so YAML presets are fully self-contained.
    Applied before CUDA/HIP initialization.
    """

    # =========================================================================
    # CCA cross-stream allocation (CSAN race reproduction)
    # =========================================================================
    cca_cross_stream_alloc: bool = False
    """
    Enable dynamic cross-stream tensor allocation to reproduce the CCA
    event race detected by CSAN in TorchRec pipelines.

    When enabled, pipeline buffers (datadist shards, H2D batches, seq
    embeddings) are allocated dynamically each iteration via torch.empty()
    on their respective side streams, instead of being pre-allocated at
    setup.  Old buffers are freed after being used on the default stream,
    triggering CCA's event-based cross-stream recycling.

    Without record_stream(), CCA only tracks the allocation stream.  When
    the side stream's event completes (its reduce_scatter / copy_ is done),
    CCA marks the block free -- even though the default stream may still be
    reading it during forward.  A subsequent torch.empty() on the side
    stream recycles the block, causing the side stream to overwrite data the
    default stream is reading.

    This matches the TorchRec pattern where KJTAllToAllTensorsAwaitable
    calls torch.empty() on data_dist_stream while alltoall_base_ on the
    default stream still reads from the same memory block (the exact race
    CSAN detected).
    """

    cca_record_stream: bool = True
    """
    Call record_stream() when tensors cross from side streams to the
    default stream.

    When True (default), after wait_stream() and before forward reads the
    tensor, record_stream(default_stream) is called.  This tells CCA that
    the default stream also uses this block, so CCA records events on both
    the allocation stream AND the default stream.  The block is not recycled
    until both events complete -- which prevents the race.

    When False, CCA only knows about the allocation stream.  As soon as
    the side stream's event completes, CCA marks the block free, even if
    forward on the default stream is still reading it.  This is the
    condition that produces NaN.

    Use True to confirm that record_stream is the fix.
    Use False to reproduce the CSAN race and trigger NaN.
    """

    cca_num_pressure_tensors: int = 0
    """
    Number of additional "pressure" tensors to create and free on the
    default stream each iteration.  Each tensor is the same size as
    the datadist shard to maximize the chance CCA recycles them when
    the side stream calls torch.empty().

    Increases the rate of CCA cross-stream recycling, simulating TorchRec's
    intermediate tensor creation pattern.  Set to 4-8 for additional
    pressure; 0 disables (only pipeline buffers are dynamic).
    """

    skip_lag_diagnostics: bool = False
    """
    Skip post-loop CPU-GPU lag diagnostics (calibration iterations,
    dispatch profiling, and AQL queue analysis).

    The diagnostics run 13 extra iterations with collectives after the
    main loop (10 calibration + 3 profiled).  With sync_policy=none,
    these can trigger NCCL watchdog timeouts because the GPU is still
    draining hundreds of queued collectives from the main loop when
    the diagnostics try to run more.

    Set to True when running with sync_policy=none on slow models or
    when hitting NCCL timeouts after the main loop completes.
    """

    cca_integrity_check: bool = False
    """
    Enable GPU-side data integrity verification for cross-stream tensors.

    After the datadist_stream writes embed_shard (via reduce_scatter), a
    checksum is computed ON the datadist_stream and stored.  After the
    default_stream's wait_stream(), the checksum is re-computed on the
    default_stream.  If the checksums differ, CCA recycled the memory
    block between write and read -- direct proof of the race.

    This detects corruption WITHOUT relying on NaN (which requires the
    forward pass to amplify small errors into overflow).  Works with
    any model type.  Zero CPU-GPU sync in the loop.
    """

    cca_event_sync: bool = False
    """
    After hipEventQuery returns success, call hipEventSynchronize before
    allowing CCA to recycle blocks.  Tests whether hipEventQuery's
    memory ordering is too weak (premature hipEventQuery hypothesis).

    When enabled, at _swap_buffers() time, the old tensors' events are
    explicitly synchronized before dropping references.  If this fixes
    the integrity check failures, it proves hipEventQuery returns
    success before memory writes are globally visible.
    """


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


# =============================================================================
# Race Injection Config (broader aorta system)
# =============================================================================


@dataclass
class RaceConfig:
    """
    Configuration for race condition injection experiments.

    Four categories of race conditions are supported:

    1. H2D Race (realistic pattern):
       - h2d_memcpy_racing: Uses separate stream for H2D batch copy
       - h2d_skip_sync_before_forward: Skips wait_stream() before forward (causes race!)
       - h2d_racing_start_step: Step to start H2D racing

    2. Datadist Race (TorchRec-style all_to_all):
       - datadist_racing: Uses separate stream for all_to_all operations
       - datadist_skip_sync_before_collective: Skips wait_stream() before FSDP collective (causes race!)
       - datadist_racing_start_step: Step to start datadist racing

    3. Timing Skew Experiment (demonstrates NaN progression):
       - timing_skew_enabled: Enable controlled timing skew
       - timing_skew_mode: none, fixed, progressive, random
       - timing_skew_us: Delay in microseconds
       - timing_skew_ranks: Which ranks get delayed
       - timing_skew_start_step: Step to start skew

    Supporting options:
       - skip_training_warmup: Skip training warmup to maximize timing variability
       - skip_rccl_warmup: Skip RCCL communicator warmup before FSDP init
       - nan_check_collectives: Enable NaN checking around RCCL collectives
       - gpu_max_hw_queues: Set GPU_MAX_HW_QUEUES (4+ needed to expose race)
    """

    # =========================================================================
    # H2D memcpy racing (realistic pattern)
    # =========================================================================
    h2d_memcpy_racing: bool = False
    """Use separate memcpy_stream for H2D batch copy."""

    h2d_skip_sync_before_forward: bool = False
    """Skip wait_stream() before forward pass - THIS CAUSES THE RACE!"""

    h2d_racing_start_step: int = 0
    """Step to start H2D racing (0 = aggressive, from first step)."""

    # =========================================================================
    # Datadist racing (TorchRec-style all_to_all on separate stream)
    # =========================================================================
    datadist_racing: bool = False
    """Use separate datadist_stream for all_to_all operations."""

    datadist_skip_sync_before_collective: bool = False
    """Skip wait_stream() before FSDP collective - THIS CAUSES THE RACE!"""

    datadist_racing_start_step: int = 0
    """Step to start datadist racing (0 = aggressive, from first step)."""

    # =========================================================================
    # Supporting options
    # =========================================================================
    skip_training_warmup: bool = False
    """Skip training warmup to maximize timing variability for race testing."""

    training_warmup_steps: int = 1
    """Number of training warmup steps to run (if not skipped)."""

    warmup_batch_size: Optional[int] = None
    """
    Batch size for warmup steps. If None, uses the training batch_size.
    Set this smaller than training batch_size to speed up warmup while
    still exercising the collectives, then use larger batch during racing
    for wider race windows.
    """

    skip_rccl_warmup: bool = False
    """Skip RCCL communicator warmup before FSDP init to test race conditions."""

    rccl_warmup_iterations: int = 10
    """Number of RCCL warmup iterations (if not skipped). Higher = more stable but slower startup."""

    nan_check_collectives: bool = False
    """Enable NaN checking before/after RCCL collectives."""

    gpu_max_hw_queues: Optional[int] = None
    """
    Set GPU_MAX_HW_QUEUES environment variable.

    CRITICAL for race exposure:
    - 1-2: Streams share HW queues → implicit serialization → RACE MASKED
    - 4+: Each stream gets own HW queue → true parallelism → RACE EXPOSED

    Recommended: 4 for race testing, or 3 with production_stream_layout.
    """

    client_stream_layout: bool = False
    """
    Use production stream layout for accurate reproduction of the NaN issue.

    When enabled:
    - 3 streams only: default_stream, memcpy_stream, datadist_stream
    - Forward/backward/clip/optimizer/FSDP collectives run on default stream
    - Only H2D and datadist operations on separate racing streams
    - DistributedOpsInterceptor is bypassed (no stream redirection for collectives)

    This matches the production TorchRec-style architecture where:
    - memcpy_stream races with default stream (H2D not synced before forward)
    - datadist_stream races with default stream (all_to_all not synced before collective)

    Recommended: Use with gpu_max_hw_queues: 3 for 1:1 stream-to-queue mapping.
    """

    # =========================================================================
    # Timing skew experiment (demonstrates NaN progression)
    # =========================================================================
    timing_skew_enabled: bool = False
    """Enable controlled timing skew experiment to show NaN progression."""

    timing_skew_mode: str = "none"
    """
    Skew mode:
    - none: No artificial skew
    - fixed: Fixed delay in microseconds
    - progressive: Increase delay each step (skew_us * step)
    - random: Random delay within range
    """

    timing_skew_us: int = 0
    """Base delay in microseconds (for fixed/progressive modes)."""

    timing_skew_ranks: List[int] = field(default_factory=list)
    """Which ranks get delayed. Empty = all ranks."""

    timing_skew_start_step: int = 3
    """Step to start introducing timing skew (after warmup)."""

    # =========================================================================
    # Debug instrumentation
    # =========================================================================
    timing_debug_logs: bool = False
    """
    Enable detailed timing logs around H2D, datadist, and forward operations.

    When enabled, logs wall-clock timestamps for:
    - H2D copy (memcpy_stream operations)
    - Datadist racing (all_to_all operations)
    - Timing skew delay
    - Forward pass

    Also logs gap/overlap between operations to verify if race windows exist.
    Negative gap = overlap = potential race.
    """

    gpu_event_timing: bool = False
    """
    Enable GPU event-based timing to measure actual stream overlap.

    Unlike timing_debug_logs which measures CPU-side timestamps, this uses
    CUDA/HIP events to measure actual GPU execution times and overlap.

    Event recording is non-blocking and doesn't interfere with race conditions.
    Timing calculations happen AFTER the iteration completes (when we sync anyway).

    Logs GPU-side durations and overlap between streams:
    - GPU_H2D_DUR: Actual GPU time for H2D copy
    - GPU_DD_DUR: Actual GPU time for datadist all_to_all
    - GPU_FWD_DUR: Actual GPU time for forward pass
    - GPU_OVERLAP: Time between stream operations (negative = overlap = race!)

    WARNING: Cross-stream timing (e.g., DD->FWD) is INACCURATE because events
    are on different streams. Use nccl_async_diagnostic for accurate overlap detection.
    """

    nccl_async_diagnostic: bool = False
    """
    Enable NCCL async behavior diagnostic.

    When enabled, checks if the datadist all_to_all work is still in-flight
    when forward pass starts. This uses work.is_completed() which is non-blocking
    and does NOT affect the race condition.

    Logs:
    - NCCL_DIAG: Whether all_to_all is PENDING (async, race possible) or
      COMPLETED (sync, race may be masked by NCCL internal sync)

    This is the authoritative way to verify if a race window exists, because
    it directly queries the NCCL work handle status rather than relying on
    GPU event timing which is inaccurate for cross-stream measurements.
    """

    datadist_use_real_dependency: bool = True
    """
    Make datadist output actually used by forward pass.

    When enabled, the all_to_all output tensor is added as noise to batch["dense"],
    creating a real data dependency that forward must read. This more accurately
    reproduces the TorchRec pattern where distributed embeddings race
    with the forward pass.

    Without this, datadist creates synthetic tensors that are discarded,
    so there's no actual data race with forward (just stream contention).
    """

    # =========================================================================
    # Controlled in-flight reads (no NaN poisoning)
    # =========================================================================
    h2d_split_dense_copy: bool = False
    """
    Copy the dense tensor in two chunks on the memcpy stream.

    The tail chunk is copied last to increase the chance it is still in-flight
    when forward starts. This helps ensure the forward reads a tensor that is
    actively being copied without injecting NaNs.
    """

    h2d_dense_tail_fraction: float = 0.5
    """
    Fraction of dense elements copied in the tail chunk (copied last).
    Clamped to (0, 1) and adjusted to keep at least one element in each chunk.
    """

    datadist_split_alltoall: bool = False
    """
    Split the all_to_all into two sequential collectives on the datadist stream.

    The tail chunk collective is launched last, so reading that tail immediately
    on the default stream is guaranteed to be in-flight.
    """

    datadist_tail_fraction: float = 0.5
    """
    Fraction of all_to_all elements assigned to the tail chunk (launched last).
    Clamped to (0, 1) and adjusted to be divisible by world_size.
    """

    datadist_read_tail_only: bool = False
    """
    Use only the tail chunk of the all_to_all output when creating the
    datadist->forward dependency. This guarantees the read targets the
    in-flight chunk without NaN poisoning.
    """

    # =========================================================================
    # In-flight read instability checks
    # =========================================================================
    inflight_read_check_enabled: bool = False
    """
    Enable repeated in-flight reads to detect instability/mismatch.

    When enabled, the tail region of in-flight tensors is read multiple times
    on the default stream while the racing stream is still writing. If the
    values change between reads, it indicates a torn read (race condition).
    """

    inflight_read_repeats: int = 0
    """
    Number of repeated reads to perform on the in-flight tail region.

    Set to 0 to disable repeated reads. Higher values increase the chance
    of detecting instability but add more GPU work on the default stream.
    Recommended: 3-10 for typical detection.
    """

    inflight_read_sample_size: int = 4096
    """
    Number of elements to sample from the tail region for instability checks.

    A smaller sample reduces GPU work; a larger sample increases detection
    probability. The sample is taken from the start of the tail region.
    """

    inflight_read_delay_size: int = 65536
    """
    Size of dummy tensor used for GPU-side delay work between reads.

    Between each repeated read, we issue GPU work (D2D copy + reduction) to
    spread reads across the race window. Larger values = more delay = higher
    detection probability but more GPU overhead.

    Recommended: 16384-131072 for typical workloads.
    """

    datadist_tensor_size: Optional[int] = 1_000_000
    """
    Fixed tensor size for datadist all_to_all operation.

    If set, uses this fixed size instead of basing it on dense.numel().
    This allows controlling the all_to_all duration independently of dense_dim.

    Recommended: 1M-10M elements for ~100-500ms all_to_all duration.
    Set to None to use dense.numel() (original behavior, but can be very slow
    with large dense_dim).
    """

    def is_any_race_enabled(self) -> bool:
        """Check if any race injection is enabled."""
        return (
            self.h2d_memcpy_racing
            or self.datadist_racing
            or self.timing_skew_enabled
        )

    def is_h2d_race_enabled(self) -> bool:
        """Check if H2D racing is enabled."""
        return self.h2d_memcpy_racing and self.h2d_skip_sync_before_forward

    def is_datadist_race_enabled(self) -> bool:
        """Check if datadist racing is enabled."""
        return self.datadist_racing and self.datadist_skip_sync_before_collective

    def is_timing_skew_active(self, step: int) -> bool:
        """Check if timing skew should be applied at this step."""
        return (
            self.timing_skew_enabled
            and self.timing_skew_mode != "none"
            and step >= self.timing_skew_start_step
        )
