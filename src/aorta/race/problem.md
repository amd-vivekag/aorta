# Meta NaN Issue Debug Report

## Context

## System Setup

- PyTorch version: 2.11.0
- Gcn arch name: gfx950:sramecc+:xnack-
- ROCm version: 7.0.2.0-17-9428210
- Rocblas version: 5.0.2-20250912-42-1205-g554bb20204
- Hipblaslt version: 100200-7e32d53eb1
- Model precision: fp32
- Driver version: 6.16.6 ( rocm-smi --showdriverversion)

## Stack

- **Model:** DLRMv3 / HSTU (Hierarchical Sequential Transduction Unit) -- [MLCommons reference](https://github.com/mlcommons/inference/tree/master/recommendation/dlrm_v3)
- **Framework:** PyTorch 2.11.0 + ROCm 7.0.2, `torch.compile` with Triton backend
- **Embeddings:** TorchRec `EmbeddingCollection` -- 1B item table (dim=64, bf16), user + category tables, sharded across GPUs via `KeyedJaggedTensor`
- **Training pipeline:** TorchRec `TrainPipelineSparseDist` -- 3-stage pipeline with triple-buffered device tensors:
  - `memcpy_stream`: H2D copy from pinned host memory (iteration N+2)
  - `data_dist_stream`: RCCL `all_to_all` embedding redistribution (iteration N+1)
  - `default_stream`: compiled forward pass + metrics (iteration N)
- **Collectives:** RCCL `all_to_all_single` / `reduce_scatter` via `ProcessGroupNCCL` with `async_op=True`
- **Dense forward:** 5-layer multi-head self-attention + FFN + LayerNorm, output MLP (256 -> 1), bf16 mixed precision
- **Hardware:** AMD Instinct MI350X (gfx950), 252 GB HBM per GPU
- **Shampoo Optimizer:"" [github] (https://github.com/facebookresearch/optimizers/blob/main/distributed_shampoo/README.md) Open source does not open multistreams it seems

---

## Shampoo Nan Trace

**Date:** March, 10 2026

trace file: /mnt/vast/huzhao/projects/aorta/data/shampoo_debug_trace.json

1.  One failing training trace covering 2-5 consecutive steps — ideally with stream labels visible so we can walk through forward, backward, optimizer, and any collectives together
Trace: debug_trace.json

2. A few yes/no questions about the training loop behavior:
    • Is there any .item(), .cpu(), or synchronize() call between loss.backward() and optimizer.step()?
	Backward and optimizer uses the same stream, so they’re synchronized
    • Is loss or any metric logged to the host every step, or only every K steps? If every K, what is K?
	Every step, k=1
    • Is GradScaler (or any gradient scaling) used?
	No
    • Is autocast used, and does it wrap only the forward pass or also the optimizer step?
	Only the forward pass 
    • Does the training loop do anything between optimizer.step() and the next forward pass? (e.g., LR scheduler step, checkpoint save, any sync).
	Yes, metric reports

3. A few Shampoo configuration values (no code needed, just the values):
    • Is DDPDistributedConfig enabled?
	Yes
    • precondition_frequency value:
	4500
    • start_preconditioning_step value:
	4500
    • Does the NaN first appear before or after start_preconditioning_step?
	Undeterministic
    • Have you tried disabling DDPDistributedConfig? If so, does the NaN go
       Away
	No

4. Environment variables you're running with — in particular GPU_MAX_HW_QUEUES, ROC_AQL_QUEUE_SIZE, and any NCCL_* / RCCL_* settings.

- GPU_MAX_HW_QUEUES=2
- NCCL_MAX_NCHANNELS=48
- HSA_KERNARG_POOL_SIZE=4194304
- HSA_NO_SCRATCH_RECLAIM=1
- AMDGCN_USE_BUFFER_OPS=1
- TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE=1
- FBGEMM_NO_JK=1
- FBGEMM_TBE_V2=1
- FBGEMM_TBE_ROCM_HIP_BACKWARD_KERNEL=1
- FBGEMM_BOUNDS_CHECK_INDICES_V2=1
- PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

5. Where the first NaN appears — if you have any existing NaN detection even just which step it first appears on and whether it's in the loss, gradients, or parameters, that context will save us time. If not, we can work through it together during the session.
Nans have been undeterministic. It has appeared in SDPA fwd, SDPA backward, torch.log, tensor.div, and across a range of steps from < 1000 to > 20000. Most likely, these operations are just catching the nans and raising errors. If we enable nan detection for all operators, that introduces enough synchronization that the nan goes away. (edited)

---

## March 11, 2026 -- Trace Analysis: Shampoo DDPDistributedConfig Stream Race

### Trace Metadata

- **File:** `data/shampoo_debug_trace.json`
- **Host:** twshared9373.01.maz5.facebook.com
- **Job:** aps-ig_ctr_new_v0_MI350X_echen40_debug-0c535ade63 (trainer rank 0)
- **Hardware:** 8x MI350X (gfx950), 64 GPUs total (8 nodes)
- **Steps captured:** 5 consecutive steps (ProfilerStep#1998 through #2002), ~650ms/step
- **Total events:** 1,160,176

### Trace Structure

**CPU threads:**
- `thread 6076 (trainer_main)`: Main training thread -- 494K events, runs forward + optimizer + collectives
- `thread 46687 (pt_autograd_0)`: Backward pass -- 452K events, runs autograd + backward collectives
- `thread 17976/17975 (pt_gloo_runloop)`: Gloo broadcast for Shampoo DDPDistributedConfig sync

**GPU streams:**
- **Stream 0 (default):** 66K kernel events, 1.36s busy -- forward pass, backward, optimizer, AND some NCCL collectives
- **Stream 4 (NCCL):** 235 NCCL kernel events, 0.91s busy -- all_to_all for TorchRec embedding redistribution
- **Stream 8:** 2.2K events -- H2D memcpy (memcpy_stream for pipeline prefetch)
- **Stream 9:** 1.6K events -- D2D memcpy + fbgemm permute kernels (data_dist_stream)
- **Streams 16-30 (15 streams):** 1-2 bf16-to-fp32 conversion kernels each -- Shampoo gradient copy streams

### Model Architecture (reconstructed from trace)

**Batch & sequence:** batch_size=1024 (per GPU, 64 GPUs = 65K global), seq_len=200, d_model=96, bf16 precision

**Embedding tables (FBGEMM TBE, sharded via TorchRec `all_to_all`, rowwise AdaGrad optimizer):**

| Group | Tables | Total Rows (this rank) | Dim | Weight (fp16) | Type |
|-------|--------|------------------------|-----|---------------|------|
| 1 | 1 | 1.1M | 8 | 0.02 GB | unweighted |
| 2 | 1 | 16.0M | 8 | 0.24 GB | unweighted |
| 3 | 1 | 32.5M | 8 | 0.48 GB | unweighted |
| 4 | 1 | 3.4K | 8 | <0.01 GB | unweighted |
| 5 | 1 | 838K | 8 | 0.01 GB | unweighted |
| 6 | 2 | 33.5M | 8 | 0.50 GB | unweighted |
| 7 | 14 | 32.4M | 128 | 7.72 GB | unweighted |
| 8 | 1 | 6.7M | 128 | 1.60 GB | weighted |
| **Total** | **22** | **~123M** | | **10.57 GB** | **5.67B params** |

Group 7 (14 tables, dim=128) accounts for 73% of all embedding parameters. Group 8 is the only weighted table (per-feature importance weights). Tables are sharded across 64 GPUs, so global table sizes are 64x larger.

**HSTU attention block:** 14 SDPA calls per step (7 layers), two head configurations:
- 32-head: Q=[1024,1,32,96], KV=[1024,1,200,96] (5 calls)
- 16-head: Q=[1024,1,16,96], KV=[1024,1,200,96] (9 calls)
- Uses CK Flash Attention forward (`ck_tile`), AITer backward (`aiter::fmha_bwd`)

**Normalization:** 148x LayerNorm (on `[1024,200,96]`, `[1024,32,96]`, `[1024,16,96]`) + 61x RMSNorm (on MLP dims: 512, 2304, 3840, 4608)

**Dense MLP widths:** 96, 128, 192, 256, 512, 960, 1024, 1536, 2048, 2176, 3840, 4608, 5120. Final fan-out: `[512 -> 30720]` and `[512 -> 15360]`. BMM cross-interactions: `[1024,40,384] x [1024,384,128]`

**1D causal conv:** `conv2d([1024,96,200,1], [96,96,3,1])` -- kernel=3 over 200 timesteps

**Activations:** 214x sigmoid (gating), 60x relu, 20x gelu, 8x silu, 27x log_sigmoid

**Loss (multi-task):** 7x cross_entropy `[1024,1024]` + 50x BCE `[1024,1]` + 4x BCE `[1048576]`

**GPU compute profile (single step):**

| Category | Kernels | Time | % |
|----------|---------|------|---|
| Elementwise | 7,476 | 97.9ms | 38% |
| Other (multi_tensor, cat) | 745 | 43.0ms | 17% |
| GEMM/matmul | 1,186 | 40.1ms | 16% |
| NCCL (_all_gather_base) | 1 | 22.7ms | 9% |
| Embedding/TBE | 30 | 16.8ms | 7% |
| Reduce | 788 | 12.4ms | 5% |
| Index select | 14 | 10.2ms | 4% |
| LayerNorm | 82 | 8.6ms | 3% |
| Flash Attention | 56 | 1.4ms | 0.6% |

### Key Findings

#### Finding 1: Shampoo DDPDistributedConfig Uses 15 Separate Streams for Gradient Conversion

Streams 16 through 30 each run exactly 1-2 `bfloat16tofloat32_copy_kernel` invocations during each step. These occur at 40-70% of the step duration (during the backward pass), with 3 streams activated per step (one group of 3 new streams per step).

This is the Shampoo optimizer converting bf16 gradients to fp32 for preconditioner accumulation. Each parameter group gets its own stream. **These streams have no visible synchronization with the default stream** in the trace -- no `hipEventRecord`/`hipStreamWaitEvent` pairs are visible between them and stream 0.

#### Finding 2: Step 1999 is a Shampoo Preconditioner Computation Step

Step 1999 has a distinctly different collective profile from all other steps:

| Step | Main Thread Collectives |
|------|------------------------|
| 1998 | 35x all_to_all, 1x _all_gather_base |
| **1999** | **35x all_to_all, 1x _all_gather_base, 3x all_reduce_barrier, 6x all_gather** |
| 2000 | 35x all_to_all, 1x _all_gather_base |
| 2001 | 35x all_to_all, 1x _all_gather_base |
| 2002 | 35x all_to_all, 1x _all_gather_base |

The extra collectives in step 1999 (+475 to +550ms) are the Shampoo `DDPDistributedConfig` preconditioner distribution pattern:
- 3x `c10d::barrier` (all_reduce_barrier) -- synchronize before preconditioner exchange
- 6x `nccl:all_gather` -- distribute computed preconditioners across ranks (input types: `long` [2] for metadata, `double` [8x28] / [4x28] for preconditioner matrices)

The `all_gather` input shapes (`[8,28]` and `[4,28]` of type `double`) are Shampoo's Kronecker-factor preconditioner matrices being gathered across the 64-rank world.

#### Finding 3: 14 NCCL Kernels Execute on Stream 0 (Default Stream)

There are 14 NCCL kernels running on the **default stream** (stream 0), not on the dedicated NCCL stream 4. These break down as:

**Per-step _all_gather_base (one per step, 16-23ms each):**
These occur at ~70-75% of each step and correspond to the Shampoo DDPDistributedConfig `_all_gather_base` call. They run on stream 0 because Shampoo's optimizer step executes on the default stream.

| Step | GPU ts | Duration | GPU annotation |
|------|--------|----------|---------------|
| 1998 | ...378528 | 22.7ms | nccl:_all_gather_base |
| 1999 | ...041692 | 16.7ms | nccl:_all_gather_base |
| 2000 | ...751095 | 22.6ms | nccl:_all_gather_base |
| 2001 | ...382008 | 18.3ms | nccl:_all_gather_base |
| 2002 | ...020294 | 22.7ms | nccl:_all_gather_base |

**Step 1999 extra Shampoo collectives (9 kernels, 0.2-3.1ms each):**
- 3x `nccl:all_reduce_barrier` (0.2ms, 3.1ms, 1.3ms)
- 6x `nccl:all_gather` (0.3-1.6ms each)

**No NCCL overlap between stream 0 and stream 4.** NCCL kernels on stream 0 and stream 4 never execute concurrently -- the GPU serializes them.

#### Finding 4: Massive Cross-Stream Overlap Between Stream 0 and Stream 4

There are **3,463 overlap windows** (>10us each) where compute kernels on stream 0 and NCCL kernels on stream 4 execute simultaneously. The largest overlaps are:

| Overlap | Stream 0 Kernel | Stream 4 NCCL | Duration |
|---------|----------------|---------------|----------|
| 7.3ms | split_embedding_backward (adagrad) | ncclDevKernel_Generic_2 | 10.6ms |
| 7.1ms | split_embedding_backward (adagrad) | ncclDevKernel_Generic_2 | 11.7ms |
| 6.5ms | split_embedding_backward (adagrad) | ncclDevKernel_Generic_2 | 10.4ms |
| 4.2ms | split_embedding_backward (adagrad) | ncclDevKernel_Generic_2 | 14.5ms |
| 2.2ms | group_index_select_backward | ncclDevKernel_Generic_2 | 4.2ms |

The fbgemm `split_embedding_backward` kernels (7ms each) and NCCL all_to_all kernels (10-44ms each) run in parallel -- the backward embedding gradient computation and the TorchRec data redistribution overlap heavily.

#### Finding 5: CPU-GPU Lag Is Small (NOT the AQL Issue)

- **p50 lag:** 27us (0.03ms)
- **p99 lag:** 49.7ms
- **max lag:** 69.7ms (0.1 steps ahead)

The CPU is at most 0.1 training steps ahead of the GPU. This is NOT the Issue A (AQL queue depth) pattern, where the CPU was 3-4 iterations ahead. The AQL mitigations (`ROC_AQL_QUEUE_SIZE` is not set in this trace) are not the primary concern here.

#### Finding 6: _all_gather_base is Preceded by Massive aten::empty_strided Allocation Burst

Before each `_all_gather_base` launch at ~70% of the step, there is a burst of 20-26 `aten::empty_strided` calls (each <1us). These are the Shampoo optimizer allocating output buffers for the all_gather result. The CachingAllocator is returning blocks from its free pool.

**This is the exact CSAN-flagged pattern:** the CachingAllocator handing out blocks while an async collective on another stream may still be reading from them.

### New Hypothesis: Shampoo DDPDistributedConfig Stream Synchronization Bug

The trace reveals a specific mechanism for the NaN that is **distinct from Issue A (AQL) and Issue B (large batch)**:

**The Shampoo DDPDistributedConfig creates 15 separate CUDA streams (streams 16-30) for bf16-to-fp32 gradient conversion.** These streams are used during the backward pass to copy gradients into fp32 buffers for preconditioner accumulation. The preconditioner computation and its NCCL collectives then run on stream 0 (the default stream).

The race condition:
1. During the backward pass (~40% of step), Shampoo launches bf16->fp32 copy kernels on streams 16-30
2. These streams read from the same gradient tensors that stream 0 is computing (backward pass writes gradients)
3. There is no visible stream synchronization between streams 16-30 and stream 0
4. Shampoo then computes preconditioners on stream 0 using the fp32 buffers, but those buffers may not be fully written yet
5. The `_all_gather_base` at ~70% of the step then distributes potentially-corrupted preconditioner data across all ranks
6. Once corrupted preconditioners are applied to parameters, NaN propagates

This would explain:
- **NaN is nondeterministic**: Depends on GPU scheduling of kernels across streams
- **NaN goes away with synchronization**: Any sync drains the bf16->fp32 streams
- **NaN appears in diverse ops**: Corrupted parameters produce NaN in any subsequent computation
- **Not fixed by AQL tuning**: This is an intra-step stream sync bug, not a cross-iteration queue depth issue
- **GPU_MAX_HW_QUEUES=2 helps**: Fewer HW queues means more serialization between the 15 copy streams and stream 0

### Reproduction Plan (Our Side)

1. **Write a minimal Shampoo multi-stream reproducer:**
   - Create a model with multiple parameter groups
   - Use `DistributedShampoo` with `DDPDistributedConfig`
   - Launch bf16->fp32 gradient copies on N separate streams (matching Shampoo's pattern)
   - Run `_all_gather_base` on stream 0 to distribute preconditioners
   - Check for NaN in gathered output after ~1000 iterations
   - Compare behavior with/without `hipStreamSynchronize` on the copy streams before preconditioner computation

2. **Instrument Shampoo's stream usage:**
   - Clone [facebookresearch/optimizers](https://github.com/facebookresearch/optimizers)
   - Add `torch.cuda.synchronize()` between the bf16->fp32 copy and the preconditioner matmul
   - If NaN disappears, this confirms the stream sync bug

3. **Test with HSA_FORCE_FINE_GRAIN_PCIE=1:**
   - If the issue is cache coherence between streams, fine-grained memory should eliminate it

### Questions for Meta to Verify

1. **Does Shampoo's DDPDistributedConfig explicitly synchronize the bf16->fp32 copy streams before computing preconditioners?**
   Specifically, in `distributed_shampoo.py`, after the gradient copy kernels are launched on per-parameter streams, is there a `stream.synchronize()` or `event.wait()` before the preconditioner matrix multiply?

2. **Can you add `torch.cuda.synchronize()` ONLY between the gradient copy and the preconditioner computation in Shampoo (not everywhere)?**
   This is more targeted than enabling full NaN detection. If NaN disappears with just this one sync point, it confirms the stream race.

3. **Which Shampoo version are you using? Has `use_separate_streams_for_gradient_computation` been modified or is it the default?**
   The trace shows 15 distinct streams (16-30) for gradient conversion, which means `DDPDistributedConfig` is using separate streams. Confirming this setting narrows the fix.

4. **Does this NaN also reproduce with `DDPDistributedConfig(num_trainers_per_group=1)` (i.e., no cross-rank preconditioner sharing)?**
   If it still NaNs with `num_trainers_per_group=1`, the race is in the local stream sync, not the all_gather. If it goes away, the race is in the collective input buffer.

5. **Can you run with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False` to see if CCA behavior changes?**
   `expandable_segments:True` (currently set) may affect how the CCA reuses blocks across streams, widening the race window.

6. **Is `torch.compile` wrapping the Shampoo optimizer step, or only the model forward?**
   If compile wraps the optimizer, it may be removing stream synchronization that eager mode would insert.

---

## CUDA Stream Sanitizer (CSAN) Results on Meta's NaN Workload

**Date:** March 2026
**Configuration:** `ROC_AQL_QUEUE_SIZE=1024`, `GPU_MAX_HW_QUEUES=1`, `CSAN=1`

Meta ran the PyTorch CUDA Stream Sanitizer against their eval workload to detect stream-level data races.

### What CSAN Detected

CSAN found a **data race on a tensor** (data pointer `139991783932688`) between two streams:

| | Stream | Operation | Pipeline Stage |
|---|--------|-----------|----------------|
| **Access 1** | `data_dist_stream` (id `140011564741504`) | `aten::empty.memory_format` -- allocating a new tensor | `wait_sparse_data_dist` |
| **Access 2** | `default_stream` (id `0`) | `c10d::alltoall_base_` -- reading input tensor | `model_fwd` |

### Code Paths from Stack Traces

**data_dist_stream side:**
`trainer.py` -> `_train_loop` -> `progress()` -> `wait_sparse_data_dist()` -> `KJTAllToAllTensorsAwaitable.__init__()` at `dist_data.py:402` -> `torch.empty(...)`.
The CachingAllocator (CCA) returned a memory block at the raced address.

**default_stream side:**
`alltoall_base_(input=..., async_op=True)` -- the NCCL/RCCL all-to-all collective was still reading from a tensor at that same address as its input argument. The collective was launched asynchronously.

### The Race Scenario

```
Timeline:
  default_stream:     alltoall_base_(input=0x7F4B...0A10, async_op=True) --> still reading input
  data_dist_stream:   torch.empty() --> CCA returns block at 0x7F4B...0A10 --> OVERWRITES
```

The CachingAllocator reused the memory block while the async `alltoall_base_` on the default stream was still reading from it.

### Meta's Interpretation

> "CSAN detected race between the data_dist_stream and the default_stream. The former is requesting a buffer during wait_sparse_data_dist stage of pipelining, while the latter is doing some NCCL ops (all2all here, but I've also seen reduce scatter etc.) during model_fwd. My interpretation: when the data_dist_stream requests the buffer, CachingAllocator handed out a block that was still in use by the default stream."

### Three Hypotheses Discussed

**Hypothesis 1: NCCL/RCCL dropped a tensor reference (RULED OUT)**

Meta initially suspected that NCCL internal code failed to keep a reference to the all-to-all input tensor, allowing the CCA to recycle it. They examined a specific PR that introduced a new tensor to hold the A2A input but failed to keep refs. However, `ProcessGroupNCCL.cpp` correctly stashes both inputs and outputs of async collective ops. They could not reproduce the race by manually queuing async comms + allocating on another stream.

**Hypothesis 2: HIP event completion reports success prematurely (STRONGEST LEAD)**

Proposed by Jeremy Hadidjojo at Meta. The mechanism:

1. When a cross-stream buffer is deleted, the CachingAllocator inserts `hipEvent` on every stream using it
2. The block doesn't get freed until all events report completion
3. If `hipEventQuery()` returns `hipSuccess` **before the GPU actually finishes** using the memory, the CCA would recycle the block too early

This would be an AMD-specific bug in HIP event management -- the event signals completion to the CPU before the GPU has actually finished the associated kernel/collective.

This hypothesis is consistent with the Shampoo experiment matrix:

- **Row 2** (`GPU_MAX_HW_QUEUES=2` fixes): Fewer HW queues = fewer concurrent operations = smaller window for premature event completion
- **Row 4** (H2D on default stream fixes): No cross-stream event needed -- same-stream ordering guaranteed by queue semantics
- **Row 6** (same non-default stream still races): Even on the same user-stream, if RCCL internally uses sub-streams or the HW queue mapping differs from default stream, event behavior changes
- **Row 8** (`NCCL_LAUNCH_ORDER_IMPLICIT=1` fixes): Forces RCCL to serialize launches, making events complete in order
- **Row 11** (logging masks bug): Extra serialization from logging gives events time to "catch up"

**Hypothesis 3: CSAN false positive due to async c10d::Work**

Raised by Jeff Daily. CSAN may not understand the semantics of `async_op=True` in NCCL collectives. When `alltoall_base_` returns a `c10d.Work` handle, the GPU operation continues after the Python call returns. CSAN hooks at the dispatcher level and records accesses at dispatch time -- it may not model the extended GPU-side lifetime of async collective ops. If CSAN thinks the `alltoall_base_` is "done" when the Python call returns, it would incorrectly flag a subsequent allocation as a race.

However, even if the CSAN report is a false positive in its detection mechanism, the NaN is real. The report may still be pointing at the right tensor and the right streams.

### Assessment

**Hypothesis 2 (premature HIP event completion) is the strongest lead** because:

1. It's AMD-specific -- CUDA's event implementation is battle-tested; HIP's maps to HSA signals which have different completion semantics
2. It explains the full Shampoo experiment matrix (serialization, default stream, implicit launch order all reduce the window for premature completion)
3. It's mechanistically sound -- the CCA relies on `hipEventQuery()` to decide when to recycle blocks, and if HSA signals report completion before memory writes are globally visible, the CCA would hand out still-in-use memory

However, we are not able to reproduce HIP event issue on our side.

### Recommended Next Steps

1. **Test the HIP event hypothesis:** Patch PyTorch's CachingAllocator to call `hipEventSynchronize()` instead of `hipEventQuery()` in `process_events()`. If NaN disappears, this confirms premature HIP event completion.

2. **Check HSA signal behavior:** On the AMD side, verify that `hsa_signal_wait_*()` used by HIP events correctly waits for memory write visibility (not just kernel completion). There may be a missing cache flush / memory fence between the RCCL kernel's last memory write and the HSA signal being decremented.

3. **Validate CSAN on CUDA:** Run the same workload on NVIDIA with CSAN enabled. If the same race is flagged on CUDA but doesn't produce NaN, it confirms a CSAN false positive for async ops. If it's NOT flagged on CUDA, the race is AMD-specific.

4. **Instrument CCA event polling:** Add logging to `process_events()` to record when events are queried, when they return complete, and when blocks are recycled. Compare the recycled block addresses against active RCCL collective inputs to catch premature reuse in the act.

---

## March 6, 2026 -- Facts and Hypotheses after Meta Debug Session

---

## Issue A: Default Stream vs Side Stream Race

### Workload

Meta's eval workload runs a pipelined forward-pass loop with zero CPU-GPU synchronization. Each iteration has:
1. Prefetch (H2D + data distribution on side streams)
2. Compiled forward pass
3. Two metrics updated on default stream

### Evidence from Meta

- NaN disappears with `ROC_AQL_QUEUE_SIZE=1024` at batch sizes <= 512
- Disabling either side stream (memcpy or datadist) independently eliminates NaN
- AMD traces show 3-4 iteration CPU-GPU lag; NVIDIA traces show 1-2 iterations
- NaN appears at ~350 iters at bs=512 without mitigation; disappears entirely with AQL=1024

### Facts

- The CPU submits dispatch packets to the AQL queue. The GPU consumes them. The gap between the write pointer (CPU) and read pointer (GPU) is how far the CPU is ahead.
- On **NVIDIA**, the queue holds ~1K packets. When full, the CPU blocks (backpressure). The CPU can never get more than ~1K dispatches ahead of the GPU.
- On **AMD**, the queue holds up to 16K packets. The CPU can submit 10K+ dispatches before any backpressure. The GPU can fall thousands of dispatches behind.

### Hypothesis

When the CPU is far ahead, two memory recycling mechanisms cause corruption:

- **Kernarg recycling:** The HIP runtime reuses kernel argument buffers before the GPU reads them. The GPU executes a dispatch but finds arguments for a different kernel.
- **Tensor recycling:** PyTorch's caching allocator recycles GPU memory blocks before the GPU finishes reading them. The GPU reads overwritten data from a later iteration.

The corrupted data is still valid GPU memory (just wrong values), so the kernel runs without error but computes garbage, producing NaN. No crash, no error -- silent corruption.

Using multiple side streams (memcpy, datadist) alongside the default stream allows the CPU to submit packets in parallel across streams, filling the queue faster than with a single stream.

### Mitigations That Work for Issue A

- `ROC_AQL_QUEUE_SIZE=1024` -- Matches NVIDIA's queue depth, provides backpressure at 1K dispatches
- Moving side stream work to the default stream -- Serializes submission, CPU fills queue slower
- `GPU_MAX_HW_QUEUES=2` -- Reduces hardware parallelism, GPU keeps up better
- Any form of CPU-GPU sync (`.item()`, `synchronize()`) -- Drains the queue periodically

---

## Issue B: Large Batch + Pipelining NaN

### Workload

Same eval pipeline as Issue A: pipelined forward-pass loop with prefetch on side streams, compiled forward pass, and two metric updates on default stream.

### Evidence from Meta

- At bs >= 1024 with `ROC_AQL_QUEUE_SIZE=1024` (Issue A fully mitigated), NaN still appears
- Local run, 2 GPUs, bs=1024, AQL=1024 + gc=0: NaN ~340 iters
- MAST (2x8, bs=4096, AQL=1024): NaN from beginning
- MAST (2x8, bs=4096, AQL=1024 + gc=0): NaN (2/2 runs)
- `torch.cuda.synchronize()` at ALL pipeline points still produces NaN at bs=4096 -- queue depth is literally zero and NaN persists
- `EVAL_DISABLE_PIPELINING=1` (disabling prefetch) eliminates NaN at any batch size, including bs=4096
- `gc_collect_interval=0` (GC disabled) does NOT prevent Issue B

### Facts

- `torch.cuda.synchronize()` drains the AQL queue to zero (rptr catches up to wptr completely).
- Disabling pipelining means each iteration is independent: load data, compute, metrics, done, next iteration. No cross-iteration buffer sharing, no prefetch overlap.
- With pipelining enabled, the pipeline object manages buffers across iterations -- it prefetches iteration N+1's data while iteration N's compute is still running, and it reuses buffer objects between iterations.
- Larger batch sizes change kernel launch parameters (tile sizes, grid dimensions, working set size).

### Hypotheses

**Hypothesis A: Torch.compile / Triton codegen bug on ROCm at larger tensor dimensions**

Torch.compile in the forward generates Triton kernels. Triton's ROCm backend is less mature than the CUDA backend. At large batch sizes (bs >= 1024), Triton selects different tile sizes, grid dimensions, and memory access patterns. The pipelined graph structure (where prefetch tensors are inputs) produces a different compiled graph than the non-pipeline version. A codegen bug in a specific kernel configuration would:

- Be AMD-specific
- Not be fixed by sync (it's wrong code, not wrong timing)
- Only trigger at large batch sizes (different kernel parameters)
- Only trigger with pipelining (different compiled graph)
- Explain NaN from the beginning at bs=4096 (deterministic wrong code)

**Hypothesis B: HIP memory coherence / cache visibility bug at large working sets**

AMD GPUs have a different cache hierarchy and coherence model than NVIDIA. `synchronize()` ensures kernel completion but may not guarantee full memory writeback. At bs=4096, the working set may exceed certain cache thresholds, and stale data in caches could be read by subsequent kernels. With pipelining, buffers are reused across iterations (same virtual addresses), making cache staleness visible. Without pipelining, fresh allocations get different addresses, avoiding stale cache lines.

**Hypothesis C: Pipeline buffer management has an AMD-specific code path or interacts differently with HIP**

The pipeline object manages buffer reuse across iterations. If it has any HIP-specific behavior (or if HIP's memory mapping / pointer semantics differ subtly from CUDA), the pipeline could hand the wrong buffer contents to the forward pass. This would be a correctness bug in the buffer management logic specific to the ROCm path.

### Mitigations That Work for Issue B

- `EVAL_DISABLE_PIPELINING=1` -- Each iteration is fully independent
- Reducing batch size to <= 512

### Mitigations That Do NOT Work for Issue B

- `ROC_AQL_QUEUE_SIZE=1024` -- Fixes Issue A but not Issue B
- `torch.cuda.synchronize()` at all pipeline points (bs=4096) -- Queue depth is zero, NaN persists
- Sync at all pipeline points (bs=4096) -- Even with serialized stream execution, NaN persists
- `gc_collect_interval=0` (disable Python GC) at bs >= 1024 -- NaN persists
- `AQL=1024 + report_interval=10` at bs=4096 -- Reducing metric reporting frequency doesn't help

---

## Jan 1, 2026

## Shampoo NaN Issue

Meta engineering team conducted systematic experiments to isolate the NaN source:

| # | Experiment | Configuration | NaN Observed? |
|---|-----------|---------------|---------------|
| 1 | Baseline | `GPU_MAX_HW_QUEUES=4` | YES |
| 2 | Serialized HW queues (workaround) | `GPU_MAX_HW_QUEUES=2` | YES |
| 3 | cudaStreamDefault flag | Changed `kDefaultFlags` in PyTorch `CUDAStream.cpp` | NO (but much slower) |
| 4 | Memcpy on default stream | H2D stream = `torch.cuda.default_stream()` | NO (5% faster than workaround) |
| 5 | Datadist on default stream | Datadist stream = `torch.cuda.default_stream()` | NO |
| 6 | H2D + datadist on same (non-default) stream | Both on 1 new stream, `GPU_MAX_HW_QUEUES=2` | YES |
| 7 | TorchRec base class instead of datadist | No datadist class | NO |
| 8 | RCCL implicit launch order | `NCCL_LAUNCH_ORDER_IMPLICIT=1` | NO (lower QPS) |
| 9 | RCCL cheap fence off | `RCCL_GFX942_CHEAP_FENCE_OFF=1` | YES |
| 10 | Implicit order + cheap fence off | `NCCL_LAUNCH_ORDER_IMPLICIT=1` + `RCCL_GFX942_CHEAP_FENCE_OFF=1` | NO (very slow) |
| 11 | AMD logging enabled | `AMD_LOG_LEVEL` set | NO |
| 12 | Disable SDMA | `HSA_ENABLE_SDMA=0` | YES |
| 13 | Forced blit copy | `GPU_FORCE_BLIT_COPY_SIZE=128` | YES |
| 14 | Increased signal pool | `ROC_SIGNAL_POOL_SIZE=16384` | YES |
| 15 | Disabled batch CPU sync | `DEBUG_CLR_BATCH_CPU_SYNC_SIZE=0` | YES |

### Key Observations from Experiments

- **HW queue parallelism is central:** NaN disappears when `GPU_MAX_HW_QUEUES=2` (serialized) but appears with `=4` (parallel).
- **Stream identity matters, not just count:** Row 6 shows NaN persists even with `GPU_MAX_HW_QUEUES=2` when H2D and datadist share a non-default stream. This means serializing HW queues alone is not the full fix -- what matters is *which stream* the work is placed on.
- **Default stream eliminates NaN:** Moving either H2D (row 4) or datadist (row 5) to the default stream eliminates NaN. Moving H2D to the default stream is the fastest workaround (~5% faster than the `GPU_MAX_HW_QUEUES=2` workaround).
- **AMD logging masks the bug:** Enabling AMD logging introduces enough serialization to hide the race condition (row 11).
- **RCCL fence alone doesn't help:** `RCCL_GFX942_CHEAP_FENCE_OFF=1` alone still produces NaN (row 9), but combined with `NCCL_LAUNCH_ORDER_IMPLICIT=1` it eliminates NaN at significant performance cost (row 10).
- **Not SDMA, signal-pool, or blit-copy related:** Rows 12-15 all still produce NaN, ruling out these mechanisms.

---

## March 12, 2026 -- Reproduction Attempts and Findings

### Reproduction Summary

We ran extensive NaN reproduction attempts on AMD MI355X GPUs using the open-source Distributed Shampoo library with DDPDistributedConfig. **None produced NaN.**

| Run | Config | Steps | GPUs | NaN? |
|-----|--------|-------|------|------|
| 1 | Shampoo + NCCL traffic + alloc stress, precond_freq=10, GPU_MAX_HW_QUEUES=4 | 10,000 | 8 | NO |
| 2 | Shampoo precond every step, 256 params, d=256, GPU_MAX_HW_QUEUES=4 | 2,300+ | 8 | NO |
| 3 | Injected bf16→fp32 on 15 unsync streams, precond every step | 5,000 | 8 | NO |
| 4 | RCCL all_to_all + Shampoo training, no record_stream, 64MB payload | 5,000 | 8 | NO |
| 5 | Pipeline CCA race (freed NCCL send bufs without record_stream) | 5,000 | 8 | NO |

### Key Finding: Open-Source Shampoo Does NOT Create Separate Streams

The trace analysis (Finding 1) identified 15 separate CUDA streams (16-30) with bf16→fp32 gradient copy kernels. However, **the open-source Distributed Shampoo library does NOT create separate streams.** All gradient processing (`_merge_and_block_gradients`, `_update_preconditioners`, `_compute_search_directions`, `update_params`) runs on the default stream.

The open-source Shampoo's `all_gather_into_tensor` call (in `DDPDistributor.update_params()`) is synchronous (`async_op=False`), meaning it waits for RCCL to complete before returning. There is no window for a RCCL-CCA race.

This means Meta has **custom modifications** to either:
1. The Shampoo optimizer (adding per-parameter-group streams for gradient conversion)
2. The training framework (additional asynchronous operations overlapping with Shampoo)
3. TorchRec's pipeline (the specific record_stream pattern differs from what we simulated)

### Revised Hypothesis

The NaN is NOT caused by the open-source Shampoo DDPDistributedConfig code alone. The root cause is in Meta's **custom training framework/pipeline code** that:
1. Creates separate CUDA streams for gradient dtype conversion (streams 16-30 in trace)
2. Does NOT synchronize those streams with the default stream before Shampoo reads the converted gradients
3. OR has a missing `record_stream` call somewhere in TorchRec's `TrainPipelineSparseDist` that allows the CCA to recycle buffers while RCCL is still reading them

### What We Need From Meta

1. **Does your Shampoo version use `use_separate_streams_for_gradient_computation` or similar?** The open-source version does not. If yes, this is almost certainly the bug.
2. **Can you test `PYTORCH_NO_CUDA_MEMORY_CACHING=1`?** This disables the CCA entirely. If NaN disappears, the bug is a missing `record_stream`. If it persists, the bug is a missing stream sync.
3. **The streams 16-30 in the trace: what creates them?** They run exactly 1-2 `bfloat16tofloat32_copy_kernel` each. If Shampoo creates them, the fix is adding stream sync. If TorchRec or another component creates them, the fix may be elsewhere.

---

## March 12, 2026 -- Trace-Faithful Multi-Node Reproduction Attempts

### Approach: Monkey-Patched Shampoo with Trace-Exact Stream Race

Instead of using the open-source Shampoo as-is, we **monkey-patched** its `merge_and_block_gradients` method to inject the exact stream pattern observed in Meta's trace:

1. Free previous step's gradient copy buffers without `record_stream` (CCA reuse pressure)
2. Allocation pressure: 16x `torch.empty` + `del` to force CCA to hand out recently-freed blocks
3. Call original `merge_and_block_gradients()` to get bf16 gradient blocks
4. Clone each gradient block on a **separate CUDA stream** (15 streams, round-robin) — NO synchronization to stream 0
5. Return cloned gradients; Shampoo's preconditioner computation reads them on stream 0 (potential race)

Additionally, the pipeline reproduces:
- 3-stage TorchRec pattern: H2D (memcpy_stream), all_to_all (datadist_stream), forward/backward/optimizer (default_stream)
- Default-stream all_to_all (matching the Shampoo `_all_gather_base` from the trace)
- Intentional CSAN race: send_buf freed before all_to_all completes, allocation pressure on the freed buffer

### Reproduction Results

All runs used `GPU_MAX_HW_QUEUES=4`, `NCCL_NET=Socket` (for cross-node), bf16 model, wide MLP (up to 4096 hidden), 22.5M dense params under Shampoo.

| Run | Nodes | GPUs | Steps | Precond Freq | Patched? | NaN? |
|-----|-------|------|-------|--------------|----------|------|
| 1 | 1 | 8 | 1,200+ | 10 | No (old wrapper) | NO |
| 2 | 2 | 16 | 600+ | 10 | No (old wrapper) | NO |
| 3 | 3 | 24 | 1,200+ | 10 | No (old wrapper) | NO |
| 4 | 3 | 24 | 800+ | 1 (every step) | Yes (monkey-patched) | NO |

### Why the Race Doesn't Produce NaN on Our Setup

The `.clone()` of gradient blocks (shapes like [96], [384], [96,96], [4096,2048]) on separate CUDA streams completes nearly instantly relative to the GPU execution pipeline. The HW queue scheduler on MI355X serializes these small copies fast enough that stream 0 never observes stale data.

Key factors that may differ from Meta's production setup:
1. **Scale**: Meta runs on 64 GPUs (8 nodes); we tested up to 24 GPUs (3 nodes). More ranks means more RCCL collectives, larger all_gather payloads, and more CCA contention.
2. **Model size**: Meta's full DLRMv3/HSTU has 5.67B embedding params + dense MLPs up to [512→30720]. Our model is 86.5M total — the gradient copy kernels are 100x smaller.
3. **HIP event behavior**: Meta's Hypothesis 2 (premature `hipEventQuery()` completion) is a hardware/driver-level issue. Our HIP driver (ROCm 7.1) may not have this bug, or it may require very specific contention patterns to trigger.
4. **RCCL internal streams**: The CSAN report shows the race involves RCCL's internal use of streams. RCCL version, transport (IB vs Socket), and network topology all affect internal stream scheduling.

### Conclusion

The trace-observed stream race pattern (15 unsynchronized bf16→fp32 copy streams + CCA reuse pressure) does **not** produce NaN in our environment, even with:
- Monkey-patched Shampoo reading potentially-unfinished copies
- Preconditioner computation every step
- 3 nodes / 24 GPUs
- 64 MB async all_to_all payloads overlapping with optimizer
- Aggressive CCA allocation pressure

This strongly suggests the NaN requires either:
- **The specific HIP event bug** (premature `hipEventQuery()` completion) that exists in Meta's driver/HW revision but not ours
- **The production-scale workload** (64 GPUs, 5.67B params, full TorchRec pipeline with FBGEMM TBE) to create enough contention
- **A different mechanism entirely** that isn't the stream race we identified in the trace

### Version-Matched Reproduction (ROCm 7.0)

We extracted the exact HIP version from Meta's trace and matched it:

| Component | Meta (from trace) | Our matched venv | Match? |
|-----------|-------------------|------------------|--------|
| HIP runtime | 7.0.51831 | **7.0.51831** | EXACT |
| HIP driver | 70051831 | 7.0.51831 | EXACT |
| PyTorch | 2.11.0.dev | 2.11.0.dev20260206+rocm7.0 | Close |
| RCCL/NCCL | 2.27.7.dev | 2.26.6 | Older |
| GPU | gfx950, 256 SMs, 288GB | gfx950 MI355X | Same |

Results with version-matched ROCm 7.0:

| Run | Nodes | GPUs | Steps | NaN? |
|-----|-------|------|-------|------|
| 5 | 1 | 8 | 1,300+ (running) | NO |
| 6 | 2 | 16 | 1,200+ (running) | NO |

Even with the **exact same HIP runtime version** as Meta, the stream race does not produce NaN.

### Updated Questions for Meta

1. ~~Which ROCm / HIP driver version are you running?~~ **Answered from trace: HIP 7.0.51831 (ROCm 7.0)**. We've now matched this exactly and still cannot reproduce.
2. **Can you run our `trace_faithful_nan.py` script (attached) on your 64-GPU setup?** It monkey-patches Shampoo with 15 unsync copy streams. If it doesn't reproduce on your infra either, the race is NOT in the copy streams.
3. **Can you test with `PYTORCH_NO_CUDA_MEMORY_CACHING=1`?** This is the most targeted test: if NaN disappears, it's definitely a CCA race (missing `record_stream`). If it persists, it's a stream synchronization issue (not CCA related).
4. **What creates streams 16-30 in the trace?** We've confirmed the open-source Shampoo does not create them. Is it a custom Shampoo fork, a TorchRec component, or the training framework?
5. **Can you share the RCCL version?** Trace shows NCCL 2.27.7.dev. Our closest match is 2.26.6 (from rocm7.0 PyTorch nightly). The `.dev` suffix suggests a custom RCCL build -- is there a specific RCCL commit hash or patch?
