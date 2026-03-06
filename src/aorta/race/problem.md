Context

We are AMD engineers and help our custmer Meta to debug NaN issue on their recommendation system workload. Meta is not able to share the propriatory codebase for us to reproduce. The NaN issue does not happen on NVIDIA hardward.


March 6, 2026 – Facts and hypotheses after Meta debug session  

Issues in Meta 

Issue A: Default stream vs Side stream Race 

Workload 

Meta’s eval workload runs a pipelined forward-pass loop with zero CPU-GPU synchronization. Each iteration has: prefetch (H2D + data distribution on side streams), compiled forward pass, and two metrics updated on default stream 

Evidence from Meta 

NaN disappears with ROC_AQL_QUEUE_SIZE=1024 at batch sizes <=512  

Disabling either side stream (memcpy or datadist) independently eliminates NaN 

AMD traces show 3-4 iteration CPU-GPU lag; NVIDIA traces show 1-2 iterations 

NaN appears at ~350 iters at bs=512 without mitigation; disappears entirely with AQL=1024 

Facts: 

The CPU submits dispatch packets to the AQL queue. The GPU consumes them. The gap between the write pointer (CPU) and read pointer (GPU) is how far the CPU is head. 

On Nvidia , the queue holds ~1K packets. When full, the CPU blocks (backpressure). The CPU can never get more than ~1K dispatches ahead of the GPU.  

On AMD, the queue holds up to 16K packets. The CPU can submit 10K+ dispatches before any backpressure. The GPU can falls thousands of dispatches behind 

Mitigations that work for Issue A: 

ROC_AQL_QUEUE_SIZE=1024: Matches NVIDIA's queue depth, provides backpressure at 1K dispatches 

 Moving side stream work to the default stream: Serializes submission, CPU fills queue slower 

 GPU_MAX_HW_QUEUES=2: Reduces hardware parallelism, GPU keeps up better 

 Any form of CPU-GPU sync (.item(), synchronize()): Drains the queue periodically 

Hypothesis 

when the CPU is far ahead, two memory recycling mechanism cause corruption: 

Kernarg recycling: The HIP runtime reuses kernel argument buffers before the GPU reads them. The GPU executes a dispatch but finds arguments for a different kernel. 

Tensor recycling: Pytorch’s caching allocator recycles GPU memory blocks before the GPU finishes reading them. The GPU reads overwritten data from a later iteration. 

The corrupted data is still valid GPU memory (just wrong values), so the kernel runs without error but compute garbage, producing NaN. No crash, no error – slient corruption 

Using multiple side streams (memcpy, datadist) alongside the default stream allows the CPU to submit patches in parallel across streams, filling the queue faster than with a single stream 

 

Issue B: Large Batch + Pipelining NaN 

Workload 

Same eval pipeline as Issue A: pipelined forward-pass loop with prefetch on side streams, compiled forward pass, and two metric updates on default stream 

Evidence from Meta 

At bs >=1024 with ROC_AQL_QUEUE_SIZE=1024 (Issue A fully mitigated), NaN still appears 

Local run, 2 GPUs, bs=1024, AQL=1024 + gc=0: NaN ~340 iters 

MAST (2x8, bs=4096, AQL=1024): NaN from beginning 

MAST (2x8, bs=4096, AQL=1024 + gc=0): NAN (2/2 runs) 

torch.cuda.synchronize() at ALL pipeline points still produces NaN at bs=4096 -- queue depth is literally zero and NaN persists 

EVAL_DISABLE_PIPELINING=1 (disabling prefetch) eliminates NaN at any batch size, including bs=4096 

gc_collect_interval=0 (GC disabled) does NOT prevent Issue B 

Facts 

torch.cuda.synchronize() drains the AQL queue to zero (rptr catches up to wptr completely).  

Disabling pipelining means each iteration is independent: load data, compute, metrics, done, next iteration. No cross-iteration buffer sharing, no prefetch overlap 

With pipelining enabled, the pipeline object manages buffers across iterations -- it prefetches iteration N+1's data while iteration N's compute is still running, and it reuses buffer objects between iterations 

larger batch sizes change kernel launch parameters (tile sizes, grid dimensions, working set size). 

Hypothesis 

Hypothesis A: Torch.compile / Triton codegen bug on ROCM at larger tensor dimensions 

Torch.compile in the forward generates Triton kernels, Triton’s ROCM backend is less mature that CUDA backend. At large batch sizes (bs >=1024), Triton selects different tile sizes, grid dimensions and memory access patterns. The pipelined graph structure (where prefetch tensors are inputs) produces a different compiled graph than the non-pipeline version. A codegen bug in a specific kernel configuration would : 

Be AMD-specific 

Not be fixed by sync (It’s wrong code, and not wrong timing) 

Only trigger at large batch sizes (different kernel paramets) 

Only trigger with pipelining (different compiled graph) 

Explain NaN from the beginning at bs=4096 (deterministic wrong code) 

Hypothesis B: HIP memory coherence / cache visibility bug at large working sets 

AMD GPUs have a different cache hierarchy and coherence model than NVIDIA. Synchronize() ensures kernel completion but may not guarantee full memory writeback. At bs=4096, the working set may exceed certain cache thresholds, and stale data in caches could be read by subsequent kernels. With pipelining, buffers are reused across iterations (same virtual addresses), making cache staleness visible. Without pipelining, fresh allocations get different addresses, avoiding stale cache lines 

Hypothesis C: Pipeline buffer management has an AMD-specific code path or interacts differently with HIP 

The pipeline object manages buffer reuse across iterations. If it has any HIP-specific behavior (or if HIP’s memory mapping / pointer semantics differ subtly from CUDA), the pipeline could hand the wrong buffer contents to the forward pass. This would be a correctness bug in the buffer management logic specific to the ROCM path 

  Mitigations that work for Issue B 

EVAL_DISABLE_PIPELINING=1: each iteration is fully independent 

Reducing batch size to <= 512:  

  Mitigations that do NOT work for Issue B 

ROC_AQL_QUEUE_SIZE=1024: Fixes Issue A but not Issue B 

 torch.cuda.synchronize() at all pipeline points (bs=4096): Queue depth is zero, NaN persists 

Sync at all pipeline points (bs=4096): Even with serialized stream execution, NaN persists.  

 gc_collect_interval=0 (disable Python GC) at bs>=1024: NaN persists.  

AQL=1024 + report_interval=10 at bs=4096: Reducing metric reporting frequency doesn't help 

 