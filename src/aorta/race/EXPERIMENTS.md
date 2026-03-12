# Eval Pipelined Mode -- Experiment Guide

This document describes the `eval_pipelined` mode and all experiments for investigating NaN in a pipelined eval loop. Three distinct root causes are covered:

- **Experiment A**: Queue depth race (CPU races ahead of GPU via AQL queue)
- **Experiment B**: Large batch + pipelining NaN (independent of queue depth)
- **CCA Cross-Stream Event Race**: CUDA Caching Allocator recycles blocks across streams without `record_stream()`

---

## Setup

### Docker

```bash
cd /apps/oyazdanb/aorta

# Build (first time only)
docker compose -f docker/docker-compose.public-rocm72-torch2.12-shampoo.yaml build

# Start
docker compose -f docker/docker-compose.public-rocm72-torch2.12-shampoo.yaml up -d

# Enter the container
docker exec -it training-rocm72-torch2.12-shampoo bash
```

Inside the container:

```bash
export PYTHONPATH=/workspace/aorta/src:$PYTHONPATH
```

All `torchrun` commands below should be run inside this container.

### Pipeline Architecture

```
Pipelined (steady state):
    memcpy_stream:   [H2D batch N+1 (prefetch)] ─────────────────────────────────┐
    datadist_stream: [reduce_scatter + send/recv (prefetch)] ───────────────────┐ │
                                                                                │ │
    default_stream:  [wait memcpy] [wait datadist]                              │ │
                     [compiled_forward(batch_N)]                                │ │
                     [update_metrics: NE + MAE]                                 │ │
                     [update_reg_metrics: calibration]                          │ │
                     ← swap buffers ────────────────────────────────────────────┘ │
                     -- NO CPU-GPU sync, CPU races ahead ────────────────────────┘

Unpipelined (control):
    [H2D] → sync → [datadist] → sync → [forward] → [metrics] → sync
```

---

## Experiment A: Queue Depth Race

### What Happens

The pipelined eval loop has **no CPU-GPU synchronization** between iterations. The CPU submits GPU work far faster than the GPU executes it. On AMD, the AQL queue holds 16K packets (vs 1K on NVIDIA), so the CPU can get thousands of dispatches ahead, causing stale/recycled data and NaN.

### How to Identify

- NaN appears after ~350 iterations (not immediately)
- Only at default AQL queue size (16K)
- `ROC_AQL_QUEUE_SIZE=1024` eliminates NaN at batch_size <= 512

### Tests

| ID | Config / Command | Expected | What It Proves |
|----|-----------------|----------|----------------|
| A1 | `--config config/race/eval_exp_a_reproduce.yaml` | NaN ~350 iters | CPU races ahead, AQL=16K |
| A2 | `--config config/race/eval_exp_a_mitigated.yaml` | Clean | AQL=1024 backpressure |
| A3 | `--mode eval_pipelined --batch-size 512 --use-compile --sync-policy none --hw-queues 2 --verify 500` | Clean | HWQ=2 reduces parallelism |
| A4 | `nproc=1` + A1 config | NaN or Clean | Isolates RCCL from H2D |
| A5 | A1 config + `--disable-sdma` | ? | SDMA engine involved? |
| A6 | `TORCH_CUDA_SANITIZER=1` + A1 config | ? | Cross-stream access? |

### Commands

**A1 -- Reproduce:**

```bash
torchrun --nproc_per_node=2 -m aorta.race \
    --config config/race/eval_exp_a_reproduce.yaml
```

**A2 -- Mitigate with AQL=1024:**

```bash
torchrun --nproc_per_node=2 -m aorta.race \
    --config config/race/eval_exp_a_mitigated.yaml
```

**A3 -- Mitigate with HW queues=2:**

```bash
torchrun --nproc_per_node=2 -m aorta.race --mode eval_pipelined \
    --batch-size 512 --use-compile --sync-policy none \
    --hw-queues 2 --verify 500
```

**A4 -- Single GPU:**

```bash
torchrun --nproc_per_node=1 -m aorta.race --mode eval_pipelined \
    --batch-size 512 --use-compile --sync-policy none --verify 500
```

**A5 -- Disable SDMA:**

```bash
torchrun --nproc_per_node=2 -m aorta.race \
    --config config/race/eval_exp_a_reproduce.yaml --disable-sdma
```

**A6 -- Stream sanitizer:**

```bash
TORCH_CUDA_SANITIZER=1 torchrun --nproc_per_node=2 -m aorta.race \
    --config config/race/eval_exp_a_reproduce.yaml
```

---

## Experiment A with DLRMv3 (Compute-Heavy Attention)

The `mlp` and `dlrm` model types are memory-bandwidth bound (~0.1-0.5ms per forward). The GPU finishes before the CPU submits the next iteration, so the CPU never races ahead. The `dlrm_v3` model uses HSTU-style causal attention (`O(seq_len^2 * embed_dim)`) which at `seq_len=200` takes ~15ms per forward on MI300X -- enough for CPU-GPU lag.

The datadist output (`embed_shard`) is projected and added to dense features on the forward data path. Any RCCL signaling or cache coherence bug directly poisons the forward input. The model returns raw logits (no sigmoid) -- `binary_cross_entropy_with_logits` is highly sensitive to corruption.

### Tests

| ID | Config / Command | Expected | What It Proves |
|----|-----------------|----------|----------------|
| A7 | `--config config/race/eval_exp_a_dlrmv3.yaml` | NaN | Datadist + attention + deep AQL |
| A8 | A7 config + `--aql-queue-size 1024` | Clean | Backpressure limits CPU lead |
| A9 | A7 config + `--use-bfloat16` | NaN, earlier | Lower precision amplifies corruption |
| A10 | `nproc=1` + A7 config | NaN or Clean | RCCL involved? |
| A11 | A7 config + `--seq-len 50` | Likely Clean | GPU too fast, no lag |
| A12 | A7 config + `--seq-len 500` | NaN, earlier | Deeper AQL fill |
| A13 | A7 config + `--profile --profile-iterations 5` | N/A | Trace collection only |

### Commands

**A7 -- DLRMv3 reproduce:**

```bash
torchrun --nproc_per_node=2 -m aorta.race \
    --config config/race/eval_exp_a_dlrmv3.yaml
```

**A8 -- DLRMv3 mitigate:**

```bash
torchrun --nproc_per_node=2 -m aorta.race \
    --config config/race/eval_exp_a_dlrmv3.yaml --aql-queue-size 1024
```

**A9 -- DLRMv3 bfloat16:**

```bash
torchrun --nproc_per_node=2 -m aorta.race \
    --config config/race/eval_exp_a_dlrmv3.yaml --use-bfloat16
```

**A10 -- DLRMv3 single GPU:**

```bash
torchrun --nproc_per_node=1 -m aorta.race \
    --config config/race/eval_exp_a_dlrmv3.yaml
```

**A11 -- DLRMv3 short seq_len:**

```bash
torchrun --nproc_per_node=2 -m aorta.race \
    --config config/race/eval_exp_a_dlrmv3.yaml --seq-len 50
```

**A12 -- DLRMv3 long seq_len:**

```bash
torchrun --nproc_per_node=2 -m aorta.race \
    --config config/race/eval_exp_a_dlrmv3.yaml --seq-len 500
```

### If A7 Does Not Reproduce

Escalate real GPU compute before synthetic padding (which adds CPU overhead):

1. Increase real compute: `--hstu-num-layers 7` or `--embedding-dim 768`
2. Increase seq_len: `--seq-len 300`, `--seq-len 500`, `--seq-len 700`
3. Extend runtime: `--verify 10000`
4. Last resort -- synthetic padding: `--gpu-padding-dispatches 25` (keep small)

### Tuning seq_len

| seq_len | GPU time (MI300X, 5 layers, bs=512) | CPU-GPU lag |
|---------|-------------------------------------|-------------|
| 17 | ~0.5ms | None |
| 100 | ~5ms | GPU starts falling behind |
| 200 | ~15ms | CPU 2-3 iters ahead (Exp A range) |
| 500 | ~80ms | Deep AQL fill |

---

## Experiment A: CCA Cross-Stream Event Race

### Background

CSAN (`TORCH_CUDA_SANITIZER=1`) detected a data race in a TorchRec pipeline: `torch.empty()` on `data_dist_stream` received a CCA block that `alltoall_base_` on `default_stream` was still reading. The root cause is **missing `record_stream()`** -- CCA only tracks the allocation stream, so it recycles blocks as soon as the side stream's event completes, even if the default stream is still reading.

### Why Earlier Tests (A7) Did Not Reproduce

A7 uses static pre-allocated buffers that are pointer-swapped each iteration. No tensor is ever freed during the loop, so CCA's cross-stream event recycling is never triggered. The race requires:

1. Dynamic allocation on side streams (new `torch.empty()` each iteration)
2. Freeing of previous iteration's tensors (CCA records events)
3. Cross-stream usage without `record_stream()` (CCA only tracks allocation stream)
4. CPU ahead of GPU (freed block's default-stream work still in-flight when recycled)

### The Race Mechanism

```
Iter N:
  datadist_stream: embed_shard_N = torch.empty(S)  [allocated on datadist_stream]
                   reduce_scatter(embed_shard_N, ...)

  default_stream:  wait_stream(datadist_stream)
                   forward(embed_shard_N)           [reads on default_stream]

  _swap_buffers(): embed_shard_current = embed_shard_N
                   old embed_shard_{N-1} freed → CCA event on datadist_stream ONLY

Iter N+1:
  datadist_stream: embed_shard_{N+1} = torch.empty(S)
                   → CCA checks free pool: embed_shard_{N-1}'s block!
                   → CCA checks datadist_stream event: COMPLETE
                   → CCA gives block to embed_shard_{N+1}
                   → reduce_scatter writes to embed_shard_{N+1}
                   → BUT default_stream is still reading embed_shard_{N-1} (SAME BLOCK!)
                   → CORRUPTION → NaN
```

With `record_stream(default_stream)`, CCA tracks both streams. The block is not recycled until both events complete.

### Tests

| ID | Config / Command | Expected | What It Proves |
|----|-----------------|----------|----------------|
| A14 | A7 config + `--cca-cross-stream-alloc --no-cca-record-stream` | NaN | CCA recycles across streams |
| A15 | A7 config + `--cca-cross-stream-alloc --cca-record-stream` | Clean | `record_stream()` is the fix |
| A16 | `PYTORCH_NO_CUDA_MEMORY_CACHING=1` + A14 | Clean | CCA is the mechanism |
| A17 | A14 + `--aql-queue-size 1024` | Clean | Backpressure masks race |
| A18 | A14 + `--cca-num-pressure-tensors 8` | NaN, earlier | More CCA pressure |
| A19 | `TORCH_CUDA_SANITIZER=1` + A14 + `--verify 50` | CSAN report | CSAN detects race |

### Commands

**A14 -- CCA race reproduce:**

```bash
torchrun --nproc_per_node=2 -m aorta.race \
    --config config/race/eval_exp_a_dlrmv3.yaml \
    --cca-cross-stream-alloc --no-cca-record-stream
```

**A15 -- CCA mitigated with record_stream:**

```bash
torchrun --nproc_per_node=2 -m aorta.race \
    --config config/race/eval_exp_a_dlrmv3.yaml \
    --cca-cross-stream-alloc --cca-record-stream
```

**A16 -- CCA mitigated by disabling CCA:**

```bash
PYTORCH_NO_CUDA_MEMORY_CACHING=1 torchrun --nproc_per_node=2 -m aorta.race \
    --config config/race/eval_exp_a_dlrmv3.yaml \
    --cca-cross-stream-alloc --no-cca-record-stream
```

**A17 -- CCA race with AQL=1024:**

```bash
torchrun --nproc_per_node=2 -m aorta.race \
    --config config/race/eval_exp_a_dlrmv3.yaml \
    --cca-cross-stream-alloc --no-cca-record-stream --aql-queue-size 1024
```

**A18 -- CCA race with extra pressure:**

```bash
torchrun --nproc_per_node=2 -m aorta.race \
    --config config/race/eval_exp_a_dlrmv3.yaml \
    --cca-cross-stream-alloc --no-cca-record-stream \
    --cca-num-pressure-tensors 8
```

**A19 -- CCA race with CSAN:**

```bash
TORCH_CUDA_SANITIZER=1 torchrun --nproc_per_node=2 -m aorta.race \
    --config config/race/eval_exp_a_dlrmv3.yaml \
    --cca-cross-stream-alloc --no-cca-record-stream --verify 50
```

### CCA Calibrated Configs (Optimized T_cpu/T_gpu Ratio)

The calibrated configs use 5-layer DLRMv3 with `seq_len=500` to achieve **T_cpu << T_gpu**:

- `seq_len=500` → ~80ms GPU compute (O(S^2) attention)
- 5 attention layers → ~2ms CPU dispatch time
- T_cpu/T_gpu ratio: ~0.025 (CPU drops refs while GPU is 2.5% through forward)

The key insight: `seq_len` controls GPU compute without increasing CPU dispatch count. More layers increase both T_cpu and T_gpu equally (each layer adds ~3.3ms CPU dispatch), but `seq_len` only increases T_gpu.

| ID | Config | torch.compile | Purpose |
|----|--------|---------------|---------|
| CCA-1 | `eval_exp_cca_calibrated.yaml` | OFF | Primary: widest intra-iteration race window |
| CCA-2 | `eval_exp_cca_calibrated_compile.yaml` | ON | Secondary: test if compile masks or triggers |

**CCA-1 -- Calibrated (compile OFF, primary):**

```bash
GPU_MAX_HW_QUEUES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --nproc_per_node=2 -m aorta.race \
    --config config/race/eval_exp_cca_calibrated.yaml
```

**CCA-2 -- Calibrated (compile ON):**

```bash
GPU_MAX_HW_QUEUES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --nproc_per_node=2 -m aorta.race \
    --config config/race/eval_exp_cca_calibrated_compile.yaml
```

**Numeric stability baseline** (run first to confirm model doesn't NaN without pipelining):

```bash
GPU_MAX_HW_QUEUES=2 torchrun --nproc_per_node=2 -m aorta.race \
    --config config/race/eval_exp_cca_calibrated.yaml --disable-pipelining
```

**Discriminators** (run after NaN reproduces):

```bash
# CCA involved?
PYTORCH_NO_CUDA_MEMORY_CACHING=1 ... --config config/race/eval_exp_cca_calibrated.yaml

# record_stream is the fix?
... --config config/race/eval_exp_cca_calibrated.yaml --cca-record-stream

# hipEventQuery premature?
... --config config/race/eval_exp_cca_calibrated.yaml --cca-event-sync

# Pipelining is the trigger?
... --config config/race/eval_exp_cca_calibrated.yaml --disable-pipelining
```

### CCA Decision Tree

```
A14: CCA cross-stream alloc + no record_stream → NaN?
  YES → Confirmed: CCA missing-record_stream is the root cause
    │
    ├── A15: Add record_stream → Clean?
    │     YES → record_stream is the fix. Add record_stream() in the TorchRec pipeline.
    │     NO  → Premature hipEventQuery (premature hipEventQuery hypothesis)
    │
    ├── A16: Disable CCA entirely → Clean?
    │     YES → CCA is definitely involved
    │     NO  → Bug is lower than CCA
    │
    └── A17: AQL=1024 → Clean?
          YES → Backpressure masks the CCA bug (matches Issue A)
          NO  → CCA race is independent of queue depth

  NO → CCA cross-stream alloc alone doesn't trigger
    │
    ├── A18: Add pressure tensors → NaN?
    │     YES → Memory pressure was needed to force CCA recycling
    │     NO  → hipEventQuery premature completion may be needed,
    │           or the specific TorchRec KJT tensor lifecycle is required
```

### Why Existing Mitigations Work

| Mitigation | Why It Prevents the Race |
|---|---|
| `ROC_AQL_QUEUE_SIZE=1024` | Backpressure limits CPU lead → by the time CCA recycles, default_stream work has completed |
| Disabling side streams | No cross-stream allocation → CCA doesn't need cross-stream events |
| `torch.cuda.synchronize()` | Drains all streams → CCA events are genuinely complete |
| `GPU_MAX_HW_QUEUES=2` | Reduces parallelism → GPU keeps up → smaller race window |
| `NCCL_LAUNCH_ORDER_IMPLICIT=1` | Serializes RCCL → reduces cross-stream CCA event complexity |
| Disabling pipelining | No cross-iteration buffer sharing → no cross-stream tensor lifecycle |

### Note on record_stream()

`record_stream()` is purely CPU-side bookkeeping -- zero GPU synchronization:

- **At call time:** one `set.insert()` on the CPU
- **At free time:** one extra `hipEventRecord()` per additional stream (non-blocking)
- **At reuse time:** one extra `hipEventQuery()` poll per additional stream

The pipeline remains fully asynchronous. `wait_stream()` and `record_stream()` serve completely different purposes:

- **`wait_stream()`**: GPU scheduling -- "don't start stream B until stream A finishes"
- **`record_stream()`**: Memory management -- "this tensor is also used on stream B"

---

## Experiment B: Large Batch + Pipelining NaN

### What Happens

At batch_size >= 1024, NaN persists **even when Experiment A is fully mitigated** (AQL=1024, `torch.cuda.synchronize()` at every pipeline point). Queue depth is zero and NaN still appears. Disabling pipelining eliminates NaN at any batch size.

### How to Identify

- NaN from the very beginning (not after ~350 iters)
- AQL=1024 does NOT fix it
- Full sync at all pipeline points does NOT fix it
- Disabling pipelining DOES fix it

### Tests -- Reproduction and Control

| ID | Config / Command | Expected | What It Proves |
|----|-----------------|----------|----------------|
| B1 | `--config config/race/eval_exp_b_reproduce.yaml` | NaN from start | Confirms Experiment B |
| B2 | `--config config/race/eval_exp_b_control.yaml` | Clean | Pipelining is trigger |
| B3 | `--mode eval_pipelined --batch-size 512 --use-compile --aql-queue-size 1024 --sync-policy all_pipeline_points --verify 500` | Clean | Batch size matters |

### Tests -- Hypothesis Isolation

Run in order. Each test isolates one subsystem.

| ID | Command | If NaN | If Clean |
|----|---------|--------|----------|
| B4 | `nproc=1`, bs=4096, AQL=1024, sync=all_pipeline_points | RCCL not involved | RCCL involved → B6 |
| B5 | B1 config + `--disable-sdma` | Not SDMA → B7 | SDMA engine bug |
| B6 | B1 config + `--nccl-implicit-order` | Deeper than RCCL → B5,B7 | RCCL signaling bug |
| B7 | bs=4096, `--no-compile`, AQL=1024, sync=all | Not compile → B8 | torch.compile codegen bug |
| B8 | bs=4096, `--fresh-buffers --disable-pipelining`, AQL=1024 | Not address reuse → B9 | Stale cache lines |
| B9 | `PYTORCH_NO_CUDA_MEMORY_CACHING=1`, bs=4096, AQL=1024, sync=all | Not allocator | Allocator recycles early |
| B10 | `TORCH_CUDA_SANITIZER=1` + B1 config | N/A | Cross-stream access? |

### Commands

**B1 -- Reproduce:**

```bash
torchrun --nproc_per_node=2 -m aorta.race \
    --config config/race/eval_exp_b_reproduce.yaml
```

**B2 -- Control (no pipelining):**

```bash
torchrun --nproc_per_node=2 -m aorta.race \
    --config config/race/eval_exp_b_control.yaml
```

**B4 -- 1 GPU:**

```bash
torchrun --nproc_per_node=1 -m aorta.race --mode eval_pipelined \
    --batch-size 4096 --use-compile --aql-queue-size 1024 \
    --sync-policy all_pipeline_points --verify 100
```

**B5 -- No SDMA:**

```bash
torchrun --nproc_per_node=2 -m aorta.race \
    --config config/race/eval_exp_b_reproduce.yaml --disable-sdma
```

**B6 -- RCCL serialization:**

```bash
torchrun --nproc_per_node=2 -m aorta.race \
    --config config/race/eval_exp_b_reproduce.yaml --nccl-implicit-order
```

**B7 -- No torch.compile:**

```bash
torchrun --nproc_per_node=2 -m aorta.race --mode eval_pipelined \
    --batch-size 4096 --no-compile --aql-queue-size 1024 \
    --sync-policy all_pipeline_points --verify 100
```

**B8 -- Fresh buffers:**

```bash
torchrun --nproc_per_node=2 -m aorta.race --mode eval_pipelined \
    --batch-size 4096 --use-compile --fresh-buffers --disable-pipelining \
    --aql-queue-size 1024 --verify 100
```

**B9 -- No caching allocator:**

```bash
PYTORCH_NO_CUDA_MEMORY_CACHING=1 torchrun --nproc_per_node=2 -m aorta.race \
    --mode eval_pipelined --batch-size 4096 --use-compile \
    --aql-queue-size 1024 --sync-policy all_pipeline_points --verify 100
```

### Decision Tree

```
B1: Reproduce NaN at bs=4096?
  NO  → Cannot reproduce, check environment
  YES → Continue

B4: 1 GPU (no collectives), still NaN?
  YES → RCCL not involved
    │
    ├── B5: HSA_ENABLE_SDMA=0, still NaN?
    │     YES → Not SDMA. Try B7 (compile), B8 (buffers), B9 (allocator)
    │     NO  → SDMA engine bug. Report to HIP/ROCr team.
    │
  NO  → RCCL is involved
    │
    ├── B6: NCCL_LAUNCH_ORDER_IMPLICIT + CONTEXT_TRACKING, still NaN?
    │     YES → Deeper than RCCL ordering. Try B5, B7, B8.
    │     NO  → RCCL signaling bug. Report to RCCL team.
```

### Recommended Test Order

1. **B4 (1 GPU)** -- takes seconds, immediately tells you if RCCL is involved
2. **B5 (no SDMA)** -- runtime env var, no rebuild needed
3. Based on results:
   - If B4 shows NaN (RCCL not involved) → focus on B5, B7, B8, B9
   - If B4 is clean (RCCL involved) → focus on B6

---

## Sync Policies

The `--sync-policy` flag controls how often the CPU synchronizes with the GPU:

| Policy | Behavior | Use Case |
|--------|----------|----------|
| `none` | Zero sync in the loop. CPU races ahead freely. | Experiment A reproduction |
| `end_only` | Sync only after all iterations complete. | Default; checks NaN at end |
| `periodic` | Sync every `--nan-check-interval` iterations. | Approximate NaN iteration |
| `every_iter` | Sync after each iteration. | Baseline; pinpoints exact iteration |
| `all_pipeline_points` | Sync at every stream interaction point. | Experiment B (proves not timing) |

---

## eval_pipelined CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--batch-size N` | 512 | Batch size |
| `--feature-dim N` | 256 | Input feature dimension |
| `--hidden-dim N` | 1024 | Hidden MLP dimension |
| `--model-layers N` | 4 | Hidden MLP layers |
| `--model-type TYPE` | `mlp` | `mlp`, `dlrm`, or `dlrm_v3` |
| `--use-compile` | off | Apply `torch.compile` |
| `--no-compile` | - | Explicitly disable compile |
| `--disable-pipelining` | - | Each iteration independent |
| `--disable-datadist` | - | All work on default stream |
| `--disable-metrics` | - | Skip metric simulation |
| `--sync-policy POLICY` | `end_only` | See table above |
| `--nan-check-interval N` | 50 | For `periodic` policy |
| `--embed-tensor-size N` | 500000 | Datadist tensor size |
| `--fresh-buffers` | off | New GPU buffers each iter |
| `--gpu-padding-dispatches N` | 0 | Extra AQL queue fill dispatches |
| `--seq-len N` | 200 | Attention seq length (dlrm_v3) |
| `--use-bfloat16` | off | bfloat16 autocast |
| `--pre-generate-pool-size N` | auto | CPU batch pool size |
| `--aql-queue-size N` | - | Set `ROC_AQL_QUEUE_SIZE` |
| `--config PATH` | - | YAML config file |

### CCA Cross-Stream Flags

| Flag | Description |
|------|-------------|
| `--cca-cross-stream-alloc` | Dynamic allocation on side streams each iteration |
| `--no-cca-record-stream` | Skip `record_stream()` (reproduces the bug) |
| `--cca-record-stream` | Call `record_stream()` (the mitigation) |
| `--cca-num-pressure-tensors N` | Extra tensors for CCA recycling pressure |
| `--cca-integrity-check` | GPU-side checksum verification |
| `--cca-event-sync` | `event.synchronize()` after `event.query()` success |

### DLRMv3 / HSTU Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--hstu-num-layers N` | 5 | HSTU attention layers |
| `--hstu-num-heads N` | 4 | Attention heads per layer |
| `--embedding-dim N` | 128 | Attention embedding dimension |
| `--seq-len N` | 200 | Sequence length (controls GPU work) |

---

## YAML Config Presets

All presets are in `config/race/`:

| File | Experiment | What It Does |
|------|-----------|--------------|
| `eval_exp_a_reproduce.yaml` | A | bs=512, DLRM, no sync, AQL=16K |
| `eval_exp_a_mitigated.yaml` | A | bs=512, DLRM, no sync, AQL=1024 |
| `eval_exp_a_dlrmv3.yaml` | A | bs=512, DLRMv3 (seq_len=200), no sync, AQL=16K |
| `eval_exp_b_reproduce.yaml` | B | bs=4096, DLRM, sync all points, AQL=1024 |
| `eval_exp_b_control.yaml` | B | bs=4096, DLRM, pipelining disabled, AQL=1024 |
| `eval_exp_cca_calibrated.yaml` | CCA | bs=512, DLRMv3 (seq_len=500), compile OFF, CCA race, integrity check |
| `eval_exp_cca_calibrated_compile.yaml` | CCA | Same as above with compile ON |

---

## Key Environment Variables

| Variable | What It Does | How to Set |
|----------|-------------|------------|
| `ROC_AQL_QUEUE_SIZE` | AQL queue depth (default 16384) | `--aql-queue-size 1024` or YAML |
| `GPU_MAX_HW_QUEUES` | HW queue count (default 4) | `--hw-queues 2` |
| `HSA_ENABLE_SDMA` | SDMA engine (default 1) | `--disable-sdma` |
| `NCCL_LAUNCH_ORDER_IMPLICIT` | RCCL serialization | `--nccl-implicit-order` (sets both) |
| `RCCL_ENABLE_CONTEXT_TRACKING` | RCCL context tracking | `--nccl-implicit-order` (sets both) |
| `PYTORCH_NO_CUDA_MEMORY_CACHING` | Disable CCA | Set manually |
| `TORCH_CUDA_SANITIZER` | Stream sanitizer | Set manually |
| `PYTORCH_CUDA_ALLOC_CONF` | Allocator config | Set manually (e.g., `expandable_segments:True`) |

---

## Multi-Node Launch

Make sure the docker container is running on all nodes. Then from the head node:

```bash
# Experiment A
./scripts/multi_node/launch_reproducer.sh \
    --docker training-rocm72-torch2.12-shampoo \
    --config config/race/eval_exp_a_reproduce.yaml

# Experiment B
./scripts/multi_node/launch_reproducer.sh \
    --docker training-rocm72-torch2.12-shampoo \
    --config config/race/eval_exp_b_reproduce.yaml

# Experiment B with SDMA disabled
./scripts/multi_node/launch_reproducer.sh \
    --docker training-rocm72-torch2.12-shampoo \
    --config config/race/eval_exp_b_reproduce.yaml \
    --disable-sdma

# Experiment B with RCCL serialization
./scripts/multi_node/launch_reproducer.sh \
    --docker training-rocm72-torch2.12-shampoo \
    --config config/race/eval_exp_b_reproduce.yaml \
    --nccl-implicit
```

Nodes are defined in `scripts/multi_node/node_ip_list.txt`.

---

## DLRMv3 (HSTU) Model Architecture

```
seq_embeddings (B, seq_len, feature_dim) ──→ pre_attn_proj ──→ (B, seq_len, embed_dim)
                                                                        │
dense_features (B, feature_dim) ──→ bottom_mlp ──→ (B, embed_dim) ──→ inject at pos[0]
                                                                        │
datadist_shard (shard_size,) ──→ datadist_proj ──→ (B, feature_dim) ──→ add to dense
                                                                        │
                                                          N × causal HSTU attention layers
                                                                        │
                                                                 mean pool → pred_head → logit
```

- No GPU-side embedding tables (sparse lookups happen on CPU)
- `torch.compile` compatible (standard PyTorch ops only)
- Compute-bound via O(seq_len^2) attention
- Raw logit output for maximum corruption sensitivity
