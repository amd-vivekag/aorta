# LLM Determinism Recipe

War-room helper for catching kernel-level nondeterminism / silent data
corruption in a transformer training step. Runs the **same** forward+
backward twice on the same inputs, with parameters and RNG restored
between runs, and compares bit-exact checksums of:

- every per-block boundary activation (entering and leaving each
  repetition of the single repeated block — the FSDP2 comm boundary)
- the final loss and logits
- every grad and every param after backward

If any checksum drifts → some kernel (matmul, attention, reduction,
collective) is producing different bits for the same inputs.

## Why a single repeated block?

The war-room execution model is a chain of repeating
`comm_kernel_i → compute_kernel_i` stages. One repetition of the same
transformer block is one such stage, and under FSDP2 `fully_shard` the
block boundary is also the all-gather / reduce-scatter boundary. So
checksumming the activation at every block boundary checksums the
tensor that actually crosses GPUs at every repetition.

This is **not a faithful LLM**. Weights are random init, the specific
architecture (LayerNorm + MHA + GLU FFN, optional top-1 MoE) is generic
on purpose. The model and its size are knobs to control how much data
crosses GPU boundaries, nothing more.

## What it doesn't do

- Not a quality benchmark.
- Not a perf benchmark — manual attention, no `torch.compile`, no graphs.
- Not a multi-step trainer by default; `steps` is configurable but each
  step is a self-contained replay.
- Does not intercept kernels directly. Module forward pre/post hooks are
  the practical proxy for the comm boundary; documented limitation.

## Running

```bash
# 40-GPU capture run (adjust to your cluster shape):
torchrun --nproc_per_node=8 --nnodes=5 -m aorta.cli run \
  --workload llm_determinism \
  --recipe recipes/example-llm-determinism.yaml

# Single rank / local dev:
python -m aorta.cli run --workload llm_determinism
```

Pass → every per-rank per-block (and per-param) checksum matched.
Fail → workload prints which checksums diverged on which rank and which
block index.

## Knobs

| Key | Default | Notes |
|---|---|---|
| `num_layers` | 24 | Block repetitions = comm/compute stages. Drop to scale down with rank count. |
| `hidden_size` | 2048 | Per-block hidden width. |
| `ffn_size` | 5632 | GLU FFN width. |
| `num_heads` | 16 | Must divide `hidden_size`. |
| `vocab_size` | 32000 | Random embedding table. |
| `seq_len` | 512 | Per-rank batch sequence length. |
| `batch_size` | 1 | Per-rank batch. |
| `num_experts` | 1 | 1 = dense FFN; ≥2 = top-1 MoE router across N experts. |
| `dtype` | `bf16` | `bf16` / `fp16` / `fp32`. |
| `seed` | 1234 | Synthetic batch + RNG seed. |
| `steps` | 1 | Number of independent replay steps. |
| `checksum_mode` | `per_rank` | `global` also all-reduces a fingerprint. |
| `capture_dir` | unset | When set, every step writes `rank<NNN>.jsonl` for offline inspection. |
| `model_label` | `generic-repeated-block` | Advisory only; recorded in metrics. |

## Rule-#4 compliance

By default (`checksum_mode: per_rank`) each rank only compares its own
run-1 vs run-2 shards / block boundaries. Pass/fail is OR'd across ranks
via a single all-reduced integer flag. `checksum_mode: global` opts in
to additionally comparing an all-reduced loss⊕output fingerprint, which
catches collective ordering drift but no longer satisfies the strict
"single-rank only" guard.

## Determinism caveats

`torch.use_deterministic_algorithms(True, warn_only=True)` is enabled at
setup. Some ops have no deterministic implementation; we tolerate that
because the checksum compare is what actually proves replayability.
`CUBLAS_WORKSPACE_CONFIG=:4096:8` is set if not already present —
hipBLAS / cuBLAS require this for deterministic matmul.

## Capture-mode output

When `capture_dir` is set, each rank writes one line per (run, step):

```json
{"rank": 7, "step": 0, "run": "r1", "loss_bits": 12345, "output_bits": 67890,
 "block_pre": [<int per block>], "block_post": [<int per block>]}
```

This is the raw material for offline analysis on the 40-GPU capture run —
e.g. `jq` for ranks whose `block_pre[i]` differs between `r1` and `r2`,
or a comparison of the same rank's checksums across two whole captures
collected on different nodes / dates.

## Metrics surface (stable for detector parsing)

`WorkloadResult.metrics` includes:

- `ranks_with_divergence`
- `steps[*].loss_bits_r1`, `loss_bits_r2`, `num_blocks_checked`
- `divergence_reasons` (first 32, when present)
- `model_label`, `num_layers`, `num_experts`, `dtype`, `checksum_mode`,
  `capture_dir`
