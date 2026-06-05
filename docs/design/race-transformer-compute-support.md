# Plan — Fully support `compute_type: transformer` + shared-weight checksums (PR #210)

**Branch:** `users/oyazdanb/race-transformer-compute` (off `main`)
**Status:** DRAFT — for review before implementation
**Related:** PR #210 (`users/mycpuorg/shared-weight-transformer-checksums`), task `race__cross-rank-and-iter0-detection.md`

## TL;DR

PR #210's transformer + per-layer-checksum code appears **complete and correctly wired on its own branch** (config field, `setup_buffers` shared-weight path, `_forward_layer` dual dispatch, `_verify_layer_checksums` defined AND called). Yet the 2026-06-05T01-12-30 run — which *accepted* `compute_type: transformer` + `shared_layer_weights: true` (no "ignoring unknown key" warning) — showed **0.0 ms step time and emitted no checksum signal**, so we cannot confirm the path actually executed. The detector is **unobservable on success**: it logs/raises only on mismatch and writes no metric, so a clean run looks identical to a no-op.

**This plan does NOT rewrite the compute path** (the branch already has it). It (1) resolves the did-it-run ambiguity, (2) makes the detector self-evidencing, and (3) hardens config validation so silent fallback is impossible. Work lands as a small change set that can either be merged into #210 or layered on top.

## UPDATE (2026-06-05): PR #210's "transformer" is not a transformer — Option A

Reading the PR branch source: `compute_type: transformer` does **`out = gelu(weight_matrix @ reference_input)`** (`fsdp.py:240-241`) — a single matmul, **no attention, no FFN, no softmax**. The "transformer" label refers to the *checksum scheme*, not the compute. So:
- **Real transformer model?** No (mm+gelu).
- **Real `torch.distributed.fsdp`?** No (simulated shards + REAL explicit `all_gather`/`reduce_scatter`).
- Both `gemm` and `transformer` compute_types are effectively a matmul.

**Consequence for PR D / L2 pressure:** PR #210's transformer does NOT supply real-model L2/HBM pressure (the documented missing ingredient for small-scale repro). A single matmul ≠ a real block's memory traffic. So #210 is *not* the cheap step toward PR D I first assumed.

**Option A (chosen):** make `compute_type: transformer` run a REAL transformer block by borrowing `RepeatedTransformerBlock` from `aorta.models.repeated_block` — the same model `llm_determinism` uses (real MHA + GLU FFN + LayerNorm, torch-only, no deps). This gives real L2/HBM pressure *inside* the existing race harness (explicit collectives preserved), without the full `torch.distributed.fsdp` refactor that is PR D. It is the recommended-for-PR-D model per the task file, used here as a compute kernel rather than under FSDP2.

### Option A — implementation spec (source-grounded; refs on PR #210 branch unless noted)

**Compute site:** `_forward_layer` shared path `fsdp.py:240-241` (fwd), `_backward_layer` `fsdp.py:279-281` (bwd). The `full_param` (all_gather output) is only *checksummed*, never fed to compute — keep that true.

**Shape adaptation:** current compute is 2D `[dim,dim] @ [dim,dim]`. `RepeatedTransformerBlock.forward` (`repeated_block.py:108`) needs 3D `[batch, seq, hidden]`, `hidden==model_dim`, `hidden % num_heads == 0`. So `reference_input` becomes 3D `[batch, seq, model_dim]`.

**Shared-weight invariant (the crux):** PR #210 already shares ONE weight across layers — `self.weight_matrices = [shared_w] * num_layers` (`fsdp.py:148-153`), `reference_input` seed 1. To keep "all layers byte-identical" with a real block, build **ONE** `RepeatedTransformerBlock` (not `RepeatedBlockModel` — we want a single block, no embed/LM-head) under a fixed seed and call it for every layer:
```python
# in setup_buffers, transformer+shared path
with torch.random.fork_rng(devices=["cuda"]):
    torch.cuda.manual_seed(0)                     # identical weights on EVERY rank
    self.shared_block = RepeatedTransformerBlock(block_cfg).to("cuda").to(self.dtype)
self.shared_block.eval()
g = torch.Generator(device="cuda"); g.manual_seed(1)
self.reference_input = torch.randn(batch, seq, model_dim, dtype=self.dtype, device="cuda", generator=g)
```
In `_forward_layer` shared branch replace mm+gelu with `out = self.shared_block(self.reference_input)` under `torch.no_grad()`. The 4 checksums map unchanged: `comm_input`=param shard, `comm_output`=full_param post-all_gather, `compute_input`=`_checksum(reference_input)` (`_checksum` `fsdp.py:200` is shape-agnostic), `compute_output`=`_checksum(out)`.

**Config mapping → `BlockConfig`:** `hidden_size <- model_dim`; build a single block (loop num_layers ourselves). **New `ReproducerConfig` fields** (`config.py` transformer section ~line 158): `num_heads: int = 16` (validate `model_dim % num_heads == 0`, else BlockConfig raises `repeated_block.py:44`), `ffn_size: int = 0` (0 ⇒ derive `4*model_dim`), `seq_len: int = 512`, `batch_size: int = 1`. **Reused:** `model_dim`, `num_layers`, `dtype`, `shared_layer_weights`, `compute_type`, `simulate_compute`, `include_backward_compute`.

**Backward:** current is manual `mm(W.T, grad)` (`fsdp.py:279-281`), NOT autograd; `reference_input.requires_grad=False`. **Keep backward as a timing-only proxy** — when `include_backward_compute`, call `self.shared_block(self.reference_input)` a second time under `no_grad`. Do NOT add autograd (breaks no-grad determinism, balloons memory, scope creep).

**Stays untouched (exercises AINIC):** all_gather (`fsdp.py:228,272`), reduce_scatter (`:284`), `_fill_patterns`, `_verify_all_gather` (`:439`), `_verify_reduce_scatter` (`:473`) — they read `full_param`/`grad_shard`, never `reference_input`. Only the compute kernel changes.

**Concrete edits:**
- `config.py` ~L158: add `num_heads`, `ffn_size`, `seq_len`, `batch_size`.
- `fsdp.py:30`: `from aorta.models import BlockConfig, RepeatedTransformerBlock`.
- `fsdp.py:80-82` `__init__`: `self.shared_block = None`.
- `fsdp.py:144-159` `setup_buffers`: block construction + 3D reference_input (transformer+shared path); leave non-shared 2D path as-is.
- `fsdp.py:236-251` `_forward_layer` shared branch: real block forward under `no_grad`.
- `fsdp.py:277-281` `_backward_layer`: block-forward proxy on shared path.

**Pitfalls:** `RepeatedTransformerBlock` defaults float32 → `.to(self.dtype)` mandatory (race=bf16). LayerNorm/softmax upcast (`repeated_block.py:119`) is deterministic — fine. Memory: real block activations `[batch,seq,hidden]` + attention `[batch,heads,seq,seq]` ≫ 2D matmul → default `batch_size=1, seq_len=512`; block built once (shared) so param memory ≈ one layer regardless of `num_layers`.

**Preserves corruption localization: YES** — `_verify_layer_checksums` (`fsdp.py:419-436`) is checksum-key-agnostic: comm mismatch ⇒ RCCL/NIC, compute mismatch ⇒ GPU compute, verbatim. Rank-fill checks on `full_param`/`grad_shard` are untouched (LOW risk).

**One real setup risk:** if ranks build the block with diverging RNG, `compute_output` differs *across ranks* but stays identical *across layers within a rank* → intra-rank `_verify_layer_checksums` still PASSES, masking a false setup bug. Mitigate: seed global CUDA RNG identically on all ranks before block construction (the `fork_rng` above), and optionally add a one-time cross-rank `compute_output` all-reduce equality assert at setup.

**Effort:** ~medium / half-day. ~4 edited functions + 4 config fields + 1 import. No new files. The earlier Steps 1–3 (observability, config validation, detector unit test) still apply on top.

## Step 0 — Resolve the contradiction FIRST (no code yet)

Before writing anything, confirm what actually ran on the cluster. On the cluster:
```
cd /it-share/oyazdanb/aorta && git branch --show-current
python - <<'PY'
import inspect, aorta.race.modes.fsdp as f
src = inspect.getsource(f)
for tok in ("_verify_layer_checksums","_checksum","reference_input","use_shared","shared_w"):
    print(tok, tok in src)
PY
```
- All `True` → branch code IS installed; the path is *present* but unobservable → go to Steps 1–3.
- Any `False` → a stale/main build is installed; `pip install -e` didn't take → reinstall, rerun, re-check. (This alone may explain the 0.0 ms / no-metric run.)

Also: 0.0 ms step is suspicious even for GEMM. Confirm `simulate_compute` is actually doing work — a real 24-layer transformer fwd+bwd at model_dim=1024 cannot be 0.0 ms. If it is 0.0 ms with the branch installed, compute is being skipped or mistimed — that's a real bug to find in `_forward_layer`/timing, not just observability.

## Step 1 — Make the detector self-evidencing (observability)

Root problem: a green result does not prove the checksum verifier ran. Fix so it is impossible to miss.

- **Startup log (once per trial):** in `setup_buffers`/`run` when `compute_type=="transformer"` and `shared_layer_weights`, emit e.g.
  `log.info("race: compute=transformer shared_layer_weights=ON layers=%d dim=%d; per-layer checksum verify ENABLED", num_layers, model_dim)`.
  And in the GEMM branch: `log.info("race: compute=gemm")`. Now fallback is one grep away.
- **Result metric (always, not just on failure):** add to `WorkloadResult.metrics` (mapped in `workloads/race.py`):
  - `layers_verified` (int, per step or total),
  - `layer_checksum_mismatches` (int, 0 on clean),
  - `compute_type` (echo the effective value).
  A clean run then shows `layer_checksum_mismatches: 0, layers_verified: >0` — provably ran. Today `metrics` is only `{avg_step_time_ms, mode, rank, world_size}`.
- **Per-step debug counter** behind `log_interval` so long runs show progress.

## Step 2 — Harden config so silent fallback is impossible

- **Validate `compute_type`** against `{"gemm","transformer"}` in `workloads/race.py` (mirror the existing `_VALID_DTYPES` pattern). Today any string is accepted, so a typo (`transfomer`) silently runs GEMM with no warning.
- **Warn on inert combo:** if `shared_layer_weights=True` but `compute_type!="transformer"`, log a WARNING — currently it silently no-ops via the `use_shared` AND-gate.
- Keep accepting the keys (they already round-trip on the branch) — this is validation, not new surface.

## Step 3 — Confirm the checksum actually catches corruption (test the detector)

A detector that never fires on a clean cluster is unproven. Add a unit/integration check:
- **Unit:** forge a layer activation buffer so one layer's checksum differs; assert `_verify_layer_checksums` reports a mismatch with the right layer index. (No GPU/dist needed.)
- **Negative:** identical layers → 0 mismatches.
- This proves the second signal works regardless of whether the AINIC bug reproduces.

## Files

- `src/aorta/race/modes/fsdp.py` — startup log; mismatch counter; ensure `_verify_layer_checksums` returns counts (verified, mismatches) rather than only logging.
- `src/aorta/workloads/race.py` — `_VALID_COMPUTE_TYPES` validation; warn on inert `shared_layer_weights`; map new metrics into `WorkloadResult.metrics`.
- `tests/workloads/test_race.py` (or race-mode test) — checksum-catches-forged-corruption + clean-pass + config-validation cases.
- `recipes/ainic-gdr-flush-sdc.yaml` — already correct on #210 (dtype `bfloat16`, transformer, shared_layer_weights). No change beyond confirming.

## Relationship to PR #210 and PR D

- **PR #210:** this is *complementary* — it makes #210's detector observable, validated, and tested. Decide at review: fold these commits into #210, or land as a follow-up on this branch.
- **PR D (real torch.distributed.fsdp):** **deferred — do NOT start.** Per the task file, the 5-node no-repro triggers **PR C first** (cold-QP/iter-0), and PR D only if real-model L2 pressure is *confirmed* the missing ingredient. PR #210's transformer compute is the cheap way to raise model-realistic L2/HBM pressure *inside* the existing harness — test that hypothesis before any PR D refactor. Also: PR D would **not** close the iter-0/cold-QP gap (both reuse the launcher's warm PG); only PR C does.

## Open questions for review
1. Step 0 result: is the branch actually installed on the cluster (path present) or is the 0.0 ms / no-metric run a stale-build artifact? This decides whether Steps 1–3 are the whole job or whether there's also a real compute-skip bug.
2. Fold into #210 vs land as follow-up branch?
3. Does `_verify_layer_checksums` currently return counts, or only log? (Determines how much of Step 1's metric plumbing is new.)
