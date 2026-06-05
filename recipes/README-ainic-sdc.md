# AINIC RCCL SDC reproducer — quick start

How to run the `race` workload's AINIC silent-data-corruption recipe
(`ainic-gdr-flush-sdc.yaml`) on a multi-node GPU cluster, plus a fast smoke
test to sanity-check your setup first.

## What it does

The `race` workload (`mode: fsdp`) simulates the FSDP communication pattern with
**explicit** `all_gather` / `reduce_scatter` collectives over the fabric, running
a real transformer block as the compute kernel between them. Every layer uses one
shared block on a fixed input, so all layers must produce byte-identical output —
a per-layer checksum then localizes any corruption to **comm** (RCCL/NIC) vs
**compute** (GPU). It also verifies a rank-fill ground truth (after `all_gather`,
chunk *j* must equal `float(j)`; `reduce_scatter` sum must equal `sum(1..N)`).

The recipe runs 5 cells that are **identical except for their NCCL env vars** —
so any difference in corruption is attributable to the flag:

| Cell | Key env | Expected |
|---|---|---|
| `baseline-no-gdr-flush` | `NCCL_GDR_FLUSH_DISABLE=1`, `NCCL_PROTO=Simple` | corruption exposed |
| `gdr-flush` | `NCCL_GDR_FLUSH_DISABLE=0` (+ strict flush) | clean (primary fix) |
| `strict-pcie-ordering` | `NCCL_IB_PCI_RELAXED_ORDERING=0`, flush off | likely still corrupts |
| `gdr-flush-and-strict-pcie` | flush on + PCIe strict | clean |
| `ll-protocol-control` | `NCCL_PROTO=LL`, flush off | clean (LL avoids GDRDMA) |

## 1. Install

In an environment that already has a working PyTorch (ROCm or CUDA) build:

```bash
pip install -e .
aorta --version
```

`race` registers via the `aorta.workloads` entry point — no extra wiring.

## 2. CPU smoke test (no GPU needed)

Confirm the code is intact on any machine, including a laptop:

```bash
python -m pytest tests/workloads/test_race.py \
                 tests/workloads/test_race_checksums.py \
                 tests/workloads/test_race_transformer_smoke.py \
                 tests/workloads/test_race_real_backward.py -v
```

## 3. Dry-run the recipe (no cluster)

```bash
aorta triage run --recipe recipes/ainic-gdr-flush-sdc.yaml --dry-run
```

## 4. Fast cluster smoke test (seconds, not the full matrix)

Before the full run, confirm the end-to-end path works on YOUR cluster with a
tiny recipe (1 cell, 1 trial, 5 iters, smaller model — same real code path):

```bash
# one rank per GPU; run the SAME command on every rank via your launcher
torchrun --nnodes=<N> --nproc-per-node=<GPUS_PER_NODE> \
  --rdzv-backend=c10d --rdzv-endpoint=<HEAD_HOST>:<PORT> \
  $(command -v aorta) triage run --recipe recipes/ainic-smoke.yaml
```

A green smoke run shows, in `triage_results/AINIC-SMOKE/race/<ts>/matrix.json`
(and per-trial JSON), metrics that prove the real path ran:

- `compute_type: transformer` — real transformer compute, not a GEMM fallback
- `layers_verified > 0` — the per-layer checksum detector actually executed
- `layer_checksum_mismatches: 0` — clean
- `avg_step_time_ms > 0` — real compute (not a 0 ms no-op)
- `exit_status: ok`, `passed: true`

If those look right, run the full recipe.

## 5. Run the full recipe on the cluster

```bash
torchrun --nnodes=<N> --nproc-per-node=<GPUS_PER_NODE> \
  --rdzv-backend=c10d --rdzv-endpoint=<HEAD_HOST>:<PORT> \
  $(command -v aorta) triage run --recipe recipes/ainic-gdr-flush-sdc.yaml
```

Launch contract (read once):

- The workload is `launch_mode = "distributed"`, `min_world_size = 2`. `setup()`
  calls `dist.init_process_group(backend="nccl")` with no args, so it reads the
  standard env contract: `RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT`,
  `LOCAL_RANK` (used to bind the GPU). Any launcher that sets these works
  (torchrun shown; under Slurm drive the same `aorta triage run` line via
  `srun`/`torchrun`).
- Run the **same** `aorta triage run` command on every rank. Only rank 0 writes
  the result artifacts; other ranks participate in the collectives.
- The recipe's per-cell NCCL env vars are applied by the runner automatically —
  do **not** set them yourself.
- The original repro is documented at ~22 ranks, 1 rank per host. The code runs
  at any `WORLD_SIZE >= 2`; scale/topology is your launcher's responsibility.

## 6. Read results

```bash
cat triage_results/AINIC-SDC-001/race/<timestamp>/matrix.md
```

`triage_results/AINIC-SDC-001/race/<timestamp>/` contains `matrix.md` (summary
table), `matrix.json` (full per-cell stats), `recipe.resolved.yaml`, and
`cells/<cell>/race/trial_*.json` (per-trial detail).

A real corruption hit shows `passed: false` with `failure_details[*].type`:
- `layer_checksum_mismatch_compute_output` → GPU compute path
- `layer_checksum_mismatch_comm_output` → RCCL / NIC collective path

A defensible result: `baseline-no-gdr-flush` FAIL, `gdr-flush` PASS,
`ll-protocol-control` PASS.

## Notes / limitations

- This **simulates** FSDP (explicit collectives) — it is not
  `torch.distributed.fsdp`, and shards are synthetic rank-filled buffers, not
  real model parameters. It exercises the real collective data path and real
  transformer compute, and only flags corruption if RCCL actually returns wrong
  bytes under a cell's env.
- The iter-0 / cold-QP precondition is hit only on the run's first `all_gather`
  (the launcher's process group is reused across iterations).
- `real_backward: true` (set in the recipe) runs genuine backward gradient
  kernels; set it to `false` for a cheaper forward-rerun timing proxy.
