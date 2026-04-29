# `aorta.registry` — mitigations + environments

Two small registries that ship with aorta:

- **Mitigations** (`name → env vars`) — process-level flags applied just before
  the workload subprocess launches. Examples: `tf32_off`, `xnack`.
- **Environments** (`name → docker / venv recipe`) — baseline state of the
  process or container the workload runs in. Examples: `local`, `default`.

Both follow the same shape: built-ins ship from this package; external
contributions arrive via Python entry-points and are merged at runtime.

## Adding a mitigation

There are two paths. **For most contributions, Path 2 (plugin) is the right
answer** — Path 1 is reserved for runtime-level flags that benefit every
workload.

### Path 1 — built-in (runtime-level flags only)

Built-ins are for env vars read by a **runtime or library** — ROCm, hipBLASLt,
PyTorch, NCCL, OpenMP, the Linux kernel — **not** by workload Python code.

The defining property: setting the var produces the same effect regardless of
which workload runs. If your workload's training script has to call
`os.environ.get(...)` for the var to take effect, it is NOT a built-in
candidate — it's a plugin candidate (Path 2).

| Qualifies as built-in | Does NOT qualify (use Path 2) |
|---|---|
| `DISABLE_TF32` (hipBLASLt reads it) | `AMP_DTYPE` (only the workload's Python reads it) |
| `HSA_XNACK` (ROCm runtime reads it) | `SHAMPOO_PRECONDITIONER_DTYPE` (only meta-recom reads it) |
| `CUDA_LAUNCH_BLOCKING` (PyTorch reads it) | Any custom flag your workload introspects |
| `NCCL_DEBUG`, `OMP_NUM_THREADS`, `LD_PRELOAD` | Anything that's a silent no-op on workloads that don't read it |

If a Path 1 candidate qualifies, edit `src/aorta/registry/mitigations.py` and
add one entry to the dict:

```python
BUILTIN_MITIGATIONS = {
    "none":     {},
    "tf32_off": {"DISABLE_TF32": "1"},
    "xnack":    {"HSA_XNACK": "1"},
    "no_sdma":  {"HSA_ENABLE_SDMA": "0"},   # <-- your addition
}
```

Open a PR. Adding a built-in requires sign-off from an aorta core maintainer —
the bar is "would this benefit any aorta user, on any workload?".

> **No need to clone:** you can do this entirely in the GitHub web UI. Navigate
> to `src/aorta/registry/mitigations.py`, click the pencil icon, edit the dict,
> scroll down, fill in a PR title, click "Propose changes". Done.

### Path 2 — plugin (everything else)

For workload-coupled flags, customer-specific recipes, experimental things, or
anything where you maintain your own Python package: ship the mitigation from
your package via the `aorta.mitigations` entry-point group.

In your package's `pyproject.toml`:

```toml
[project.entry-points."aorta.mitigations"]
my_mits = "my_package.mitigations:get_all"
```

In your package's source:

```python
# my_package/mitigations.py
def get_all() -> dict[str, dict[str, str]]:
    return {
        "my_flag": {"MY_ENV_VAR": "1"},
    }
```

After `pip install` (or `pip install -e .`), your mitigation appears in
`aorta mitigations list` tagged with your package name.

> Plugin authors: re-run `pip install` after editing `pyproject.toml` —
> entry-points are read at install time, not at import time.

## Trying things locally without upstreaming

Sometimes you want to test a mitigation on your own machine without sending a
PR or publishing a package. Two options:

### One-off — CLI override

For "is this even worth pursuing" experiments:

```bash
aorta run --workload fsdp --env HSA_ENABLE_SDMA=0
```

No registry involvement. The flag isn't named, isn't tracked, doesn't appear
in `aorta mitigations list`. Doesn't work for triage sweeps (which need
named entries).

### Repeated local use — throwaway plugin package

Make a tiny folder anywhere:

```
my-local-mits/
├── pyproject.toml
└── my_mits.py
```

`pyproject.toml`:

```toml
[project]
name = "my-local-mits"
version = "0.1"

[project.entry-points."aorta.mitigations"]
local = "my_mits:get_all"
```

`my_mits.py`:

```python
def get_all():
    return {
        "no_sdma": {"HSA_ENABLE_SDMA": "0"},
    }
```

Then once:

```bash
pip install -e my-local-mits/
```

After that, your mitigations work everywhere — `--mitigation` flags, triage
sweeps, the lot. Nothing leaves your machine. To remove: `pip uninstall
my-local-mits`.

This is just Path 2 with a throwaway local package. Same workflow applies to
workloads via the `aorta.workloads` entry-point group.

## Adding an environment

Same two paths, with `aorta.environments` as the entry-point group. Plugin
payloads must use only the keys `docker` and `venv`; anything else (e.g.
`rocm`) raises `RegistryError` at load time. ROCm version is implicit in
the docker image digest or in the host the venv runs on — capture it from
`aorta env probe` at runtime, not via static declaration.

## Collisions

If two contributors register the same name (built-in vs plugin, or plugin vs
plugin), `load_mitigations()` / `load_environments()` raises
`RegistryCollisionError` naming both packages. There is no winner — the human
resolves it by renaming or removing one of the entries. This is intentional:
silent overrides cause hours of "why does my mitigation behave wrong"
debugging.

## Verifying what's registered

```bash
aorta mitigations list
aorta environments list
```

Each shows every registered entry plus its source package — useful when
debugging "did my plugin actually load?".

## Hard rule: no logic in registries

These modules contain only **data + lookup**. No environment manipulation, no
docker invocation, no validation of whether `DISABLE_TF32=1` is a "good"
value. Logic that consumes the registry data lives in the dispatchers (the
mitigation harness, the workload runtime). Mixing the two would make the
registry untestable in isolation and impossible to mock.
