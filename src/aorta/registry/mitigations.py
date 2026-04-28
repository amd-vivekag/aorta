"""Mitigations registry: dict literal of name -> env var bundle, plus a lookup helper.

Iteration 1 ships only the built-ins and a synchronous lookup. Entry-point
discovery (`load_mitigations()`, collisions) arrives in iteration 2.
"""

from aorta.registry.errors import UnknownMitigationError

# Only true runtime-level flags belong here — env vars consumed by hipBLASLt or
# the ROCm runtime, transparent to the workload. Workload-internal env vars
# (e.g. AMP_DTYPE, SHAMPOO_PRECONDITIONER_DTYPE) only "work" if the workload's
# training script literally reads os.environ for them; those belong with the
# workload's own package, registered via the `aorta.mitigations` entry-point group.
BUILTIN_MITIGATIONS: dict[str, dict[str, str]] = {
    "none":     {},
    "tf32_off": {"DISABLE_TF32": "1"},  # consumed by hipBLASLt itself
    "xnack":    {"HSA_XNACK": "1"},     # consumed by ROCm runtime
}


def get_mitigation(name: str) -> dict[str, str]:
    """Return the env-var bundle for a mitigation name. Empty dict for 'none'.

    Returns a defensive copy — mutating the result does not affect the registry.

    Iteration 1 reads only built-ins. Iteration 2 routes through
    `load_mitigations()` to also see entry-point-registered plugins.
    """
    if name not in BUILTIN_MITIGATIONS:
        raise UnknownMitigationError(
            f"unknown mitigation '{name}'; available: {sorted(BUILTIN_MITIGATIONS)}; "
            f"if you expected a plugin-contributed entry, ensure the plugin is installed"
        )
    return dict(BUILTIN_MITIGATIONS[name])
