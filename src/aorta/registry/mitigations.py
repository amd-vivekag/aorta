"""Mitigations registry: built-ins + entry-point discovery + collision detection.

`load_mitigations()` returns the merged registry of built-ins and plugin
contributions, keyed by name. Each entry carries its `source_package` so
collision errors can name the conflicting parties.

Plugin authors register one entry-point per mitigation in their `pyproject.toml`
under the `aorta.mitigations` group. The entry-point name IS the mitigation
name; the loaded object is the env-var bundle (`dict[str, str]`). This mirrors
the existing `aorta.workloads` extension-point pattern.
"""

from importlib.metadata import entry_points

from aorta.registry.errors import (
    RegistryCollisionError,
    RegistryError,
    UnknownMitigationError,
)
from aorta.registry.types import Mitigation

_GROUP = "aorta.mitigations"

# Only runtime-level flags belong here — env vars read by a runtime or library
# (ROCm, hipBLASLt, PyTorch, NCCL, OpenMP, the kernel, etc.), transparent to
# the workload. Workload-internal env vars (e.g. AMP_DTYPE,
# SHAMPOO_PRECONDITIONER_DTYPE) only "work" if the workload's training script
# literally reads os.environ for them; those belong with the workload's own
# package, registered via the `aorta.mitigations` entry-point group.
# See src/aorta/registry/README.md for the full criterion.
BUILTIN_MITIGATIONS: dict[str, dict[str, str]] = {
    "none":     {},
    "tf32_off": {"DISABLE_TF32": "1"},  # consumed by hipBLASLt itself
    "xnack":    {"HSA_XNACK": "1"},     # consumed by ROCm runtime
}


def load_mitigations() -> dict[str, Mitigation]:
    """Discover and merge all mitigations: built-ins first, then entry-point plugins.

    No caching — re-reads entry-points each call. Cheap for MVP; revisit if profiling
    shows it matters.

    Raises:
        RegistryCollisionError: two contributors registered the same mitigation name.
        RegistryError: a plugin's entry-point payload was not a `dict[str, str]`.
    """
    registry: dict[str, Mitigation] = {
        name: Mitigation(name=name, env=dict(env), source_package="aorta")
        for name, env in BUILTIN_MITIGATIONS.items()
    }

    for ep in entry_points(group=_GROUP):
        env = ep.load()
        plugin_name = ep.dist.name
        if not isinstance(env, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in env.items()
        ):
            raise RegistryError(
                f"plugin '{plugin_name}' mitigation '{ep.name}' must resolve to "
                f"dict[str, str]; got {type(env).__name__}"
                + (f" with non-string entries {dict(env)!r}" if isinstance(env, dict) else "")
            )
        if ep.name in registry:
            existing = registry[ep.name].source_package
            raise RegistryCollisionError(
                f"mitigation '{ep.name}' registered by both '{existing}' "
                f"and '{plugin_name}' — rename one or remove the duplicate"
            )
        registry[ep.name] = Mitigation(
            name=ep.name, env=dict(env), source_package=plugin_name
        )

    return registry


def get_mitigation(name: str) -> dict[str, str]:
    """Return the env-var bundle for a mitigation name. Empty dict for 'none'.

    Returns a defensive copy — mutating the result does not affect the registry.
    """
    registry = load_mitigations()
    if name not in registry:
        raise UnknownMitigationError(
            f"unknown mitigation '{name}'; available: {sorted(registry)}; "
            f"if you expected a plugin-contributed entry, ensure the plugin is installed"
        )
    return dict(registry[name].env)
