"""Mitigations registry: built-ins + entry-point discovery + collision detection.

`load_mitigations()` returns the merged registry of built-ins and plugin
contributions, keyed by name. Each entry carries its `source_package` so
collision errors can name the conflicting parties.

Plugin authors register a function in their `pyproject.toml` under the
`aorta.mitigations` entry-point group. The function returns a dict of
`mitigation_name -> {env_var_name: value}`.
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
        RegistryError: a plugin's entry-point payload was not a dict.
    """
    registry: dict[str, Mitigation] = {
        name: Mitigation(name=name, env=dict(env), source_package="aorta")
        for name, env in BUILTIN_MITIGATIONS.items()
    }

    for ep in entry_points(group=_GROUP):
        payload = ep.load()
        if not isinstance(payload, dict):
            raise RegistryError(
                f"plugin '{ep.dist.name}' entry-point '{ep.name}' returned "
                f"{type(payload).__name__}, expected dict[str, dict[str, str]]"
            )
        plugin_name = ep.dist.name
        for mit_name, env in payload.items():
            if mit_name in registry:
                existing = registry[mit_name].source_package
                raise RegistryCollisionError(
                    f"mitigation '{mit_name}' registered by both '{existing}' "
                    f"and '{plugin_name}' — rename one or remove the duplicate"
                )
            registry[mit_name] = Mitigation(
                name=mit_name, env=dict(env), source_package=plugin_name
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
