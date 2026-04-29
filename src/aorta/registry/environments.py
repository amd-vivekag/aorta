"""Environments registry: built-ins + entry-point discovery + collision detection.

Mirrors the mitigations registry. Plugin payloads are validated against
`_VALID_ENV_KEYS` — only `docker` and `venv` are accepted. ROCm version is
intentionally not a valid key (see `Environment` docstring).

Plugin authors register one entry-point per environment in their `pyproject.toml`
under the `aorta.environments` group. The entry-point name IS the environment
name; the loaded object is the recipe (`dict[str, str | None]` with keys
`docker` and/or `venv`). Mirrors the `aorta.workloads` extension-point pattern.
"""

from importlib.metadata import entry_points

from aorta.registry.errors import (
    RegistryCollisionError,
    RegistryError,
    UnknownEnvironmentError,
)
from aorta.registry.types import Environment

_GROUP = "aorta.environments"
_VALID_ENV_KEYS = frozenset({"docker", "venv"})

# Built-in environments. `local` and `default` are both "current process, no
# overrides" — `default` is reserved as a site-configurable alias. Customer
# docker recipes (nan-repro, hipblaslt-develop) ship from aorta-internal via
# the `aorta.environments` entry-point group, NOT here.
BUILTIN_ENVIRONMENTS: dict[str, dict[str, str | None]] = {
    "local":   {},
    "default": {},
}


def load_environments() -> dict[str, Environment]:
    """Discover and merge all environments: built-ins first, then entry-point plugins.

    No caching — re-reads entry-points each call.

    Raises:
        RegistryCollisionError: two contributors registered the same environment name.
        RegistryError: a plugin's payload was not a dict, contained keys other
            than `docker` / `venv`, or had non-`str | None` values.
    """
    registry: dict[str, Environment] = {
        name: Environment(
            name=name,
            docker=spec.get("docker"),
            venv=spec.get("venv"),
            source_package="aorta",
        )
        for name, spec in BUILTIN_ENVIRONMENTS.items()
    }

    for ep in entry_points(group=_GROUP):
        spec = ep.load()
        plugin_name = ep.dist.name
        if not isinstance(spec, dict):
            raise RegistryError(
                f"plugin '{plugin_name}' environment '{ep.name}' must resolve to "
                f"dict[str, str | None]; got {type(spec).__name__}"
            )
        invalid = set(spec) - _VALID_ENV_KEYS
        if invalid:
            raise RegistryError(
                f"plugin '{plugin_name}' environment '{ep.name}' has invalid "
                f"keys {sorted(invalid)}; allowed keys: {sorted(_VALID_ENV_KEYS)}"
            )
        bad_values = {k: v for k, v in spec.items() if v is not None and not isinstance(v, str)}
        if bad_values:
            raise RegistryError(
                f"plugin '{plugin_name}' environment '{ep.name}' has non-string values "
                f"{ {k: type(v).__name__ for k, v in bad_values.items()} }; "
                f"each value must be `str | None`"
            )
        if ep.name in registry:
            existing = registry[ep.name].source_package
            raise RegistryCollisionError(
                f"environment '{ep.name}' registered by both '{existing}' "
                f"and '{plugin_name}' — rename one or remove the duplicate"
            )
        registry[ep.name] = Environment(
            name=ep.name,
            docker=spec.get("docker"),
            venv=spec.get("venv"),
            source_package=plugin_name,
        )

    return registry


def get_environment(name: str) -> Environment:
    """Return the Environment dataclass for a given name.

    Unlike `get_mitigation` (which returns a dict), environments are richer than
    a flat env-var bundle, so the dataclass IS the public surface.
    """
    registry = load_environments()
    if name not in registry:
        raise UnknownEnvironmentError(
            f"unknown environment '{name}'; available: {sorted(registry)}; "
            f"if you expected a plugin-contributed entry, ensure the plugin is installed"
        )
    return registry[name]
