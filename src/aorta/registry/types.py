"""Data types for the mitigations + environments registry."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Mitigation:
    """A named bundle of environment variables that modifies workload behavior.

    `frozen=True` prevents reassigning attributes (e.g. `m.name = "x"` raises),
    but the `env` dict itself is still mutable in place. Callers should treat
    `env` as read-only; `get_mitigation()` returns a defensive copy.
    """

    name: str
    env: dict[str, str]
    source_package: str  # "aorta" for built-ins, dist name for entry-point contributors


@dataclass(frozen=True)
class Environment:
    """A baseline process / container recipe for a workload run.

    `docker` and `venv` are independent ways of describing the baseline; either,
    both, or neither may be set (built-in `local` has neither — current process).
    No `rocm` field: ROCm version is implicit in the docker image digest or in
    the host the venv runs on; capture it from `aorta env probe` at runtime.
    """

    name: str
    docker: str | None = None
    venv: str | None = None
    source_package: str = "aorta"
