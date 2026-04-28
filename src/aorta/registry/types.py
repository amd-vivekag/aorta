"""Data types for the mitigations + environments registry.

Iteration 1 only defines `Mitigation` — `Environment` arrives in iteration 3.
"""

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
