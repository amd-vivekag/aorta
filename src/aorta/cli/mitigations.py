"""`aorta mitigations` — inspect the merged mitigations registry."""

import click

from aorta.registry import load_mitigations


@click.group()
def mitigations() -> None:
    """Inspect the merged mitigations registry (built-ins + plugins)."""


@mitigations.command(name="list")
def list_() -> None:
    """List every registered mitigation, its source package, and its env vars."""
    registry = load_mitigations()
    name_w = max(len("NAME"), *(len(n) for n in registry))
    src_w = max(len("SOURCE"), *(len(m.source_package) for m in registry.values()))

    click.echo(f"{'NAME'.ljust(name_w)}  {'SOURCE'.ljust(src_w)}  ENV")
    for name in sorted(registry):
        m = registry[name]
        env_str = " ".join(f"{k}={v}" for k, v in sorted(m.env.items())) or "(none)"
        click.echo(f"{name.ljust(name_w)}  {m.source_package.ljust(src_w)}  {env_str}")
