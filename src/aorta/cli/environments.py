"""`aorta environments` — inspect the merged environments registry."""

import click

from aorta.registry import load_environments


@click.group()
def environments() -> None:
    """Inspect the merged environments registry (built-ins + plugins)."""


@environments.command(name="list")
def list_() -> None:
    """List every registered environment, its source package, and its docker/venv."""
    registry = load_environments()
    name_w = max(len("NAME"), *(len(n) for n in registry))
    src_w = max(len("SOURCE"), *(len(e.source_package) for e in registry.values()))
    docker_w = max(len("DOCKER"), *(len(e.docker or "-") for e in registry.values()))

    click.echo(
        f"{'NAME'.ljust(name_w)}  {'SOURCE'.ljust(src_w)}  "
        f"{'DOCKER'.ljust(docker_w)}  VENV"
    )
    for name in sorted(registry):
        e = registry[name]
        click.echo(
            f"{name.ljust(name_w)}  {e.source_package.ljust(src_w)}  "
            f"{(e.docker or '-').ljust(docker_w)}  {e.venv or '-'}"
        )
