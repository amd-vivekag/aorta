"""`aorta run` - universal workload runner."""

from pathlib import Path

import click


@click.command()
@click.option(
    "--workload",
    required=True,
    help="Workload name (from aorta.workloads entry-point group).",
)
@click.option(
    "--trials",
    type=int,
    default=1,
    show_default=True,
    help="Number of trials per (docker, mitigation) cell.",
)
@click.option(
    "--dockers",
    default="",
    help="Comma-separated docker image names. Empty = current environment.",
)
@click.option(
    "--mitigations",
    default="",
    help="Comma-separated mitigation names from aorta_internal.mitigations.",
)
@click.option(
    "--mitigations-file",
    "mitigation_files",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    multiple=True,
    help=(
        "JSON file with ad-hoc mitigations and/or environments (repeatable). "
        "Parsed and validated now; consumption by `aorta run` lands with task B1."
    ),
)
@click.option(
    "--steps",
    type=int,
    default=None,
    help="Steps per trial (workload-specific; passes through to workload config).",
)
@click.option(
    "--results-dir",
    type=click.Path(file_okay=False, writable=True),
    default="results",
    show_default=True,
    help="Directory to write per-trial JSON.",
)
def run(
    workload: str,
    trials: int,
    dockers: str,
    mitigations: str,
    mitigation_files: tuple[Path, ...],
    steps: int | None,
    results_dir: str,
) -> None:
    """Run a workload across trials x dockers x mitigations.

    Implemented by task B1 (cli/run.py + run/dispatcher.py).
    """
    raise click.ClickException("aorta run - not yet implemented (task B1)")
