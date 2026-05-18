"""``aorta env`` - thin CLI wrapper around :func:`collect_env`.

The library function in :mod:`aorta.instrumentation.environment` does all
the probing; this module only handles arg parsing, dispatch between
output modes (full snapshot, brief summary, or one-field lookup), and
writing the JSON snapshot to disk. Per #147 acceptance: this file does
no probing of its own.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click


@click.group()
def env() -> None:
    """Capture and compare GPU/library environment for trial reproducibility."""


@env.command()
@click.option(
    "--output",
    "-o",
    # NOTE: deliberately not setting ``writable=True`` -- Click validates
    # that during arg parsing, which fails for ``-o newdir/env.json``
    # *before* we get a chance to ``mkdir`` the parent. We create the
    # parent ourselves and let the actual write surface any I/O error.
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("env.json"),
    show_default=True,
    help="Path to write env.json (ignored with --summary / --field).",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="After the brief, also print the full snapshot JSON to stdout.",
)
@click.option(
    "--summary",
    is_flag=True,
    help=(
        "Print only the one-screen brief and exit (skip JSON write). "
        "Use for quick eyeballing without producing an artifact."
    ),
)
@click.option(
    "--field",
    "field_path",
    type=str,
    default=None,
    metavar="DOTTED.PATH",
    help=(
        "Print one snapshot field as JSON and exit (skip file write). "
        "Example: --field pytorch_build.ninja_hipcc.targets.ck_sdpa."
        "use_defines_present.USE_ROCM_CK_SDPA. For keys containing "
        "'.' (e.g. 'libaotriton_v2.so'), use jq on a full snapshot."
    ),
)
def probe(
    output: Path,
    verbose: bool,
    summary: bool,
    field_path: str | None,
) -> None:
    """Capture trial-environment state to env.json (issue #147)."""
    from aorta.instrumentation.environment import collect_env

    # --summary and --field both bypass the file write -- only one
    # output mode at a time makes sense.
    if summary and field_path is not None:
        raise click.ClickException(
            "--summary and --field are mutually exclusive"
        )

    snapshot = collect_env()
    snapshot_dict = snapshot.to_dict()

    # --field: print one value as JSON and exit. Skips the file write
    # entirely; pair with `jq` / `xargs` for scripting.
    if field_path is not None:
        value = _lookup_field(snapshot_dict, field_path)
        # Compact JSON so a scalar (bool / int / str) prints as one
        # line with the value's type preserved. `default=str` would
        # mask a non-serializable leak; we want loud failure instead.
        click.echo(json.dumps(value))
        return

    # --summary: print only the brief + partial reasons. Skips the
    # file write -- pair with `aorta env probe -o env.json` separately
    # when you need both.
    if summary:
        click.echo(snapshot.summary())
        if snapshot.partial:
            click.echo("\nPartial reasons:")
            for reason in snapshot.partial_reasons:
                click.echo(f"  - {reason}")
        return

    # Default mode: write the JSON artifact AND print the brief. The
    # full snapshot stays the single source of truth for downstream
    # diffing; the brief is courtesy stdout for the operator who just
    # ran the command.
    output = output.expanduser().resolve()
    # Wrap the two filesystem ops so common operator errors (unwritable
    # parent, read-only mount, full disk, etc.) surface as a clean
    # one-line Click error instead of a Python traceback. collect_env()
    # itself never raises, so it stays outside the try blocks.
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise click.ClickException(
            f"Failed to create parent directory for {output}: {exc}"
        ) from exc

    # NOTE: deliberately not passing ``default=str`` -- the snapshot is
    # supposed to be JSON-native (str/int/bool/None/list/dict). If a
    # non-serializable type sneaks in (e.g. a Path or datetime), we want
    # the failure to be loud rather than silently stringified.
    try:
        output.write_text(json.dumps(snapshot_dict, indent=2))
    except OSError as exc:
        raise click.ClickException(
            f"Failed to write env probe to {output}: {exc}"
        ) from exc

    partial = " [PARTIAL]" if snapshot.partial else ""
    click.echo(
        f"Wrote env probe to {output} "
        f"(schema_version={snapshot.schema_version}){partial}"
    )
    click.echo(snapshot.summary())

    # Always inline the partial_reasons -- they are action items and
    # forcing the operator to ``jq env.json`` to read them is friction.
    # Costs nothing (the list is already in memory).
    if snapshot.partial:
        click.echo("\nPartial reasons:")
        for reason in snapshot.partial_reasons:
            click.echo(f"  - {reason}")

    if verbose:
        click.echo("\n--- Full snapshot ---")
        click.echo(json.dumps(snapshot_dict, indent=2))

    # Closing marker -- repeats the [PARTIAL] state at end-of-output so
    # an operator scrolled to the bottom (especially after --verbose
    # dumps the full JSON) immediately sees probe status without
    # scrolling back up. Matches the marker shown next to the "Wrote
    # env probe..." line.
    if snapshot.partial:
        click.echo(
            f"\n[PARTIAL, {len(snapshot.partial_reasons)} reason(s)]"
        )
    else:
        click.echo("\n[OK]")


def _lookup_field(snapshot_dict: dict[str, Any], dotted_path: str) -> Any:
    """Resolve a dotted-path into a nested dict, ClickException on miss.

    Walks ``snapshot_dict[a][b][c]`` for path ``"a.b.c"``. Surfaces a
    helpful error when:

    * A segment is not present (lists the keys actually available at
      that level, capped at ~10 so the message fits one screen).
    * A non-dict is encountered mid-path (e.g. the user tried to
      descend into a list or a scalar leaf).

    Limitation: dotted-path notation cannot reference keys that
    themselves contain a ``.``. The only such key in the current
    schema is ``"libaotriton_v2.so"`` under
    ``pytorch_build.binary_introspection.torch_lib_bundled``; reach it
    via ``jq`` on a full snapshot.
    """
    if not dotted_path:
        raise click.ClickException("--field path must be non-empty")
    parts = dotted_path.split(".")
    cur: Any = snapshot_dict
    for i, part in enumerate(parts):
        prefix = ".".join(parts[:i]) or "<root>"
        if cur is None:
            raise click.ClickException(
                f"Cannot descend into '{part}' at '{prefix}': value is null"
            )
        if not isinstance(cur, dict):
            raise click.ClickException(
                f"Cannot descend into '{part}' at '{prefix}': "
                f"value is {type(cur).__name__}, not an object"
            )
        if part not in cur:
            available = sorted(cur.keys())
            shown = ", ".join(available[:10])
            more = f" (+ {len(available) - 10} more)" if len(available) > 10 else ""
            raise click.ClickException(
                f"Key '{part}' not found at '{prefix}'. "
                f"Available keys: {shown}{more}"
            )
        cur = cur[part]
    return cur
