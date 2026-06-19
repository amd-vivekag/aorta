#!/usr/bin/env python3
"""Bump the project version in ``pyproject.toml``.

Used by the release workflow (and runnable locally) to compute the next version
without hard-coding it anywhere. Supports semantic-version bumps
(``major``/``minor``/``patch``) or setting an explicit version, and rewrites
only the single ``version = "..."`` line inside the ``[project]`` table so the
rest of the file is left byte-for-byte untouched.

Examples:
    python scripts/bump_version.py patch                 # 0.2.0 -> 0.2.1
    python scripts/bump_version.py minor                 # 0.2.0 -> 0.3.0
    python scripts/bump_version.py --set 1.4.2           # set an explicit version
    python scripts/bump_version.py --suffix rc20260619   # 0.2.0 -> 0.2.0rc20260619

Prints the new version to stdout so callers (e.g. CI) can capture it.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_SEMVER_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")
_SEMVER_PREFIX_RE = re.compile(r"^(\d+\.\d+\.\d+)")
_VERSION_LINE_RE = re.compile(r'^(\s*version\s*=\s*")([^"]*)(".*)$')


def bump_version(current: str, level: str) -> str:
    """Return ``current`` bumped by ``level`` (``major``/``minor``/``patch``)."""
    match = _SEMVER_RE.match(current)
    if match is None:
        raise ValueError(
            f"cannot bump non-semver version {current!r}; expected MAJOR.MINOR.PATCH"
        )
    major, minor, patch = (int(part) for part in match.groups())
    if level == "major":
        return f"{major + 1}.0.0"
    if level == "minor":
        return f"{major}.{minor + 1}.0"
    if level == "patch":
        return f"{major}.{minor}.{patch + 1}"
    raise ValueError(f"unknown bump level {level!r}; expected major/minor/patch")


def apply_suffix(current: str, suffix: str) -> str:
    """Return the ``MAJOR.MINOR.PATCH`` base of ``current`` with ``suffix`` appended.

    Any existing pre-release/local part on ``current`` is dropped first, so
    re-stamping (e.g. ``0.2.0rc20260101`` -> ``0.2.0rc20260619``) is idempotent
    on the base version. Used to mint nightly release-candidate versions such as
    ``0.2.0rc20260619``.
    """
    match = _SEMVER_PREFIX_RE.match(current)
    if match is None:
        raise ValueError(
            f"cannot suffix non-semver version {current!r}; expected a MAJOR.MINOR.PATCH prefix"
        )
    return f"{match.group(1)}{suffix}"


def read_version(text: str) -> str:
    """Return the version declared in the ``[project]`` table of ``text``."""
    in_project = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_project = stripped == "[project]"
            continue
        if in_project:
            match = _VERSION_LINE_RE.match(line)
            if match is not None:
                return match.group(2)
    raise ValueError("no project.version found in pyproject text")


def set_version(text: str, new_version: str) -> str:
    """Return ``text`` with the ``[project]`` version line set to ``new_version``."""
    out: list[str] = []
    in_project = False
    replaced = False
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_project = stripped == "[project]"
            out.append(line)
            continue
        if in_project and not replaced:
            match = _VERSION_LINE_RE.match(line.rstrip("\n"))
            if match is not None:
                newline = "\n" if line.endswith("\n") else ""
                out.append(f"{match.group(1)}{new_version}{match.group(3)}{newline}")
                replaced = True
                continue
        out.append(line)
    if not replaced:
        raise ValueError("no project.version line to replace in pyproject text")
    return "".join(out)


def resolve_new_version(
    current: str,
    level: str | None,
    explicit: str | None,
    suffix: str | None = None,
) -> str:
    """Resolve the target version from an ``explicit`` value, a ``suffix``, or a bump ``level``.

    Precedence: ``explicit`` (``--set``) > ``suffix`` > ``level``.
    """
    if explicit is not None:
        if _SEMVER_RE.match(explicit) is None:
            raise ValueError(f"explicit version {explicit!r} is not MAJOR.MINOR.PATCH")
        return explicit
    if suffix is not None:
        return apply_suffix(current, suffix)
    if level is not None:
        return bump_version(current, level)
    raise ValueError(
        "one of a bump level (major/minor/patch), --set VERSION, or --suffix SUFFIX is required"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "level",
        nargs="?",
        choices=("major", "minor", "patch"),
        help="semantic-version component to bump",
    )
    parser.add_argument(
        "--set",
        dest="explicit",
        help="set an explicit MAJOR.MINOR.PATCH version (overrides the bump level)",
    )
    parser.add_argument(
        "--suffix",
        help="append SUFFIX to the base MAJOR.MINOR.PATCH version, e.g. 'rc20260619' "
        "(used for nightly release candidates; overrides the bump level)",
    )
    parser.add_argument(
        "--pyproject",
        type=Path,
        default=Path("pyproject.toml"),
        help="path to pyproject.toml (default: ./pyproject.toml)",
    )
    args = parser.parse_args(argv)

    text = args.pyproject.read_text()
    current = read_version(text)
    new_version = resolve_new_version(current, args.level, args.explicit, args.suffix)
    args.pyproject.write_text(set_version(text, new_version))
    print(new_version)
    return 0


if __name__ == "__main__":
    sys.exit(main())
