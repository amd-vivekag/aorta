"""Detector-disable knob for ``aorta probe`` (issue #229).

Operators sometimes need to silence a detector that fires on benign,
workload-specific behaviour (e.g. ``tier2:hang`` on a repro that
legitimately idles, or ``tier3:vram_growth`` on an opaque docker
wrapper where allocation is normal). Rather than editing the
classifier, a recipe (or ``--disable-detector`` CLI flag) can name
detectors / whole tiers to skip.

A disabled detector is NOT evaluated and does NOT count toward the
verdict or the fired-detector lists. This module is the single source
of truth for which tokens are valid so the recipe-builder, the CLI,
and :func:`aorta.probe.classifier.classify_trial` agree.

Two token shapes:

* tier token -- one of :data:`KNOWN_TIERS` (``tier1`` .. ``tier5``).
  Disables the whole tier; ``tier5`` is the custom-pattern tier.
* detector-id token -- ``<prefix>:<id>`` where ``prefix`` is in
  :data:`KNOWN_DETECTOR_PREFIXES` (e.g. ``tier2:hang``,
  ``tier3:vram_growth``, ``custom:my_pattern``). The ``id`` half is
  free-form (custom patterns are user-named) so we validate the
  prefix and the presence of an id, not the full catalogue.
"""

from __future__ import annotations

KNOWN_TIERS: tuple[str, ...] = ("tier1", "tier2", "tier3", "tier4", "tier5")

# ``custom`` is the detector-id prefix the Tier-5 scanner stamps
# (``custom:<id>``); the tier token for that tier is ``tier5``.
KNOWN_DETECTOR_PREFIXES: tuple[str, ...] = (
    "tier1",
    "tier2",
    "tier3",
    "tier4",
    "custom",
)


class DetectorSpecError(ValueError):
    """A disable-spec token is malformed or names an unknown tier/prefix."""


def normalize_tier(token: str) -> str:
    """Validate + canonicalise a whole-tier disable token.

    Case-insensitive; surrounding whitespace stripped. Raises
    :class:`DetectorSpecError` for anything not in :data:`KNOWN_TIERS`.
    """
    cleaned = token.strip().lower()
    if cleaned not in KNOWN_TIERS:
        raise DetectorSpecError(
            f"unknown tier {token!r}; expected one of {', '.join(KNOWN_TIERS)}"
        )
    return cleaned


def normalize_detector_id(token: str) -> str:
    """Validate a ``<prefix>:<id>`` detector-id disable token.

    The prefix is lower-cased and checked against
    :data:`KNOWN_DETECTOR_PREFIXES`; the id half has surrounding
    whitespace stripped but its case preserved (custom-pattern ids are
    user-named and case-sensitive). Stripping the id matters: a
    copy/paste token like ``'tier2: hang'`` would otherwise normalise to
    ``'tier2: hang'`` and never match the fired id ``'tier2:hang'``,
    silently disabling nothing. Raises :class:`DetectorSpecError` when
    the colon / id / prefix is missing or unknown (an id that is empty
    after trimming counts as missing).
    """
    cleaned = token.strip()
    prefix, sep, rest = cleaned.partition(":")
    prefix = prefix.strip()
    rest = rest.strip()
    if not sep or not rest:
        raise DetectorSpecError(
            f"malformed detector id {token!r}; expected '<tier>:<id>' "
            f"(e.g. 'tier2:hang')"
        )
    if prefix.lower() not in KNOWN_DETECTOR_PREFIXES:
        raise DetectorSpecError(
            f"unknown detector prefix {prefix!r} in {token!r}; expected one "
            f"of {', '.join(KNOWN_DETECTOR_PREFIXES)}"
        )
    return f"{prefix.lower()}:{rest}"


def normalize_tiers(tokens: object) -> tuple[str, ...]:
    """Validate a list of tier tokens into a de-duplicated, ordered tuple."""
    return _normalize_list("disable_detector_tiers", tokens, normalize_tier)


def normalize_detector_ids(tokens: object) -> tuple[str, ...]:
    """Validate a list of detector-id tokens into an ordered tuple."""
    return _normalize_list("disable_detectors", tokens, normalize_detector_id)


def _normalize_list(field: str, tokens: object, fn) -> tuple[str, ...]:
    if tokens is None:
        return ()
    if isinstance(tokens, str) or not isinstance(tokens, (list, tuple)):
        raise DetectorSpecError(f"{field}: must be a list of strings, got {tokens!r}")
    out: list[str] = []
    for tok in tokens:
        if not isinstance(tok, str):
            raise DetectorSpecError(f"{field}: entries must be strings, got {tok!r}")
        norm = fn(tok)
        if norm not in out:
            out.append(norm)
    return tuple(out)


__all__ = [
    "KNOWN_TIERS",
    "KNOWN_DETECTOR_PREFIXES",
    "DetectorSpecError",
    "normalize_tier",
    "normalize_detector_id",
    "normalize_tiers",
    "normalize_detector_ids",
]
