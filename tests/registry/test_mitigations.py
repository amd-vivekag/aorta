"""Tests for the mitigations registry (iteration 1 — built-ins only)."""

import pytest

from aorta.registry.errors import UnknownMitigationError
from aorta.registry.mitigations import get_mitigation


def test_get_mitigation_returns_env_vars():
    assert get_mitigation("tf32_off") == {"DISABLE_TF32": "1"}


def test_get_mitigation_none_is_empty_dict():
    assert get_mitigation("none") == {}


def test_get_mitigation_unknown_raises_with_helpful_message():
    with pytest.raises(UnknownMitigationError) as exc:
        get_mitigation("not_a_real_mitigation")
    msg = str(exc.value)
    assert "available:" in msg
    assert "plugin" in msg
