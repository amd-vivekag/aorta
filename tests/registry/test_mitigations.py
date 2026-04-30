"""Tests for the mitigations registry."""

import pytest

from aorta.registry.errors import (
    RegistryCollisionError,
    RegistryError,
    UnknownMitigationError,
)
from aorta.registry.mitigations import get_mitigation, load_mitigations


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
    # str() must NOT wrap message in quotes (KeyError's default repr behavior)
    assert not msg.startswith("'") and not msg.endswith("'")


def test_load_mitigations_includes_builtins(fake_eps):
    fake_eps([])
    result = load_mitigations()
    assert "tf32_off" in result
    assert result["tf32_off"].source_package == "aorta"


def test_load_mitigations_discovers_plugin(fake_eps):
    fake_eps([("foo", {"FOO": "1"}, "fake_plugin")])
    result = load_mitigations()
    assert result["foo"].env == {"FOO": "1"}
    assert result["foo"].source_package == "fake_plugin"


def test_collision_between_plugins_raises(fake_eps):
    fake_eps([
        ("foo", {"X": "1"}, "plugin_a"),
        ("foo", {"X": "2"}, "plugin_b"),
    ])
    with pytest.raises(RegistryCollisionError, match="plugin_a.*plugin_b"):
        load_mitigations()


def test_collision_plugin_vs_builtin_raises(fake_eps):
    fake_eps([("tf32_off", {"X": "1"}, "plugin_x")])
    with pytest.raises(RegistryCollisionError, match="aorta.*plugin_x"):
        load_mitigations()


def test_non_string_env_value_raises(fake_eps):
    fake_eps([("bad", {"K": 123}, "plugin_x")])
    with pytest.raises(RegistryError, match="plugin_x.*bad.*dict\\[str, str\\]"):
        load_mitigations()


def test_non_dict_payload_raises(fake_eps):
    fake_eps([("bad", "not-a-dict", "plugin_x")])
    with pytest.raises(RegistryError, match="plugin_x.*bad.*str"):
        load_mitigations()
