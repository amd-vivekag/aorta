"""Tests for the environments registry."""

import pytest

from aorta.registry.errors import (
    RegistryCollisionError,
    RegistryError,
    UnknownEnvironmentError,
)
from aorta.registry.environments import get_environment, load_environments
from aorta.registry.types import Environment


def test_load_environments_includes_builtins(fake_env_eps):
    fake_env_eps([])
    result = load_environments()
    assert "local" in result
    assert "default" in result
    assert result["local"].source_package == "aorta"


def test_get_environment_returns_dataclass(fake_env_eps):
    fake_env_eps([])
    env = get_environment("local")
    assert isinstance(env, Environment)
    assert env.docker is None
    assert env.venv is None
    assert env.source_package == "aorta"


def test_get_environment_unknown_raises(fake_env_eps):
    fake_env_eps([])
    with pytest.raises(UnknownEnvironmentError) as exc:
        get_environment("not_a_real_env")
    msg = str(exc.value)
    assert "available:" in msg
    assert "plugin" in msg


def test_load_environments_discovers_plugin(fake_env_eps):
    fake_env_eps([
        ("envs", {"nan-repro": {"docker": "rocm/private@sha256:abc"}}, "fake_internal"),
    ])
    result = load_environments()
    assert result["nan-repro"].docker == "rocm/private@sha256:abc"
    assert result["nan-repro"].source_package == "fake_internal"


def test_collision_between_plugins_raises(fake_env_eps):
    fake_env_eps([
        ("a", {"shared": {"docker": "img:1"}}, "plugin_a"),
        ("b", {"shared": {"docker": "img:2"}}, "plugin_b"),
    ])
    with pytest.raises(RegistryCollisionError, match="plugin_a.*plugin_b"):
        load_environments()


def test_invalid_key_raises(fake_env_eps):
    fake_env_eps([("p", {"bad": {"rocm": "6.0"}}, "plugin_x")])
    with pytest.raises(RegistryError, match="plugin_x.*rocm"):
        load_environments()
