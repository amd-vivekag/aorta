"""Shared fixtures for registry tests — fake entry-point discovery.

The `fake_eps` and `fake_env_eps` fixtures patch the `entry_points` lookup in
the mitigations and environments modules respectively, with a controlled list
of fake entry-points. Use them to simulate plugins being installed without
actually installing anything.
"""

from dataclasses import dataclass

import pytest


@dataclass
class _FakeDist:
    name: str


@dataclass
class _FakeEntryPoint:
    name: str
    payload: dict
    dist: _FakeDist

    def load(self):
        return self.payload


def _fake_eps_for(monkeypatch, module_path: str):
    """Build a factory that, when called, installs fake entry-points at module_path."""
    def _install(specs):
        eps = [
            _FakeEntryPoint(name=n, payload=p, dist=_FakeDist(name=d))
            for n, p, d in specs
        ]
        monkeypatch.setattr(module_path, lambda group: eps)

    return _install


@pytest.fixture
def fake_eps(monkeypatch):
    """Install fake mitigation entry-points: fake_eps([(name, payload, dist), ...])."""
    return _fake_eps_for(monkeypatch, "aorta.registry.mitigations.entry_points")


@pytest.fixture
def fake_env_eps(monkeypatch):
    """Install fake environment entry-points: fake_env_eps([(name, payload, dist), ...])."""
    return _fake_eps_for(monkeypatch, "aorta.registry.environments.entry_points")
