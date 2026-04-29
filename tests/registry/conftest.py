"""Shared fixtures for registry tests — fake entry-point discovery.

The `fake_eps` fixture patches `aorta.registry.mitigations.entry_points` with
a controlled list of fake entry-points. Use it to simulate plugins being
installed without actually installing anything.
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


@pytest.fixture
def fake_eps(monkeypatch):
    """Install a controlled list of fake mitigation entry-points.

    Usage:
        fake_eps([(ep_name, payload_dict, dist_name), ...])

    Each tuple becomes one fake EntryPoint exposing the payload as its `.load()`
    return value and the dist name as `.dist.name`.
    """
    def _install(specs):
        eps = [
            _FakeEntryPoint(name=n, payload=p, dist=_FakeDist(name=d))
            for n, p, d in specs
        ]
        monkeypatch.setattr(
            "aorta.registry.mitigations.entry_points",
            lambda group: eps,
        )

    return _install
