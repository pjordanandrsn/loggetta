"""Shared test setup: the per-install card-id key (``hardware.KEY_FILE``) lives in a temporary directory for every test,
so no test writes a key under the real home directory, and no test reads one a developer's own install made.

The redirect carries its own trip wire: the assert below, and ``test_gpu_identity``'s guard test, both FAIL (never skip)
when the key would sit under the real home directory, so removing the redirect cannot pass silently."""
from pathlib import Path

import pytest

from loggetta import hardware

REAL_STATE = Path.home() / ".local" / "state"


@pytest.fixture(autouse=True)
def _install_key_in_tmp(tmp_path_factory, monkeypatch):
    monkeypatch.setattr(hardware, "KEY_FILE", tmp_path_factory.mktemp("state") / "gpu-id.key")
    assert REAL_STATE not in Path(hardware.KEY_FILE).resolve().parents, "a test would use the real card-id key"
