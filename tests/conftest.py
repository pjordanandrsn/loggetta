"""Shared test setup: the per-install card-id key (``hardware.KEY_FILE``) lives in a temporary directory for every test,
so no test writes a key under the real home directory, and no test reads one a developer's own install made."""
import pytest

from loggetta import hardware


@pytest.fixture(autouse=True)
def _install_key_in_tmp(tmp_path_factory, monkeypatch):
    monkeypatch.setattr(hardware, "KEY_FILE", tmp_path_factory.mktemp("state") / "gpu-id.key")
