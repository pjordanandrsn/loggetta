"""The trip wire for conftest's card-id key redirect: no test may read or write the real per-install key. This file has
no fixture of its own, so it sees exactly what conftest leaves; if the redirect is removed, this FAILS (never skips)."""
from pathlib import Path

from loggetta import hardware


def test_the_card_id_key_is_never_the_real_one_in_tests():
    real = Path.home().resolve() / ".local" / "state"       # resolved like KEY_FILE, so a symlinked home cannot disarm it
    assert real not in Path(hardware.KEY_FILE).resolve().parents, hardware.KEY_FILE
