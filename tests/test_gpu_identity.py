"""The card id (hardware.uuid_hash): plans, receipts and saved profiles record a keyed hash of the GPU's UUID, never the UUID,
and execute still recognizes the card a plan was made for. The key is per install, owner-only, and never written out.

Every UUID here is assembled at run time, so no UUID-shaped literal sits in the source."""
import errno
import json
import os
import re
import stat
import threading

import pytest

from loggetta import hardware
from loggetta.execution import PlanNotExecutable, check_here
from loggetta.hardware import GPU, Fact, HardwareProfile, Host, gpu_identity, install_key, uuid_hash
from loggetta.plan import Constraints, ExecutionPlan, Workload

GiB = 1 << 30


@pytest.fixture(autouse=True)
def _own_key_file(tmp_path, monkeypatch):
    """Every test here writes its key under its own tmp_path, whatever conftest does (test_key_redirect guards that)."""
    monkeypatch.setattr(hardware, "KEY_FILE", tmp_path / "state" / "gpu-id.key")
UUID_SHAPE = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")


def a_uuid(seed: int) -> str:
    h = f"{seed:032x}"
    return "GPU-" + "-".join((h[:8], h[8:12], h[12:16], h[16:20], h[20:]))


def a_gpu(uuid, name="NVIDIA RTX A4000", index=0, **kw):
    r = lambda v: Fact(v, "reported")  # noqa: E731
    return GPU(index=index, vendor="nvidia", name=name, uuid=uuid, compute_capability=r((8, 6)), memory_total=r(16 * GiB),
               memory_free=r(16 * GiB), driver=r("575.64.05"), pcie_gen_max=r(4), pcie_width_max=r(16),
               pcie_gen_current=r(4), pcie_width_current=r(16), **kw)


def a_profile(*gpus):
    r = lambda v: Fact(v, "reported")  # noqa: E731
    host = Host(cpu_model=r("test cpu"), cpus=r(8), memory_total=r(64 * GiB), memory_available=r(60 * GiB),
                memory_limit=r(None))
    return HardwareProfile(gpus=gpus, host=host, platform="Linux x86_64")


def a_plan(gpu: dict):
    return ExecutionPlan(status="refused", model={"model": "org/Tiny"}, hardware={"gpu": gpu}, workload=Workload(),
                         constraints=Constraints(), budget={}, selected=None, alternatives=(), reasons=())


# ----------------------------------------------------------------------------------------------- the key and the hash


def test_the_key_is_created_once_owner_only_and_reused():
    path = hardware.KEY_FILE
    assert not path.exists()
    k = install_key()
    assert len(k) == 32 and path.read_bytes() == k and install_key() == k
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600


def test_the_key_directory_is_owner_only(tmp_path, monkeypatch):
    monkeypatch.setattr(hardware, "KEY_FILE", tmp_path / "new" / "state" / "gpu-id.key")   # a directory not made yet
    install_key()
    assert stat.S_IMODE(os.stat(hardware.KEY_FILE.parent).st_mode) == 0o700


def test_an_existing_key_with_a_wider_mode_is_narrowed_and_kept():
    k = install_key()
    os.chmod(hardware.KEY_FILE, 0o644)
    assert install_key() == k and stat.S_IMODE(os.stat(hardware.KEY_FILE).st_mode) == 0o600


def test_a_key_file_of_the_wrong_length_is_refused(monkeypatch):
    monkeypatch.setattr(hardware, "KEY_READ_TRIES", 2)
    monkeypatch.setattr(hardware, "KEY_READ_WAIT_S", 0)
    hardware.KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
    hardware.KEY_FILE.write_bytes(b"short")
    with pytest.raises(RuntimeError, match="holds 5 bytes, not 32"):
        install_key()


def test_a_key_still_being_placed_is_waited_for_never_used_short(monkeypatch):
    full = b"\x03" * 32
    hardware.KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
    hardware.KEY_FILE.write_bytes(b"")
    monkeypatch.setattr(hardware.time, "sleep", lambda s: hardware.KEY_FILE.write_bytes(full))
    assert install_key() == full


def test_many_concurrent_first_uses_agree_on_one_whole_key(tmp_path, monkeypatch):
    # one read, no waiting: the key's name must never hold a partial key, so a reader never needs the retry
    monkeypatch.setattr(hardware, "KEY_READ_TRIES", 1)
    for trial in range(200):
        path = tmp_path / f"t{trial}" / "gpu-id.key"
        start, got, errors = threading.Barrier(8), [], []

        def first_use():
            start.wait()
            try:
                got.append(install_key(path))
            except Exception as e:  # noqa: BLE001 -- every caller must succeed; collected and asserted below
                errors.append(e)
        threads = [threading.Thread(target=first_use) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors and len(got) == 8 and len(set(got)) == 1 and len(got[0]) == 32, (trial, errors, {len(k) for k in got})
        assert sorted(p.name for p in path.parent.iterdir()) == ["gpu-id.key"], trial   # no temporary file left behind


def test_a_first_use_that_dies_mid_write_leaves_no_key_behind(monkeypatch):
    real_fdopen = os.fdopen

    def dies(fd, *a, **k):                                # the process is killed before the key's bytes are written
        os.close(fd)
        raise KeyboardInterrupt
    monkeypatch.setattr(hardware.os, "fdopen", dies)
    with pytest.raises(KeyboardInterrupt):
        install_key()
    monkeypatch.setattr(hardware.os, "fdopen", real_fdopen)
    assert not hardware.KEY_FILE.exists()                 # never an empty key under the name
    k = install_key()                                     # so the next use simply makes one
    assert len(k) == 32 and hardware.KEY_FILE.read_bytes() == k


def test_a_first_use_that_loses_the_link_reads_the_winners_key(monkeypatch):
    winner = b"\x05" * 32
    real_link = os.link

    def lose(src, dst):                                   # another process links its whole key in first
        other = os.path.join(os.path.dirname(dst), ".winner.tmp")
        with open(other, "wb") as f:
            f.write(winner)
        real_link(other, dst)
        os.unlink(other)
        raise FileExistsError(errno.EEXIST, "exists", dst)
    monkeypatch.setattr(hardware.os, "link", lose)
    assert install_key() == winner and hardware.KEY_FILE.read_bytes() == winner


def test_a_filesystem_without_hard_links_fails_closed_and_never_renames(monkeypatch):
    def no_link(src, dst):
        raise OSError(errno.EPERM, "Operation not permitted")

    def never(*a, **k):
        raise AssertionError("the key is never renamed into place")
    monkeypatch.setattr(hardware.os, "link", no_link)
    monkeypatch.setattr(hardware.os, "rename", never)
    monkeypatch.setattr(hardware.os, "replace", never)
    with pytest.raises(RuntimeError, match="refused a hard link"):
        install_key()
    assert not hardware.KEY_FILE.exists() and list(hardware.KEY_FILE.parent.iterdir()) == []


def test_the_key_comes_from_secrets(monkeypatch):
    monkeypatch.setattr(hardware.secrets, "token_bytes", lambda n: b"\x09" * n)
    assert install_key() == b"\x09" * 32


def test_the_hash_is_stable_per_install_and_differs_across_installs():
    u = a_uuid(1)
    a, b = uuid_hash(u, b"\x01" * 32), uuid_hash(u, b"\x02" * 32)
    assert a == uuid_hash(u, b"\x01" * 32) and a != b
    assert a.startswith(hardware.UUID_HASH_PREFIX) and len(a) == len(hardware.UUID_HASH_PREFIX) + 32
    assert u not in a and uuid_hash(None) is None and uuid_hash("") is None
    assert uuid_hash(a_uuid(2), b"\x01" * 32) != a


# ------------------------------------------------------------------------------------------- what gets written out


def test_a_saved_profile_carries_the_card_id_and_never_the_uuid():
    u = a_uuid(3)
    d = a_profile(a_gpu(u)).to_dict()
    text = json.dumps(d)
    assert "uuid" not in d["gpus"][0] and d["gpus"][0]["uuid_hash"] == uuid_hash(u)
    assert u not in text and not UUID_SHAPE.search(text)
    assert install_key().hex() not in text                          # the key is never written into a saved file
    back = HardwareProfile.from_dict(d)
    assert back.gpus[0].uuid is None and gpu_identity(back.gpus[0]) == uuid_hash(u)


def test_a_profile_saved_before_card_ids_still_loads():
    u = a_uuid(4)
    old = a_profile(a_gpu(u)).to_dict()
    old["gpus"][0].pop("uuid_hash")
    old["gpus"][0]["uuid"] = u
    back = HardwareProfile.from_dict(old)
    assert back.gpus[0].uuid == u and gpu_identity(back.gpus[0]) == uuid_hash(u)


def test_a_plans_hardware_carries_the_card_id_and_never_the_uuid():
    from loggetta.planner import _hw_dict

    u = a_uuid(5)
    g = a_gpu(u)
    d = _hw_dict(a_profile(g), g)
    text = json.dumps(d)
    assert "uuid" not in d["gpu"] and d["gpu"]["uuid_hash"] == uuid_hash(u)
    assert u not in text and not UUID_SHAPE.search(text) and install_key().hex() not in text


# ------------------------------------------------------------------------------------- execute's card check, kept


def test_a_new_plan_runs_on_its_own_card():
    u = a_uuid(6)
    check_here(a_plan({"name": "NVIDIA RTX A4000", "uuid_hash": uuid_hash(u)}), a_profile(a_gpu(u)))


def test_a_new_plan_refuses_another_card_of_the_same_model_without_printing_a_uuid():
    u, other = a_uuid(7), a_uuid(8)
    with pytest.raises(PlanNotExecutable, match="plan again on this machine") as e:
        check_here(a_plan({"name": "NVIDIA RTX A4000", "uuid_hash": uuid_hash(u)}), a_profile(a_gpu(other)))
    msg = str(e.value)
    assert "is not that card" in msg and "a plan carried from another machine or install reads as a different card" in msg
    assert not UUID_SHAPE.search(msg)


def test_a_new_plan_from_another_install_refuses_even_on_the_same_card():
    u = a_uuid(9)
    with pytest.raises(PlanNotExecutable, match="plan again on this machine"):
        check_here(a_plan({"name": "NVIDIA RTX A4000", "uuid_hash": uuid_hash(u, b"\x07" * 32)}), a_profile(a_gpu(u)))


def test_a_plan_saved_before_card_ids_still_checks_its_raw_uuid():
    u, other = a_uuid(10), a_uuid(11)
    check_here(a_plan({"name": "NVIDIA RTX A4000", "uuid": u}), a_profile(a_gpu(u)))
    with pytest.raises(PlanNotExecutable, match="plan again on this machine") as e:
        check_here(a_plan({"name": "NVIDIA RTX A4000", "uuid": u}), a_profile(a_gpu(other)))
    assert not UUID_SHAPE.search(str(e.value))


def test_without_any_card_id_the_name_decides_as_before():
    check_here(a_plan({"name": "NVIDIA RTX A4000"}), a_profile(a_gpu(None)))
    with pytest.raises(PlanNotExecutable, match="a different model of card"):
        check_here(a_plan({"name": "NVIDIA RTX A4000"}), a_profile(a_gpu(None, name="NVIDIA RTX A6000")))
