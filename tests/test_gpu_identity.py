"""The card id (hardware.uuid_hash): plans, receipts and saved profiles record a keyed hash of the GPU's UUID, never the UUID,
and execute still recognizes the card a plan was made for. The key is per install, owner-only, and never written out.

Every UUID here is assembled at run time, so no UUID-shaped literal sits in the source."""
import json
import os
import re
import stat

import pytest

from loggetta import hardware
from loggetta.execution import PlanNotExecutable, check_here
from loggetta.hardware import GPU, Fact, HardwareProfile, Host, gpu_identity, install_key, uuid_hash
from loggetta.plan import Constraints, ExecutionPlan, Workload

GiB = 1 << 30
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
    assert "a different card id" in str(e.value) and not UUID_SHAPE.search(str(e.value))


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
    with pytest.raises(PlanNotExecutable, match="a different card name"):
        check_here(a_plan({"name": "NVIDIA RTX A4000"}), a_profile(a_gpu(None, name="NVIDIA RTX A6000")))
