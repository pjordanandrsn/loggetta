"""Execution is orchestration: a feasible plan made for this device goes to the backend it selected, and what comes
back becomes a receipt. Nothing else runs.

The plans here are built by hand and the backend's executor is a stand-in, so these tests need no backend package
installed: they check the handoff, not the backend.
"""
import json

import pytest

from loggetta.backends import experts4bit
from loggetta.execution import RECEIPT_SCHEMA, PlanNotExecutable, execute, load_observations
from loggetta.hardware import GPU, Fact, HardwareProfile, Host
from loggetta.plan import Candidate, Constraints, ExecutionPlan, MemoryLine, Workload

GiB = 1 << 30
SETUP = {"expert_residency": "device", "expert_kernel": "grouped_nf4", "attn_4bit": False}


def hw(name="Test GPU", uuid=None, gpus=1):
    r = lambda v: Fact(v, "reported")  # noqa: E731
    g = GPU(index=0, vendor="nvidia", name=name, uuid=uuid, compute_capability=r((8, 6)), memory_total=r(12 * GiB),
            memory_free=r(12 * GiB), driver=r("575.64.05"), pcie_gen_max=r(4), pcie_width_max=r(16),
            pcie_gen_current=r(4), pcie_width_current=r(16))
    host = Host(cpu_model=r("test cpu"), cpus=r(8), memory_total=r(64 * GiB), memory_available=r(40 * GiB),
                memory_limit=r(64 * GiB))
    return HardwareProfile(gpus=(g,) * gpus, host=host, platform="Linux x86_64")


def a_plan(status="feasible", kind="train", gpu="Test GPU", uuid=None):
    lines = (MemoryLine("frozen expert stacks", "device", 3 * GiB, "derived"),
             MemoryLine("allocator reserve (cached, unallocated blocks)", "device", GiB // 2, "measured"),
             MemoryLine("CUDA context + library workspaces", "device", GiB // 4, "inferred"),
             MemoryLine("process baseline (torch, CUDA, libraries, model objects)", "host", GiB, "inferred"))
    sel = Candidate(backend="experts4bit", setup=SETUP, lines=lines, device_bytes=3 * GiB + GiB // 2 + GiB // 4,
                    host_bytes=GiB, feasible=True)
    return ExecutionPlan(
        status=status, model={"model": "org/Tiny-MoE"}, hardware={"gpu": {"name": gpu, "uuid": uuid}},
        workload=Workload(kind=kind, seq_len=512), constraints=Constraints(),
        budget={"device": 12 * GiB, "device_source": "user", "host": 40 * GiB, "host_source": "user",
                "headroom": GiB // 2},
        selected=sel if status == "feasible" else None, alternatives=(), reasons=(),
        refusal=None if status == "feasible" else {"reasons": ["device 13.00 + headroom 0.50 GiB > budget 12.00 GiB"]})


def _no_run(plan, *, seed, log, adapter_dir, run_id):
    raise AssertionError("the backend was called for a plan that must not run")


def test_execute_hands_the_selected_setup_to_its_backend_and_writes_the_receipt(monkeypatch, tmp_path):
    p = a_plan()
    seen = {}

    def run(pl, *, seed, log, adapter_dir, run_id):                           # the backend: here, a stand-in that loads nothing
        seen["setup"] = pl.selected.setup
        seen["adapter_dir"] = adapter_dir
        seen["run_id"] = run_id
        return {"status": "OK", "measured": {"device_peak_bytes": 3 * GiB + GiB // 8}, "correctness": {},
                "engaged": {}, "data": {}}

    monkeypatch.setattr(experts4bit, "executor", lambda kind: run if kind == "train" else None)
    r = execute(p, out_dir=str(tmp_path), hardware=hw(), prov={"sources": {}})
    assert seen["setup"] == SETUP
    assert seen["adapter_dir"].endswith("/adapter")
    assert r["schema"] == RECEIPT_SCHEMA and r["plan"] == p.to_dict() and r["setup"] == SETUP
    assert r["run_id"].startswith("Tiny-MoE-device-grouped_nf4-t512-")       # the backend names its setup
    assert seen["run_id"] == r["run_id"]                                      # the executor is told its run id
    alloc = r["comparison"]["device_allocator"]                               # context and reserve lines excluded
    assert alloc["estimated"] == 3 * GiB and alloc["residual"] == GiB // 8
    assert [o["run_id"] for o in load_observations(str(tmp_path))] == [r["run_id"]]   # the next plan's evidence


@pytest.mark.parametrize("plan_kw, here, match", [
    ({"status": "refused"}, hw(), "refused plan"),
    ({"kind": "serve"}, hw(), "planned only"),
    ({}, hw(name="Other GPU"), "plan again on this machine"),          # its receipt would name the wrong card
    ({"uuid": "GPU-aaa"}, hw(uuid="GPU-bbb"), "plan again on this machine"),  # same model of card, another card
    ({}, hw(gpus=0), "no GPU"),
])
def test_only_a_feasible_plan_for_this_device_reaches_the_backend(monkeypatch, plan_kw, here, match):
    monkeypatch.setattr(experts4bit, "executor", lambda kind: _no_run if kind == "train" else None)
    with pytest.raises(PlanNotExecutable, match=match):
        execute(a_plan(**plan_kw), hardware=here)


def test_the_cli_executes_a_saved_plan_only_where_it_was_made(tmp_path, capsys):
    from loggetta.cli import main

    saved = tmp_path / "plan.json"
    saved.write_text(a_plan().to_json())
    assert ExecutionPlan.from_dict(json.loads(saved.read_text())).to_json() == a_plan().to_json()
    assert main(["execute", str(saved), "--out", str(tmp_path / "receipts")]) == 2   # made for "Test GPU", not here
    assert "not executable" in capsys.readouterr().err
    assert not (tmp_path / "receipts").exists()


@pytest.mark.parametrize("value", ["expandable_segments:True", None])
def test_provenance_records_the_allocator_setting(monkeypatch, value):
    """The allocator setting a run starts under is recorded, and recorded as null when unset (not left out), so a
    reader can tell "unset" from "not recorded"."""
    from loggetta.measure import provenance

    if value is None:
        monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    else:
        monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", value)
    p = provenance(modules=())
    assert "pytorch_cuda_alloc_conf" in p and p["pytorch_cuda_alloc_conf"] == value


def _e4b_env(**set_):
    from loggetta.measure import e4b_env

    return e4b_env({"PATH": "/usr/bin", **set_})


def test_e4b_env_records_values_only_for_the_compute_switches():
    from loggetta.measure import E4B_VALUE_KNOBS

    rec = _e4b_env(E4B_ABSMAX_DQ="0", E4B_PAGED_GRAPHS="1", E4B_MODEL_ID="org/private-model", E4B_RECEIPTS_PATH="/srv/r")
    assert rec["names"] == ["E4B_ABSMAX_DQ", "E4B_MODEL_ID", "E4B_PAGED_GRAPHS", "E4B_RECEIPTS_PATH"]
    assert rec["values"] == {"E4B_ABSMAX_DQ": "0", "E4B_PAGED_GRAPHS": "1"}
    assert set(rec["values"]) <= set(E4B_VALUE_KNOBS) and "PATH" not in rec["names"]
    assert "org/private-model" not in json.dumps(rec) and "/srv/r" not in json.dumps(rec)


def test_e4b_env_never_carries_a_rental_host_or_a_secret():
    """A rental's ssh host and a token are set in the same environment as the switches. Neither value reaches the
    record, and neither moves the hash, so the hash cannot be used to guess a low-entropy host name."""
    host, token = "ssh7.rental-host.invalid", "tok-0123456789abcdef"
    rec = _e4b_env(E4B_RENT_SSH_HOST=host, E4B_RENT_SSH_PORT="40123", E4B_TOKEN=token, E4B_HOST="10.9.8.7",
                   E4B_ABSMAX_DQ="0")
    text = json.dumps(rec)
    assert host not in text and token not in text and "40123" not in text and "10.9.8.7" not in text
    assert {"E4B_RENT_SSH_HOST", "E4B_RENT_SSH_PORT", "E4B_TOKEN", "E4B_HOST"} <= set(rec["names"])
    other = _e4b_env(E4B_RENT_SSH_HOST="ssh2.other.invalid", E4B_RENT_SSH_PORT="1", E4B_TOKEN="x", E4B_HOST="h",
                     E4B_ABSMAX_DQ="0")
    assert other["sha256"] == rec["sha256"]                     # plumbing values do not move the hash


def test_e4b_env_hash_tells_two_compute_environments_apart():
    a, b = _e4b_env(E4B_ABSMAX_DQ="0"), _e4b_env(E4B_ABSMAX_DQ="1")
    assert a["sha256"] != b["sha256"]
    assert _e4b_env(E4B_K25_PLAN="x")["sha256"] != _e4b_env(E4B_K25_PLAN="y")["sha256"]   # a value not on the list
    assert _e4b_env()["names"] == [] and _e4b_env()["values"] == {}


def test_provenance_records_the_e4b_env(monkeypatch):
    from loggetta.measure import provenance

    monkeypatch.setenv("E4B_ABSMAX_DQ", "0")
    monkeypatch.setenv("E4B_RENT_SSH_HOST", "ssh7.rental-host.invalid")
    p = provenance(modules=())
    assert p["e4b_env"]["values"].get("E4B_ABSMAX_DQ") == "0" and "E4B_RENT_SSH_HOST" in p["e4b_env"]["names"]
    assert "ssh7.rental-host.invalid" not in json.dumps(p)
