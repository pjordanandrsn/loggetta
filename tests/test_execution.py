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


def _no_run(plan, *, seed, log):
    raise AssertionError("the backend was called for a plan that must not run")


def test_execute_hands_the_selected_setup_to_its_backend_and_writes_the_receipt(monkeypatch, tmp_path):
    p = a_plan()
    seen = {}

    def run(pl, *, seed, log):                                   # the backend: here, a stand-in that loads nothing
        seen["setup"] = pl.selected.setup
        return {"status": "OK", "measured": {"device_peak_bytes": 3 * GiB + GiB // 8}, "correctness": {},
                "engaged": {}, "data": {}}

    monkeypatch.setattr(experts4bit, "executor", lambda kind: run if kind == "train" else None)
    r = execute(p, out_dir=str(tmp_path), hardware=hw(), prov={"sources": {}})
    assert seen["setup"] == SETUP
    assert r["schema"] == RECEIPT_SCHEMA and r["plan"] == p.to_dict() and r["setup"] == SETUP
    assert r["run_id"].startswith("Tiny-MoE-device-grouped_nf4-t512-")       # the backend names its setup
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
