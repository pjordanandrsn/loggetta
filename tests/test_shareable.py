"""A shared receipt carries no local path: not in its provenance (argv, source files), not in its model, not in its
artifacts or notes. Built from a real command line with absolute paths and a real provenance record."""
import json
import os
import sys

import pytest

from loggetta import shareable
from loggetta.backends import experts4bit
from loggetta.execution import execute
from loggetta.hardware import GPU, Fact, HardwareProfile, Host
from loggetta.measure import provenance
from loggetta.plan import Candidate, Constraints, ExecutionPlan, MemoryLine, Workload

GiB = 1 << 30


def hw():
    r = lambda v: Fact(v, "reported")  # noqa: E731
    g = GPU(index=0, vendor="nvidia", name="Test GPU", uuid=None, compute_capability=r((8, 6)), memory_total=r(12 * GiB),
            memory_free=r(12 * GiB), driver=r("575.64.05"), pcie_gen_max=r(4), pcie_width_max=r(16),
            pcie_gen_current=r(4), pcie_width_current=r(16))
    host = Host(cpu_model=r("test cpu"), cpus=r(8), memory_total=r(64 * GiB), memory_available=r(40 * GiB),
                memory_limit=r(64 * GiB))
    return HardwareProfile(gpus=(g,), host=host, platform="Linux x86_64")


def _plan(model, setup, workload, allow=False):
    lines = (MemoryLine("frozen expert stacks", "device", 3 * GiB, "derived"),
             MemoryLine("process baseline (torch, CUDA, libraries, model objects)", "host", GiB, "inferred"))
    sel = Candidate(backend="experts4bit", setup=setup, lines=lines, device_bytes=3 * GiB, host_bytes=GiB, feasible=True)
    return ExecutionPlan(status="feasible", model={"model": model}, hardware={"gpu": {"name": "Test GPU", "uuid": None}},
                         workload=workload, constraints=Constraints(allow_development_executor=allow),
                         budget={"device": 12 * GiB, "device_source": "user", "host": 40 * GiB, "host_source": "user",
                                 "headroom": GiB // 2}, selected=sel, alternatives=(), reasons=(), refusal=None)


def _serve_files(d):
    arena, calib = d / "olmoe.nf4", d / "calib.json"
    arena.write_bytes(b"\x01" * 4096)
    calib.write_text("{}")
    return str(arena), str(calib)


def _strings(o):
    if isinstance(o, dict):
        for v in o.values():
            yield from _strings(v)
    elif isinstance(o, (list, tuple)):
        for v in o:
            yield from _strings(v)
    elif isinstance(o, str):
        yield o


def _assert_no_path(receipt, *roots):
    """No root (the temp dir, the home directory) anywhere in the JSON, and no string that IS an absolute path."""
    text = json.dumps(receipt)
    for root in roots:
        assert str(root) not in text, root
    leaks = [s for s in _strings(receipt) if s.startswith(("/", "~/"))]
    assert not leaks, leaks


@pytest.fixture
def machine(tmp_path, monkeypatch):
    """A command line and home directory the way a real run has them: absolute paths everywhere."""
    home = tmp_path / "home" / "someone"
    (home / "models" / "olmoe-local").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    plan_file, script = home / "plan.json", home / "venv" / "bin" / "lg"
    script.parent.mkdir(parents=True)
    plan_file.write_text("{}")
    script.write_text("#!/bin/sh\n")
    arena, calib = _serve_files(home)
    argv = [str(script), "execute", str(plan_file), "--arena", arena, f"--calib={calib}", "--out", str(home / "r")]
    monkeypatch.setattr(sys, "argv", argv)
    return {"home": home, "tmp": tmp_path, "model": str(home / "models" / "olmoe-local"), "arena": arena,
            "calib": calib, "plan_file": plan_file}


def test_a_train_receipt_from_a_real_command_line_carries_no_path(machine, monkeypatch):
    p = _plan(machine["model"], {"expert_residency": "device", "expert_kernel": "grouped_nf4", "attn_4bit": False},
              Workload(seq_len=512))
    adapter = str(machine["home"] / "r" / "run" / "adapter")

    def run(pl, *, seed, log, adapter_dir, run_id):
        return {"status": "OK", "measured": {"device_peak_bytes": 3 * GiB}, "correctness": {}, "engaged": {},
                "data": {}, "artifacts": {"adapter": {"path": adapter, "tensor_count": 2}},
                "artifact_error": f"saved to {adapter}, then the manifest write failed"}

    monkeypatch.setattr(experts4bit, "executor", lambda kind: run)
    r = execute(p, out_dir=str(machine["home"] / "r"), hardware=hw(), prov=provenance())
    _assert_no_path(r, machine["tmp"], machine["home"])
    on_disk = json.loads(next((machine["home"] / "r").glob("*.json")).read_text())
    _assert_no_path(on_disk, machine["tmp"], machine["home"])
    assert r["model"]["model"] == "local: olmoe-local" and r["plan"]["model"]["model"] == "local: olmoe-local"
    assert r["artifacts"]["adapter"]["path"] == "local: adapter" and "local: adapter" in r["artifact_error"]
    prov = r["provenance"]
    assert prov["argv"][2] == "local: plan.json" and prov["argv"][5].startswith("--calib=local: ")
    assert len(prov["argv_files"]["plan.json"]) == 64                       # a file argument keeps its sha256
    for name, src in prov["sources"].items():                               # package-relative, versioned, hashed
        assert not os.path.isabs(src["file"]) and len(src["sha256"]) == 64, (name, src)


def test_a_serve_receipt_from_a_real_command_line_carries_no_path(machine, monkeypatch):
    from loggetta.backends import experts4bit_serve as sv

    monkeypatch.setattr(sv, "serve_env", lambda setup, model, arena, calib: {
        "E4B_PAGED_PLACEMENT": "all-vram", "E4B_PAGED_MODEL": model, "E4B_PAGED_ARENA": arena,
        "E4B_PAGED_CALIB": calib})
    real = sv.run
    info = {"adapter": {"path": str(machine["home"] / "ad")},
            "note": f"arena {machine['arena']} mapped, calibration from {machine['calib']}"}
    monkeypatch.setattr(experts4bit, "executor", lambda kind: (lambda pl, **kw: real(
        pl, engine=lambda env, **k: {"measured": {"decode_step_ms_b1": 4.35}, "requests": {"count": 1},
                                     "info": info}, **kw)))
    p = _plan(machine["model"], {"placement": "all-vram", "graphs": False},
              Workload(kind="serve", context_len=4096, concurrency=1), allow=True)
    r = execute(p, out_dir=str(machine["home"] / "r"), hardware=hw(), prov=provenance(),
                inputs={"arena": machine["arena"], "calib": machine["calib"]})
    _assert_no_path(r, machine["tmp"], machine["home"])
    assert r["data"]["server_env"]["E4B_PAGED_MODEL"] == "local: olmoe-local"
    assert "local: calib.json" in r["engaged"]["info"]["note"]


@pytest.mark.parametrize("value, want", [
    ("org/Model", "org/Model"),                                      # a Hub id: as given
    ("https://host/org/model.whl", "https://host/org/model.whl"),     # a URL: as given
    ("/srv/models/olmoe", "local: olmoe"),
    ("~/models/olmoe/", "local: olmoe"),
    ("./olmoe", "local: olmoe"),
    ("local: olmoe", "local: olmoe"),                                # labelling twice changes nothing
])
def test_label_names_a_local_path_by_its_name_only(value, want):
    assert shareable.label(value) == want and shareable.label(shareable.label(value)) == want


@pytest.mark.parametrize("text, want", [
    ("profile loaded from /srv/hw/a.json, not probed", "profile loaded from local: a.json, not probed"),
    ("51.2 GB/s at 1/2 load", "51.2 GB/s at 1/2 load"),               # units and fractions are not paths
    ("see https://example.com/a/b", "see https://example.com/a/b"),   # nor is a URL
    ("e4b/gnf4 pair", "e4b/gnf4 pair"),
])
def test_scrub_replaces_paths_inside_text_and_nothing_else(text, want):
    assert shareable.scrub({"note": text}) == {"note": want}


def test_a_local_model_receipt_still_matches_a_plan_of_that_model(tmp_path):
    """Receipts name a local model directory by name; the planner compares by the same label, so a run on a local model
    still informs the next plan of it (and a Hub id compares exactly as before)."""
    from loggetta.planner import _observed

    model_dir = tmp_path / "olmoe-local"
    model_dir.mkdir()
    gpu = hw().gpus[0]
    w = Workload(seq_len=512)
    setup = {"expert_residency": "device"}
    obs = {"model": {"model": "local: olmoe-local"}, "setup": setup, "status": "OK",
           "workload": {"tokens_per_microbatch": w.tokens_per_microbatch}, "hardware": {"gpu": {"name": gpu.name}}}
    assert _observed([obs], str(model_dir), setup, w, gpu) is obs
    hub = {**obs, "model": {"model": "org/Model"}}
    assert _observed([hub], "org/Model", setup, w, gpu) is hub and _observed([hub], "org/Other", setup, w, gpu) is None
