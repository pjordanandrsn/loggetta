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
    ({"kind": "serve"}, hw(), "serve execution is a development executor"),   # no executor: the backend says why
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
    assert _e4b_env(E4B_K25_PLAN="x")["sha256"] != _e4b_env()["sha256"]      # a set name off the list still counts
    assert _e4b_env()["names"] == [] and _e4b_env()["values"] == {}


@pytest.mark.parametrize("name", ["E4B_SSH_HOSTNAME", "E4B_ENDPOINT", "E4B_SERVER_URL", "E4B_APIKEY", "E4B_PASSWD",
                                  "E4B_CREDENTIALS", "E4B_RECEIPTS_PATH", "E4B_MODEL_ID", "E4B_K25_PLAN"])
def test_e4b_env_takes_a_value_only_from_the_list(name):
    """An allowlist, not a denylist: a variable off the list moves neither the record nor the hash by its VALUE, whatever
    its name looks like, so no naming pattern can let a host, path, model id or secret through."""
    secret = "v-" + name.lower() + "-0123456789"
    rec, other = _e4b_env(**{name: secret}), _e4b_env(**{name: "something-else"})
    assert secret not in json.dumps(rec) and rec["names"] == [name] and rec["values"] == {}
    assert rec["sha256"] == other["sha256"]


def test_provenance_records_the_e4b_env(monkeypatch):
    from loggetta.measure import provenance

    monkeypatch.setenv("E4B_ABSMAX_DQ", "0")
    monkeypatch.setenv("E4B_RENT_SSH_HOST", "ssh7.rental-host.invalid")
    p = provenance(modules=())
    assert p["e4b_env"]["values"].get("E4B_ABSMAX_DQ") == "0" and "E4B_RENT_SSH_HOST" in p["e4b_env"]["names"]
    assert "ssh7.rental-host.invalid" not in json.dumps(p)



# ---- the serve executor (experts4bit_serve): its engine is a stand-in here, so no GPU and no backend package

SERVE_SETUP = {"placement": "all-vram", "graphs": False, "max_seqs": 1, "max_tokens_per_seq": 4096}


def a_serve_plan(allow=True, concurrency=1, model="org/Tiny-MoE"):
    from dataclasses import replace

    p = a_plan()
    sel = replace(p.selected, setup=SERVE_SETUP)
    return replace(p, selected=sel, model={"model": model},
                   workload=Workload(kind="serve", context_len=4096, concurrency=concurrency),
                   constraints=Constraints(allow_development_executor=allow))


def _serve_files(tmp_path):
    arena, calib = tmp_path / "olmoe.nf4", tmp_path / "calib.json"
    arena.write_bytes(b"\x01" * 4096)
    calib.write_text("{}")
    return str(arena), str(calib)


@pytest.fixture
def stub_env(monkeypatch):
    from loggetta.backends import experts4bit_serve

    monkeypatch.setattr(experts4bit_serve, "serve_env", lambda setup, model, arena, calib: {
        "E4B_PAGED_PLACEMENT": setup["placement"], "E4B_PAGED_MODEL": model, "E4B_PAGED_ARENA": arena,
        "E4B_PAGED_CALIB": calib})
    return experts4bit_serve


def test_a_serve_plan_runs_through_the_engine_and_its_receipt_carries_no_path(stub_env, monkeypatch, tmp_path):
    arena, calib = _serve_files(tmp_path)
    seen = {}

    def engine(env, *, concurrency, context_len, prompt_tokens, new_tokens, seed, trace_path, log):
        seen.update(env=env, concurrency=concurrency, prompt_tokens=prompt_tokens, trace_path=trace_path)
        return {"measured": {"device_peak_bytes": 3 * GiB + GiB // 8, "driver_process_peak_bytes": 4 * GiB,
                             "decode_step_ms_b1": 21.5, "tokens_per_s": 46.5},
                "info": {"moe_layers": 16}, "placement": None,
                "requests": {"count": concurrency, "prompt_tokens": prompt_tokens, "new_tokens": new_tokens}}

    real = stub_env.run
    monkeypatch.setattr(experts4bit, "executor", lambda kind: (lambda pl, **kw: real(pl, engine=engine, **kw)))
    r = execute(a_serve_plan(), out_dir=str(tmp_path / "receipts"), hardware=hw(), prov={"sources": {}},
                inputs={"arena": arena, "calib": calib})
    assert r["status"] == "OK" and r["setup"] == SERVE_SETUP
    assert "-serve-all-vram-c4096x1-" in r["run_id"]                       # a serve run id names its context x concurrency
    assert seen["env"]["E4B_PAGED_ARENA"] == arena and seen["concurrency"] == 1 and seen["prompt_tokens"] == 1024
    assert seen["trace_path"].endswith("steps.jsonl")                       # one sequence: the decode step is traced
    assert r["measured"]["decode_step_ms_b1"] == 21.5
    assert r["data"]["arena"] == "olmoe.nf4" and len(r["data"]["arena_head_sha256"]) == 64
    assert "E4B_PAGED_ARENA" not in r["data"]["server_env"] and r["data"]["server_env"]["E4B_PAGED_PLACEMENT"] == "all-vram"
    assert str(tmp_path) not in json.dumps(r)                               # no path reaches a shared receipt
    assert "serving" in r["comparison"]["device_allocator"]["what"]
    assert r["comparison"]["device_driver"]["residual"] == 4 * GiB - (3 * GiB + GiB // 2 + GiB // 4)


def test_a_serve_runs_data_carries_no_path_for_a_local_model_directory(stub_env, tmp_path):
    """The server environment names the model; a local model directory is a path, so the receipt's data block drops
    it (the receipt's own model field names the model). The arena and calibration file appear by name only."""
    arena, calib = _serve_files(tmp_path)
    local_model = str(tmp_path / "models" / "olmoe-local")
    out = stub_env.run(a_serve_plan(model=local_model), seed=0, log=print, adapter_dir=str(tmp_path / "r" / "adapter"),
                       run_id="x", arena=arena, calib=calib,
                       engine=lambda env, **kw: {"measured": {}, "requests": {"count": 1}})
    text = json.dumps(out["data"]) + json.dumps(out["engaged"])
    assert str(tmp_path) not in text and "olmoe-local" not in json.dumps(out["data"]["server_env"])


def test_a_batched_serve_run_is_not_traced(stub_env, tmp_path):
    arena, calib = _serve_files(tmp_path)
    got = {}
    stub_env.run(a_serve_plan(concurrency=4), seed=0, log=print, adapter_dir=str(tmp_path / "r" / "adapter"), run_id="x",
                 arena=arena, calib=calib, engine=lambda env, **kw: got.update(kw) or {"measured": {}, "requests": {}})
    assert got["trace_path"] is None and got["concurrency"] == 4


def _never(env, **kw):
    raise AssertionError("the engine was started for a serve run that must be refused first")


@pytest.mark.parametrize("allow, files, inputs, match", [
    (False, True, {}, "development executor pending its GPU proof"),          # not without the explicit permission
    (True, False, {"arena": None}, "needs the model's NF4 arena"),             # never baked implicitly
    (True, False, {"arena": "missing.nf4"}, "no NF4 arena at 'missing.nf4'"),
    (True, True, {"calib": None}, "calibration file"),
    (True, True, {"prompt_tokens": 4090, "new_tokens": 64}, "must fit inside the plan's context"),
])
def test_a_serve_run_is_refused_by_name_before_anything_loads(stub_env, tmp_path, allow, files, inputs, match):
    arena, calib = _serve_files(tmp_path) if files else (None, str(tmp_path / "calib.json"))
    kw = {"arena": arena, "calib": calib, **inputs}
    with pytest.raises(PlanNotExecutable, match=match):
        stub_env.run(a_serve_plan(allow=allow), seed=0, log=print, adapter_dir=str(tmp_path / "r" / "adapter"),
                     run_id="x", engine=_never, **kw)


def test_the_cli_hands_a_serve_plans_inputs_to_execute(monkeypatch, tmp_path):
    from loggetta import execution
    from loggetta.cli import main

    saved = tmp_path / "plan.json"
    saved.write_text(a_serve_plan(allow=False).to_json())
    got = {}

    def fake_execute(p, **kw):
        got.update(kw, allow=p.constraints.allow_development_executor)
        raise PlanNotExecutable("stop here")

    monkeypatch.setattr(execution, "execute", fake_execute)
    assert main(["execute", str(saved), "--arena", "a.nf4", "--calib", "c.json", "--new-tokens", "32",
                 "--allow-development-executor", "--out", str(tmp_path / "receipts")]) == 2
    assert got["inputs"] == {"arena": "a.nf4", "calib": "c.json", "new_tokens": 32} and got["allow"] is True
