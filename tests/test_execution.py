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
    ({"kind": "serve"}, hw(), "does not execute 'serve' workloads"),   # no executor: the backend says why
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


@pytest.mark.parametrize("installed", [True, False])
def test_a_receipt_names_the_loggetta_that_wrote_it(monkeypatch, installed):
    """Both with an installed distribution and from a source tree without one (the dist lookup then fails), the
    shareable provenance names loggetta by version, not only by one file's hash."""
    from importlib import metadata

    import loggetta
    from loggetta import shareable
    from loggetta.measure import provenance

    real = metadata.version

    def version(dist):
        if dist == "loggetta":
            if not installed:
                raise metadata.PackageNotFoundError(dist)
            return "9.9.9"
        return real(dist)

    monkeypatch.setattr(metadata, "version", version)
    src = shareable.provenance(provenance(modules=("loggetta",)))["sources"]["loggetta"]
    assert src["module_version"] == loggetta.__version__
    assert src["version"] == ("9.9.9" if installed else None)       # the installed distribution's, when there is one
    assert len(src["sha256"]) == 64 and src["file"] == "loggetta/__init__.py"


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
                                  "E4B_CREDENTIALS", "E4B_RECEIPTS_PATH", "E4B_MODEL_ID", "E4B_EXPERT_PROFILE"])
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


def a_serve_plan(allow=False, concurrency=1, model="org/Tiny-MoE"):
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


@pytest.mark.parametrize("local, recorded", [
    (False, "org/Tiny-MoE"),                               # a Hub id: recorded as given
    (True, "local: olmoe-local"),                          # a local model directory: its name only, never its path
])
def test_a_serve_runs_data_names_the_model_without_a_path(stub_env, tmp_path, local, recorded):
    """The recorded server environment names the model by the rule first_run.sh applies (#105): a local model
    directory by its name only. The arena and calibration file appear by name only."""
    arena, calib = _serve_files(tmp_path)
    model = str(tmp_path / "models" / "olmoe-local") if local else "org/Tiny-MoE"
    out = stub_env.run(a_serve_plan(model=model), seed=0, log=print, adapter_dir=str(tmp_path / "r" / "adapter"),
                       run_id="x", arena=arena, calib=calib,
                       engine=lambda env, **kw: {"measured": {}, "requests": {"count": 1}})
    assert out["data"]["server_env"]["E4B_PAGED_MODEL"] == recorded
    assert str(tmp_path) not in json.dumps(out["data"]) + json.dumps(out["engaged"])


def test_a_batched_serve_run_is_not_traced(stub_env, tmp_path):
    arena, calib = _serve_files(tmp_path)
    got = {}
    stub_env.run(a_serve_plan(concurrency=4), seed=0, log=print, adapter_dir=str(tmp_path / "r" / "adapter"), run_id="x",
                 arena=arena, calib=calib, engine=lambda env, **kw: got.update(kw) or {"measured": {}, "requests": {}})
    assert got["trace_path"] is None and got["concurrency"] == 4


def _never(env, **kw):
    raise AssertionError("the engine was started for a serve run that must be refused first")


@pytest.mark.parametrize("files, inputs, match", [
    (False, {"arena": None}, "needs the model's NF4 arena"),             # never baked implicitly
    (False, {"arena": "missing.nf4"}, "no NF4 arena at 'missing.nf4'"),
    (True, {"calib": None}, "calibration file"),
    (True, {"prompt_tokens": 4090, "new_tokens": 64}, "must fit inside the plan's context"),
])
def test_a_serve_run_is_refused_by_name_before_anything_loads(stub_env, tmp_path, files, inputs, match):
    arena, calib = _serve_files(tmp_path) if files else (None, str(tmp_path / "calib.json"))
    kw = {"arena": arena, "calib": calib, **inputs}
    with pytest.raises(PlanNotExecutable, match=match):
        stub_env.run(a_serve_plan(), seed=0, log=print, adapter_dir=str(tmp_path / "r" / "adapter"),
                     run_id="x", engine=_never, **kw)


@pytest.mark.parametrize("allow", [False, True])
def test_a_serve_run_needs_no_development_opt_in(stub_env, tmp_path, allow):
    """The serve executor passed its GPU proof (evidence/2026-10-23-serve-executor-rtx4090), so a plan runs with or
    without the development opt-in, which still gates the dense executor."""
    arena, calib = _serve_files(tmp_path)
    ran = []
    stub_env.run(a_serve_plan(allow=allow), seed=0, log=print, adapter_dir=str(tmp_path / "r" / "adapter"),
                 run_id="x", arena=arena, calib=calib,
                 engine=lambda env, **kw: ran.append(1) or {"measured": {}, "requests": {"count": 1}})
    assert ran == [1]


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
                 "--out", str(tmp_path / "receipts")]) == 2
    assert got["inputs"] == {"arena": "a.nf4", "calib": "c.json", "new_tokens": 32} and got["allow"] is False



@pytest.mark.parametrize("name", ["E4B_PAGED_ADAPTER", "E4B_PAGED_GRAPHS", "E4B_SERVE_ATTN_INT4",
                                  "E4B_NF4_T1_DEVICE_GROUPING", "E4B_K25_PLAN", "E4B_ABSMAX_DQ", "GNF4_ANY_SWITCH"])
def test_an_inherited_backend_switch_refuses_the_run_by_name(stub_env, monkeypatch, tmp_path, name):
    """The plan's environment is MERGED into the process's, so a switch the shell set and the plan did not would be
    live for the run while the receipt describes the plan. Denied by default (route switches outside any prefix, and
    every GNF4_*, included), refused by name, its value never printed."""
    arena, calib = _serve_files(tmp_path)
    monkeypatch.setenv(name, str(tmp_path / "secret-value"))
    with pytest.raises(PlanNotExecutable, match=name) as e:
        stub_env.run(a_serve_plan(), seed=0, log=print, adapter_dir=str(tmp_path / "r" / "adapter"), run_id="x",
                     arena=arena, calib=calib, engine=_never)
    assert "secret-value" not in str(e.value)


def test_plumbing_variables_do_not_block_a_serve_run(stub_env, monkeypatch, tmp_path):
    arena, calib = _serve_files(tmp_path)
    for k, v in {"E4B_RENT_SSH_HOST": "h", "E4B_RENT_SSH_PORT": "1", "E4B_TOKEN": "t", "E4B_HOST": "x",
                 "E4B_PORT": "8778", "UNRELATED_VAR": "y"}.items():
        monkeypatch.setenv(k, v)
    out = stub_env.run(a_serve_plan(), seed=0, log=print, adapter_dir=str(tmp_path / "r" / "adapter"), run_id="x",
                       arena=arena, calib=calib, engine=lambda env, **kw: {"measured": {}, "requests": {}})
    assert out["status"] == "OK"


def test_a_switch_the_plan_sets_is_not_inherited(monkeypatch):
    from loggetta.backends.experts4bit_serve import inherited_switches

    monkeypatch.setenv("E4B_INT4_ARTIFACT_DIR", "/srv/artifacts")
    monkeypatch.setenv("GNF4_X", "1")
    assert inherited_switches({}) == ["E4B_INT4_ARTIFACT_DIR", "GNF4_X"]
    assert inherited_switches({"E4B_INT4_ARTIFACT_DIR": "x", "GNF4_X": "1"}) == []      # the plan sets them


def test_the_engines_path_valued_report_is_named_without_its_path(stub_env, tmp_path):
    arena, calib = _serve_files(tmp_path)
    info = {"adapter": {"path": str(tmp_path / "adapters" / "my-adapter"), "adapter_id": "a1"},
            "levers_env": {"E4B_INT4_ARTIFACT_DIR": str(tmp_path / "art"), "E4B_SERVE_ATTN_INT4": "1"}}
    out = stub_env.run(a_serve_plan(), seed=0, log=print, adapter_dir=str(tmp_path / "r" / "adapter"), run_id="x",
                       arena=arena, calib=calib, engine=lambda env, **kw: {"measured": {}, "requests": {}, "info": info})
    got = out["engaged"]["info"]
    assert got["adapter"] == {"path": "local: my-adapter", "adapter_id": "a1"}
    assert got["levers_env"] == {"E4B_INT4_ARTIFACT_DIR": "local: art", "E4B_SERVE_ATTN_INT4": "1"}
    assert str(tmp_path) not in json.dumps(out["engaged"])



class _Plan:
    def __init__(self, empty):
        self.is_empty = empty


class _Sched:
    """Two requests of three decode steps each, run one step at a time (the shape of experts4bit-qlora's scheduler)."""

    def __init__(self, steps=3, stall=False):
        self.queue, self.active, self.left, self.stall = [1], [], steps, stall

    def step(self):
        if self.stall:
            return _Plan(True)
        if self.queue:
            self.active, self.queue = self.queue, []
        self.left -= 1
        if self.left == 0:
            self.active = []
        return _Plan(False)


class _Tracer:
    def __init__(self):
        self.calls, self.open = [], False

    def begin(self, **kw):
        assert not self.open, "a row was opened twice"
        self.open = True
        self.calls.append("begin")

    def end(self):
        assert self.open
        self.open = False
        self.calls.append("end")

    def discard(self):
        self.open = False
        self.calls.append("discard")


def test_drive_opens_and_closes_a_trace_row_around_every_step():
    """The server loop's shape: begin, step, end. scheduler.run_until_idle() never opens a row, so a trace attached
    to it stays empty, and the GPU smoke found no decode step."""
    from loggetta.backends.experts4bit_serve import drive

    tr = _Tracer()
    assert drive(_Sched(steps=3), tr) == 3
    assert tr.calls == ["begin", "end"] * 3 and not tr.open
    tr = _Tracer()
    assert drive(_Sched(stall=True), tr) == 0 and tr.calls == ["begin", "discard"]   # an empty step is discarded
    assert drive(_Sched(steps=2)) == 2                                                  # untraced: the same steps


def test_a_missing_or_empty_trace_is_a_note_not_a_crash(tmp_path):
    from loggetta.backends.experts4bit_serve import decode_step_from_trace

    for got, why in ((decode_step_from_trace(str(tmp_path / "steps.jsonl")), "wrote no file"),
                     ((tmp_path / "empty.jsonl").write_text("") or decode_step_from_trace(str(tmp_path / "empty.jsonl")),
                      "0 row(s)")):
        assert "decode_step_ms_b1" in got and got["decode_step_ms_b1"] is None   # ABSENT: null, never 0 or a guess
        assert why in got["decode_step_note"]
    rows = [{"decode_rows": 1, "prefill_tokens": 0, "step_ms": 20.0 + i} for i in range(9)]
    (tmp_path / "b1.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    got = decode_step_from_trace(str(tmp_path / "b1.jsonl"))
    assert "decode_step_note" not in got and got["step_trace"] == "b1.jsonl" and got



def test_driving_the_steps_gives_the_real_trace_one_row_per_step(tmp_path):
    """experts4bit-qlora's own StepTrace (on the CPU): the fix must PRODUCE rows, one per non-empty step, not only stop
    the crash. Through scheduler.run_until_idle() the same trace stays empty."""
    st = pytest.importorskip("experts4bit_qlora.engines.step_trace")
    from loggetta.backends.experts4bit_serve import drive

    path = tmp_path / "steps.jsonl"
    tr = st.StepTrace(str(path), cuda=False)
    assert drive(_Sched(steps=5), tr) == 5
    tr.close()
    rows = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
    assert len(rows) == 5 and [r["step"] for r in rows] == list(range(5)) and all("step_ms" in r for r in rows)

    other = tmp_path / "untraced.jsonl"
    tr = st.StepTrace(str(other), cuda=False)
    sched = _Sched(steps=5)
    sched.tracer = tr
    while sched.queue or sched.active:              # what run_until_idle does: step() with no row opened
        sched.step()
    tr.close()
    assert not other.exists()



@pytest.mark.parametrize("measured, want", [
    ({"decode_step_ms_b1": 4.3561, "decode_steps_b1": 63, "tokens_per_s": 151.8, "load_seconds": 4.9,
      "generate_seconds": 0.42}, "decode step median 4.36 ms over 63 steps"),
    ({"decode_step_ms_b1": None, "decode_step_note": "the step trace wrote no file (no traced step)",
      "tokens_per_s": None}, "decode step ABSENT: the step trace wrote no file"),
])
def test_summarize_reads_a_serve_receipt(stub_env, monkeypatch, tmp_path, measured, want):
    """The CLI summary was train-only (a serve receipt raised KeyError after it was written: found by the GPU smoke)."""
    from loggetta.execution import summarize

    arena, calib = _serve_files(tmp_path)
    real = stub_env.run
    monkeypatch.setattr(experts4bit, "executor", lambda kind: (lambda pl, **kw: real(
        pl, engine=lambda env, **k: {"measured": {"device_peak_bytes": 5 * GiB, **measured},
                                     "requests": {"count": 1, "prompt_tokens": 1024, "new_tokens": 64}}, **kw)))
    r = execute(a_serve_plan(), out_dir=str(tmp_path / "receipts"), hardware=hw(), prov={"sources": {}},
                inputs={"arena": arena, "calib": calib})
    text = summarize(r)
    assert want in text and "1 request(s) of 1024 prompt + 64 new tokens" in text and "device_allocator" in text



def test_the_cli_executes_a_serve_plan_end_to_end_through_its_summary(stub_env, monkeypatch, tmp_path, capsys):
    """`loggetta execute` on a saved serve plan, a stand-in engine underneath, through the printed summary: the hop the
    GPU smoke found raising after the receipt was written. Exit 0, a receipt on disk, the decode step in the summary."""
    from loggetta import execution
    from loggetta.cli import main

    arena, calib = _serve_files(tmp_path)
    saved = tmp_path / "plan.json"
    saved.write_text(a_serve_plan(allow=False).to_json())
    real = stub_env.run
    monkeypatch.setattr(execution, "check_here", lambda plan, hardware=None: None)   # the plan names a test card
    monkeypatch.setattr(experts4bit, "executor", lambda kind: (lambda pl, **kw: real(
        pl, engine=lambda env, **k: {"measured": {"device_peak_bytes": 5 * GiB, "decode_step_ms_b1": 4.3561,
                                                  "decode_steps_b1": 63, "tokens_per_s": 151.8},
                                     "requests": {"count": 1, "prompt_tokens": 1024, "new_tokens": 64}}, **kw)))
    out = tmp_path / "receipts"
    rc = main(["execute", str(saved), "--arena", arena, "--calib", calib, "--out", str(out)])
    text = capsys.readouterr().out
    assert rc == 0 and len(list(out.glob("*.json"))) == 1
    assert "decode step median 4.36 ms over 63 steps" in text and "receipt:" in text



@pytest.mark.parametrize("name, a, b", [
    ("E4B_K25_PLAN", "c1", "today"),                       # a route switch: two plans of the same kernel
    ("E4B_NF4_T1_DEVICE_GROUPING", "0", "1"),
    ("GNF4_GEMV_BW", "0", "1"),                            # grouped-nf4-gemm's own switches too
    ("GNF4_PDL", "0", "1"),
])
def test_a_route_switch_is_recorded_by_value_and_separates_the_hash(name, a, b):
    """Receipts could not tell E4B_K25_PLAN=c1 from =today, or a GNF4_* switch on from off: their VALUES are recorded
    now, and two runs differing only there hash differently."""
    ra, rb = _e4b_env(**{name: a}), _e4b_env(**{name: b})
    assert ra["values"] == {name: a} and rb["values"] == {name: b} and ra["sha256"] != rb["sha256"]


def test_a_gnf4_path_switch_and_a_path_valued_knob_stay_names_only():
    rec = _e4b_env(GNF4_NATIVE_CACHE="/srv/cache/gnf4", E4B_K25_PLAN="/srv/plans/x.json")
    assert rec["names"] == ["E4B_K25_PLAN", "GNF4_NATIVE_CACHE"]
    assert "GNF4_NATIVE_CACHE" not in rec["values"] and rec["values"]["E4B_K25_PLAN"] == "local: x.json"
    assert "/srv" not in json.dumps(rec)
    other = _e4b_env(GNF4_NATIVE_CACHE="/elsewhere/c", E4B_K25_PLAN="/other/dir/x.json")
    assert other["sha256"] == rec["sha256"]               # neither path moves the hash


def test_provenance_records_a_gnf4_switch_from_the_process_environment(monkeypatch):
    from loggetta.measure import provenance

    monkeypatch.setenv("GNF4_PDL", "1")
    monkeypatch.setenv("GNF4_NATIVE_CACHE", "/srv/cache/gnf4")
    rec = provenance(modules=())["e4b_env"]
    assert rec["values"].get("GNF4_PDL") == "1" and "GNF4_NATIVE_CACHE" in rec["names"]
    assert "GNF4_NATIVE_CACHE" not in rec["values"]



def test_every_backend_switch_is_classified_value_or_name_only():
    """Every E4B_/GNF4_ name the INSTALLED backends read (their package sources, tests and conftest aside) is either
    recorded by value (E4B_VALUE_KNOBS) or by name with a reason (NAME_ONLY), never both. A backend release that adds a
    knob fails this until someone decides whether its value is recorded."""
    import importlib.metadata as md
    import os
    import re

    from loggetta.measure import E4B_VALUE_KNOBS, NAME_ONLY

    pat = re.compile(r"""(["'])((?:E4B|GNF4)_[A-Z0-9_]*[A-Z0-9])\1""")          # either quote: e4b uses both
    assert [m[1] for m in pat.findall("""env("E4B_A1") or os.environ.get('GNF4_B2') or "E4B_SERVE_" """)] == [
        "E4B_A1", "GNF4_B2"]                                    # a prefix literal ("E4B_SERVE_") is not a name
    found = {}
    for dist in ("experts4bit-qlora", "grouped-nf4-gemm"):
        try:
            files = md.files(dist) or ()
        except md.PackageNotFoundError:
            pytest.skip(f"{dist} is not installed")
        for f in files:
            p = str(f)
            if p.endswith(".py") and "/tests/" not in "/" + p and not os.path.basename(p).startswith(("test_", "conftest")):
                with open(f.locate(), encoding="utf-8", errors="replace") as fh:
                    text = fh.read()
                for _, name in pat.findall(text):
                    found.setdefault(name, p)
    assert found, "no backend switch found: the scan itself is broken"
    assert not set(E4B_VALUE_KNOBS) & set(NAME_ONLY)
    from loggetta.measure import is_secret_name

    assert not [n for n in E4B_VALUE_KNOBS if is_secret_name(n)]     # a credential's value is never recorded
    assert all(reason.strip() for reason in NAME_ONLY.values())
    unclassified = {n: where for n, where in found.items() if n not in E4B_VALUE_KNOBS and n not in NAME_ONLY}
    assert not unclassified, ("classify each new backend switch in loggetta.measure: E4B_VALUE_KNOBS if it can change a "
                              "route, a kernel or a number, else NAME_ONLY with a reason", unclassified)


def test_a_newly_valued_switch_is_recorded_and_a_secret_never_is():
    rec = _e4b_env(E4B_FUSE_SWIGLU="0", E4B_TOKEN="tok-0123456789", E4B_PAGED_SPEC_HEAD="/srv/heads/eagle3")
    assert rec["values"]["E4B_FUSE_SWIGLU"] == "0" and rec["values"]["E4B_PAGED_SPEC_HEAD"] == "local: eagle3"
    assert "tok-0123456789" not in json.dumps(rec)
    assert _e4b_env(E4B_FUSE_SWIGLU="1")["sha256"] != _e4b_env(E4B_FUSE_SWIGLU="0")["sha256"]



@pytest.mark.parametrize("name, secret", [
    ("E4B_TOKEN", True), ("E4B_API_KEY", True), ("GNF4_AUTH", True), ("E4B_DB_PASSWORD", True),
    ("E4B_HF_CREDENTIALS", True), ("E4B_FOO_TOKEN", True), ("E4B_HF_AUTH_TOKEN", True),
    ("E4B_MAX_NEW_TOKENS", False), ("E4B_K_SLOTS", False), ("E4B_KEYFRAME", False), ("E4B_INT4_KEEP_NF4", False),
    ("E4B_CHUNKED_LM_LOSS", False),
])
def test_a_credential_name_is_told_by_whole_word(name, secret):
    from loggetta.measure import is_secret_name

    assert is_secret_name(name) is secret


def test_a_credential_value_is_never_recorded_even_if_listed(monkeypatch):
    """The runtime guard behind the test: were a credential name ever put on the value list, e4b_env still records and
    hashes it by name only."""
    from loggetta import measure

    monkeypatch.setattr(measure, "E4B_VALUE_KNOBS", measure.E4B_VALUE_KNOBS + ("E4B_API_KEY",))
    rec = _e4b_env(E4B_API_KEY="sk-0123456789abcdef")
    assert "E4B_API_KEY" in rec["names"] and "E4B_API_KEY" not in rec["values"]
    assert "sk-0123456789abcdef" not in json.dumps(rec)
    assert rec["sha256"] == _e4b_env(E4B_API_KEY="other")["sha256"]
