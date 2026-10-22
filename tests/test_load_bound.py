"""The load-time host bound: the plan's host estimate prices the steady state, and loading also maps the checkpoint's
weight files, whose pages count in the process's RSS. The bound is their sum. It warns when it exceeds what the
machine has available and never refuses; execute reads what is available again before loading and the receipt grades
the bound against the run's measured peak RSS.

The numbers in the WSL2 case come from one run's receipt: a 15.3 GiB WSL2 VM with 14.5 GiB available, a host estimate
of 6.38 GiB, a 12.89 GiB checkpoint, and a peak RSS of 13.1 GiB.
"""
import dataclasses
import json
import os

import pytest

from loggetta import model as model_mod
from loggetta.backends import experts4bit
from loggetta.execution import available_before_load, execute, grade_load_bound, summarize
from loggetta.hardware import Fact, HardwareProfile, Host
from loggetta.plan import ExecutionPlan
from loggetta.planner import load_bound, load_bound_warning

GiB = 1 << 30


def host(avail_gib, source="/proc/meminfo MemAvailable"):
    r = lambda v: Fact(v, "reported")  # noqa: E731
    return Host(cpu_model=r("test cpu"), cpus=r(8), memory_total=r(16 * GiB),
                memory_available=Fact(None if avail_gib is None else int(avail_gib * GiB), "reported", source),  # (value, kind, note)
                memory_limit=r(None))


def hwp(avail_gib):
    return HardwareProfile(gpus=(), host=host(avail_gib), platform="Linux x86_64")


# --------------------------------------------------------------------------------------------- which files count


@pytest.mark.parametrize("listing, picked", [
    ({"model-00001-of-00002.safetensors": 5, "model-00002-of-00002.safetensors": 3, "pytorch_model.bin": 8,
      "config.json": 1}, {"model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"}),
    ({"pytorch_model-00001-of-00002.bin": 4, "pytorch_model-00002-of-00002.bin": 4, "tokenizer.json": 2},
     {"pytorch_model-00001-of-00002.bin", "pytorch_model-00002-of-00002.bin"}),
    ({"model.safetensors": 4, "original/model.safetensors": 4}, {"model.safetensors"}),     # a folder is not loaded
    ({"README.md": 1}, set()),
])
def test_the_weight_files_are_the_ones_a_load_reads(listing, picked):
    assert set(model_mod._weight_files(listing)) == picked


def test_a_local_directory_is_listed(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"x" * 1000)
    (tmp_path / "config.json").write_text("{}")
    got = model_mod.checkpoint_size(str(tmp_path))
    assert got == {"bytes": 1000, "files": 1, "source": f"local directory {tmp_path.name}"}   # no home path recorded


def test_an_unreadable_directory_is_no_answer_not_a_traceback(tmp_path):
    d = tmp_path / "ckpt"
    d.mkdir()
    (d / "model.safetensors").write_bytes(b"x")
    d.chmod(0)
    try:
        if os.access(d, os.R_OK):
            pytest.skip("running as a user who reads a mode-0 directory (root)")
        got = model_mod.checkpoint_size(str(d))
    finally:
        d.chmod(0o755)
    assert got == {"bytes": None, "files": 0, "source": "local directory ckpt: PermissionError"}


def test_no_answer_is_none_with_the_reasons_and_never_raises(monkeypatch):
    import huggingface_hub

    def boom(*a, **k):
        raise OSError("offline")

    monkeypatch.setattr(huggingface_hub.HfApi, "model_info", boom)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", boom)
    got = model_mod.checkpoint_size("org/model", revision="abc")
    assert got["bytes"] is None and got["files"] == 0
    assert "Hub API: OSError" in got["source"] and "local Hub cache: OSError" in got["source"]


def test_the_hub_answer_is_file_metadata_only(monkeypatch):
    import huggingface_hub

    seen = {}

    class Sib:
        def __init__(self, name, size):
            self.rfilename, self.size = name, size

    class Info:
        sha = "f" * 40
        siblings = [Sib("model-00001-of-00002.safetensors", 7), Sib("model-00002-of-00002.safetensors", 5),
                    Sib("config.json", 1)]

    def model_info(self, repo, *, revision=None, files_metadata=False, timeout=None):
        seen.update(repo=repo, revision=revision, files_metadata=files_metadata, timeout=timeout)
        return Info()

    monkeypatch.setattr(huggingface_hub.HfApi, "model_info", model_info)
    got = model_mod.checkpoint_size("org/model", revision="abc")
    assert got == {"bytes": 12, "files": 2, "source": f"Hub API file metadata, org/model@{'f' * 40}"}
    assert seen == {"repo": "org/model", "revision": "abc", "files_metadata": True, "timeout": 10}


# --------------------------------------------------------------------------------------------- the bound


def test_the_wsl2_receipt_exceeds_its_bound_check_and_the_bound_held():
    est, ckpt = int(6.38 * GiB), 13838721960
    lb = load_bound(est, hwp(14.5), {"bytes": ckpt, "files": 3, "source": "Hub API file metadata"})
    assert lb["bound_bytes"] == est + ckpt and lb["exceeds_available"] is True
    assert lb["basis"].startswith("inferred bound") and "upper" not in lb["basis"]
    text = load_bound_warning(lb)
    for part in ("19.27 GiB", "6.38 GiB", "12.89 GiB", "14.50 GiB", "/proc/meminfo MemAvailable", "OOM-killed",
                 "inferred bound, not a measurement", "WSL2"):
        assert part in text, part
    graded = grade_load_bound(_with_bound(lb), {"available_at_execute_bytes": int(14.5 * GiB)},
                              {"host_peak_bytes": int(13.1 * GiB)})
    assert graded["held"] is True and graded["measured_peak_bytes"] == int(13.1 * GiB)


@pytest.mark.parametrize("checkpoint", [None, {"bytes": None, "files": 0, "source": "Hub API: OSError"}])
def test_an_unknown_checkpoint_prices_nothing(checkpoint):
    assert load_bound(GiB, hwp(14.5), checkpoint) is None


def test_an_unknown_available_amount_cannot_exceed():
    lb = load_bound(GiB, hwp(None), {"bytes": GiB, "files": 1, "source": "s"})
    assert lb["exceeds_available"] is False and lb["available_at_plan_bytes"] is None


def test_a_bound_within_what_is_available_does_not_exceed():
    assert load_bound(2 * GiB, hwp(40), {"bytes": 12 * GiB, "files": 3, "source": "s"})["exceeds_available"] is False


# --------------------------------------------------------------------------------------------- the plan


@pytest.fixture(scope="module")
def topo():
    tr = pytest.importorskip("transformers")
    pytest.importorskip("experts4bit_qlora.recipe")
    from loggetta import describe_model

    cfg = tr.Qwen3MoeConfig(hidden_size=1024, intermediate_size=2048, moe_intermediate_size=768, num_experts=128,
                            num_experts_per_tok=4, num_hidden_layers=8, num_attention_heads=8, num_key_value_heads=4,
                            head_dim=128, vocab_size=32000, max_position_embeddings=4096, decoder_sparse_step=1)
    return describe_model(cfg)


def _hw(ram_gib):
    from tests.test_planner import hw

    return hw(ram_gib=ram_gib)


def test_the_plan_warns_records_and_stays_feasible(topo):
    from loggetta import Workload, plan

    p = plan(topo, _hw(14.5), Workload(seq_len=512), checkpoint={"bytes": 13 * GiB, "files": 3, "source": "s"})
    assert p.status == "feasible"                                          # a warning, never a refusal
    lb = p.provenance["load_bound"]
    assert lb["host_estimate_bytes"] == p.selected.host_bytes and lb["exceeds_available"]
    assert sum("load-time host bound" in w for w in p.warnings) == 1
    assert ExecutionPlan.from_dict(json.loads(p.to_json())).provenance["load_bound"] == lb   # a saved plan keeps it


def test_a_roomy_host_records_the_bound_without_a_warning(topo):
    from loggetta import Workload, plan

    p = plan(topo, _hw(64), Workload(seq_len=512), checkpoint={"bytes": 13 * GiB, "files": 3, "source": "s"})
    assert p.provenance["load_bound"]["exceeds_available"] is False
    assert not any("load-time host bound" in w for w in p.warnings)


def test_no_checkpoint_size_leaves_the_plan_as_it_was(topo):
    from loggetta import Workload, plan

    a = plan(topo, _hw(14.5), Workload(seq_len=512))
    b = plan(topo, _hw(14.5), Workload(seq_len=512), checkpoint={"bytes": None, "files": 0, "source": "offline"})
    assert "load_bound" not in a.provenance and a.to_dict() == b.to_dict()


# --------------------------------------------------------------------------------------------- execute and the receipt


def _with_bound(lb):
    from tests.test_execution import a_plan

    p = a_plan()
    return dataclasses.replace(p, provenance={**p.provenance, "load_bound": lb})


def _lb(bound_gib, exceeds=False):
    return {"basis": "inferred bound: host estimate + checkpoint weight files on disk; not measured, graded per receipt",
            "bound_bytes": int(bound_gib * GiB), "host_estimate_bytes": GiB, "checkpoint_bytes": int(bound_gib * GiB) - GiB,
            "checkpoint_files": 1, "checkpoint_source": "s", "available_at_plan_bytes": 14 * GiB,
            "available_at_plan_source": "src", "exceeds_available": exceeds}


@pytest.mark.parametrize("peak, held", [(int(13.1 * GiB), True), (int(20 * GiB), False), (None, None)])
def test_the_receipt_grades_the_bound_against_the_measured_peak(peak, held):
    g = grade_load_bound(_with_bound(_lb(19.27)), None, {"host_peak_bytes": peak})
    assert g["held"] is held and g["measured_peak_source"].startswith("VmHWM")


def test_a_plan_without_a_bound_grades_nothing_and_reads_nothing(monkeypatch):
    from tests.test_execution import a_plan

    import loggetta.hardware as hardware

    monkeypatch.setattr(hardware, "_probe_host", lambda: pytest.fail("read for a plan without a bound"))
    assert available_before_load(a_plan()) is None
    assert grade_load_bound(a_plan(), None, {"host_peak_bytes": GiB}) is None


def test_execute_reads_what_is_available_before_the_backend_loads_and_grades_it(monkeypatch):
    from tests.test_execution import hw

    import loggetta.hardware as hardware

    order = []

    def probe_host():
        order.append("read")
        return host(10.0, source="cgroup v2: limit - usage + reclaimable page cache")

    def run(pl, *, seed, log, adapter_dir, run_id):
        order.append("run")
        return {"status": "OK", "measured": {"host_peak_bytes": 12 * GiB}, "correctness": {}, "engaged": {},
                "data": {}}

    monkeypatch.setattr(hardware, "_probe_host", probe_host)
    monkeypatch.setattr(experts4bit, "executor", lambda kind: run if kind == "train" else None)
    logged = []
    r = execute(_with_bound(_lb(19.27)), hardware=hw(), prov={"sources": {}}, log=logged.append)
    assert order == ["read", "run"]
    lb = r["load_bound"]
    assert lb["available_at_execute_bytes"] == 10 * GiB and lb["available_at_execute_source"].startswith("cgroup v2")
    assert lb["held"] is True and lb["bound_bytes"] == int(19.27 * GiB)
    assert any("exceeds the 10.00 GiB available now" in x for x in logged)


def test_the_summary_labels_the_bound_inferred_and_the_peak_measured():
    receipt = {"run_id": "r", "status": "OK", "comparison": {},
               "measured": {"s_per_step_median": 1.0, "timed_steps": 1, "tokens_per_s": 1.0},
               "correctness": {"losses": [1.0, 0.5], "loss_first_third_mean": 1.0, "loss_last_third_mean": 0.5,
                               "all_finite": True, "frozen_expert_bytes_unchanged": True, "adapters_moved": True},
               "load_bound": {**_lb(19.27), "measured_peak_bytes": int(13.1 * GiB), "held": True,
                              "available_at_execute_bytes": int(14.5 * GiB)}}
    text = summarize(receipt)
    assert ("load bound 19.27 GiB [inferred bound] vs peak RSS 13.10 GiB [measured]: held; "
            "available before load 14.50 GiB") in text
