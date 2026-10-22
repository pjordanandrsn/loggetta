"""T1 step 3's kit (bench/first_use/): the reducer graded on a REAL first_run.json, the box runner driven end to end over a
stub first_run.sh (including its refusal of a tampered staged file), the host screen on a fake transport, and the staged
manifest checked against the files it pins.

The fixture tests/fixtures/first_use/no-gpu-macos.first_run.json is scripts/first_run.sh's own output on a Mac with no
NVIDIA GPU (exit 3), unedited: PACKAGE_SPEC was a relative path, so it carries no home directory.
"""
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "bench" / "first_use"
REAL = ROOT / "tests" / "fixtures" / "first_use" / "no-gpu-macos.first_run.json"


def load(name):
    spec = importlib.util.spec_from_file_location(name, KIT / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fu = load("fu_reduce")


def real():
    return json.loads(REAL.read_text())


# --------------------------------------------------------------------------------------------- the reducer


def test_the_real_mac_record_is_a_named_refusal_but_not_the_linux_no_gpu_cell():
    r = real()
    assert r["exit"] == 3 and r["gpu"] is None and r["refusal"].startswith("no NVIDIA GPU discovered")
    g = fu.grade("no-gpu", r)
    assert g["gate"] is True                                     # named: exit 3 with its reason, no traceback
    assert g["verdict"] == "CELL_MISMATCH" and "Linux" in g["why"]   # the registered cell is a Linux host


def test_the_same_record_on_linux_meets_the_no_gpu_cell():
    g = fu.grade("no-gpu", {**real(), "platform": "Linux-6.8.0-x86_64-with-glibc2.39"})
    assert g["verdict"] == "MET" and g["meaning"] == "refused: no torch build fits this machine"


@pytest.mark.parametrize("cell", ["below-default", "at-matched", "cpu-torch"])
def test_a_record_with_no_driver_is_a_mismatch_for_every_gpu_cell(cell):
    assert fu.grade(cell, real())["verdict"] == "CELL_MISMATCH"


def test_cells_are_graded_by_the_driver_the_box_read_and_keep_the_listed_one():
    rec = {**real(), "exit": 0, "refusal": None, "torch_index": "cu130", "platform": "Linux",
           "gpu": "NVIDIA GeForce RTX 4090, 595.45.04", "seconds_to_trained": 700,
           "receipt": {"run_id": "OLMoE-r1", "device_check": "checked"}}
    g = fu.grade("at-matched", rec, listed_driver="580.82.07")
    assert g["verdict"] == "CELL_MISMATCH" and g["box_band"] == "current" and g["listed_driver"] == "580.82.07"
    assert fu.grade("current-matched", rec)["verdict"] == "MET"


def test_exit_7_trains_but_is_not_a_pass_of_the_device_check():
    assert fu.EXIT_MEANING[7] == "TRAINED, device check not run (not a pass of it)"
    rec = {**real(), "exit": 7, "refusal": "trained; the device check did not run", "torch_index": "cu130",
           "platform": "Linux", "gpu": "NVIDIA RTX A4000, 580.82.07", "seconds_to_trained": 800,
           "receipt": {"run_id": "r7", "device_check": "not run"}}
    g = fu.grade("at-matched", rec)
    assert g["verdict"] == "MET" and g["gate"] is True and g["device_check"] == "not run"


@pytest.mark.parametrize("exit_code, refusal, named", [
    (4, "the plan was refused (exit 2): torch 2.14 ... cannot use GPU 0", True),
    (4, "", False),                                                       # a refusal must say why
    (6, "Traceback (most recent call last):\n  File ...", False),        # never a traceback
    (2, "refusing: ./x already exists", False),                          # usage is not a named refusal
    (1, "anything", False),
])
def test_the_gate_admits_only_trained_exits_and_named_refusals(exit_code, refusal, named):
    assert fu.named({"exit": exit_code, "refusal": refusal}) is named


def test_reduce_reads_only_the_cells_this_box_ran(tmp_path):
    d = tmp_path / "no-gpu"
    d.mkdir()
    (d / "first_run.json").write_text(json.dumps({**real(), "platform": "Linux"}))
    (tmp_path / "at-matched").mkdir()                                     # ran, but wrote no record
    v = fu.reduce(str(tmp_path), "fu-x")
    assert [c["cell"] for c in v["cells"]] == ["at-matched", "no-gpu"]
    assert v["gate"] == "INCOMPLETE" and v["met"] == 1


# --------------------------------------------------------------------------------------------- the box runner

STUB = r"""#!/bin/bash
# stands in for scripts/first_run.sh: exits by TORCH_INDEX the way the PREREG expects on a below-band box
mkdir -p "$1" || exit 2
case "${TORCH_INDEX:-}" in
  pypi) c=4; t=pypi; why="the plan was refused (exit 2): torch cannot use GPU 0 (driver 575)";;
  cpu)  c=4; t=cpu;  why="the plan was refused (exit 2): torch (a CPU-only build) cannot use GPU 0";;
  "")   c=0; t=cu126; why="";;
esac
if [ -n "${STUB_CRASH:-}" ] && [ -z "${TORCH_INDEX:-}" ]; then
  # execute raised: first_run.sh keeps only the LAST lines of execute.txt as the refusal, never the header
  mkdir -p "$1" && printf 'Traceback (most recent call last):\n  File "x.py", line 1, in <module>\nValueError: boom\n' > "$1/execute.txt"
  c=6; why="the run did not train (exit 1):   File x.py, line 1, in <module> ValueError: boom"
fi
python3 - "$1" "$c" "$t" "$why" <<'PY'
import json, sys
d, c, t, why = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
json.dump({"schema": "first-run/1", "exit": c, "refusal": why or None, "trained": c in (0, 7), "torch_index": t,
           "gpu": "NVIDIA RTX A4000, 575.64.05", "platform": "Linux-6.8", "seconds_to_trained": 640 if c == 0 else None,
           "receipt": {"run_id": "OLMoE-stub", "device_check": "checked"} if c == 0 else None},
          open(d + "/first_run.json", "w"))
PY
exit $c
"""


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def stage(tmp_path, tamper=False):
    w = tmp_path / "box"
    w.mkdir()
    for f in ("fu_run.sh", "fu_reduce.py", "staged-first_use.sha256"):
        shutil.copy(KIT / f, w / f)
    if tamper:
        with open(w / "fu_reduce.py", "a") as f:
            f.write("# changed after review\n")
    (w / "first_run.sh").write_text(STUB)
    (w / "loggetta-0.7.0rc1-py3-none-any.whl").write_bytes(b"not really a wheel")
    env = {**os.environ, "FU_WORK_DIR": str(w), "FU_RUN_ID": "fu-below-1", "FU_RUN_NONCE": "n" * 64,
           "FU_DEADLINE_EPOCH": "9999999999", "FU_CELLS": "below-default below-matched cpu-torch",
           "FU_WHEEL": "loggetta-0.7.0rc1-py3-none-any.whl",
           "FU_WHEEL_SHA256": sha(w / "loggetta-0.7.0rc1-py3-none-any.whl"),
           "FU_FIRST_RUN_SHA256": sha(w / "first_run.sh")}
    return w, env


@pytest.mark.skipif(not shutil.which("perl"), reason="the runner bounds each cell with perl's alarm")
def test_the_runner_runs_each_cell_and_reduces_them(tmp_path):
    w, env = stage(tmp_path)
    r = subprocess.run(["bash", str(w / "fu_run.sh")], env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    v = json.loads((w / "verdict_first_use.json").read_text())
    assert {c["cell"]: c["verdict"] for c in v["cells"]} == {"below-default": "MET", "below-matched": "MET",
                                                              "cpu-torch": "MET"}
    assert v["gate"] == "PASS" and v["times"] == [{"cell": "below-matched", "seconds_to_trained": 640,
                                                    "run_id": "OLMoE-stub", "exit": 0}]
    assert (w / ("FU_SUCCESS." + "n" * 64)).exists() and (w / ("TP_DONE." + "n" * 64)).exists()
    assert json.loads((w / "cells" / "cpu-torch" / "cell.json").read_text())["torch_index_env"] == "cpu"


@pytest.mark.skipif(not shutil.which("perl"), reason="the runner bounds each cell with perl's alarm")
def test_a_crash_whose_refusal_hides_the_traceback_header_fails_the_gate(tmp_path):
    w, env = stage(tmp_path)
    r = subprocess.run(["bash", str(w / "fu_run.sh")], env={**env, "STUB_CRASH": "1"}, capture_output=True, text=True,
                       timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    v = json.loads((w / "verdict_first_use.json").read_text())
    crashed = next(c for c in v["cells"] if c["cell"] == "below-matched")
    assert "Traceback" not in crashed["refusal"]                     # the recorded refusal alone would read "named"
    assert crashed["tracebacks"] == ["below-matched/run/execute.txt"] and crashed["gate"] is False
    assert v["gate"] == "FAIL"


def test_a_deadline_skipped_cell_is_on_record_and_the_run_is_incomplete(tmp_path):
    w, env = stage(tmp_path)
    import time

    r = subprocess.run(["bash", str(w / "fu_run.sh")], env={**env, "FU_DEADLINE_EPOCH": str(int(time.time()) + 60)},
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 40                                         # no cell ran
    v = json.loads((w / "verdict_first_use.json").read_text())      # but the reduce still recorded every cell
    assert [(c["cell"], c["verdict"]) for c in v["cells"]] == [("below-default", "MISSING"), ("below-matched", "MISSING"),
                                                               ("cpu-torch", "MISSING")]
    assert all(c["skipped"].startswith("deadline") for c in v["cells"]) and v["gate"] == "INCOMPLETE"


def test_one_ran_and_one_skipped_reads_incomplete_never_pass(tmp_path):
    ran = tmp_path / "cells" / "at-matched"
    ran.mkdir(parents=True)
    (ran / "first_run.json").write_text(json.dumps({"exit": 0, "refusal": None, "torch_index": "cu130", "platform": "Linux",
                                                    "gpu": "NVIDIA RTX A4000, 580.82.07", "seconds_to_trained": 9,
                                                    "receipt": {"run_id": "r", "device_check": "checked"}}))
    skipped = tmp_path / "cells" / "at-default"
    skipped.mkdir()
    (skipped / "cell.json").write_text(json.dumps({"cell": "at-default", "skipped": "deadline: 900s left, 1800s needed"}))
    v = fu.reduce(str(tmp_path / "cells"), "fu-at-1")
    assert v["met"] == 1 and v["gate"] == "INCOMPLETE"


def test_the_runner_refuses_a_tampered_staged_file_before_any_cell(tmp_path):
    w, env = stage(tmp_path, tamper=True)
    r = subprocess.run(["bash", str(w / "fu_run.sh")], env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 9 and "STAGED FILES DIFFER" in r.stdout
    assert not list((w / "cells").iterdir())


@pytest.mark.parametrize("field", ["FU_WHEEL_SHA256", "FU_FIRST_RUN_SHA256"])
def test_the_runner_refuses_a_release_candidate_file_with_another_hash(tmp_path, field):
    w, env = stage(tmp_path)
    r = subprocess.run(["bash", str(w / "fu_run.sh")], env={**env, field: "0" * 64}, capture_output=True, text=True,
                       timeout=60)
    assert r.returncode == 9 and "HASH MISMATCH" in r.stdout


def test_the_runner_never_runs_the_no_gpu_cell_on_a_rented_box(tmp_path):
    w, env = stage(tmp_path)
    r = subprocess.run(["bash", str(w / "fu_run.sh")], env={**env, "FU_CELLS": "no-gpu"}, capture_output=True,
                       text=True, timeout=60)
    assert r.returncode == 78 and "never on a rented GPU box" in r.stdout


def test_the_staged_manifest_pins_the_files_as_committed():
    lines = [ln.split() for ln in (KIT / "staged-first_use.sha256").read_text().splitlines() if ln.strip()]
    assert sorted(name for _h, name in lines) == ["fu_reduce.py", "fu_run.sh"]
    for want, name in lines:
        assert sha(KIT / name) == want, f"{name} changed without re-pinning staged-first_use.sha256"


def test_the_prereg_carries_the_exit_7_line_verbatim():
    assert "7 = TRAINED, device check not run (not a pass of it)" in (KIT / "PREREG-first_use.md").read_text()


# --------------------------------------------------------------------------------------------- the host screen


class FakeVast:
    def __init__(self, offers, status=200):
        self.offers, self.status, self.queries = offers, status, []

    def request(self, method, path, *, query=None, body=None, timeout=30.0):
        q = json.loads(query["q"])
        self.queries.append(q)
        if self.status != 200:
            return self.status, {"error": "busy"}
        want = q.get("machine_id", {}).get("eq")
        return 200, {"offers": [o for o in self.offers if o["gpu_name"] == q["gpu_name"]["eq"]
                                and (want is None or o["machine_id"] == want)]}


def flt(gpu, *, min_disk_gb, min_ram_gb):
    return {"gpu_name": {"eq": gpu}, "disk_space": {"gte": min_disk_gb}}


OFFERS = [
    {"id": 1, "machine_id": 11, "gpu_name": "RTX A4000", "driver_version": "575.64.05", "dph_total": 0.12},
    {"id": 2, "machine_id": 12, "gpu_name": "RTX 4090", "driver_version": "580.82.07", "dph_total": 0.30},
    {"id": 3, "machine_id": 13, "gpu_name": "RTX 3090", "driver_version": "595.45.04", "dph_total": 0.20},
    {"id": 4, "machine_id": 14, "gpu_name": "RTX 3090", "driver_version": "510.10", "dph_total": 0.05},
]


def test_the_screen_buckets_by_listed_driver_and_reads_each_pin_by_machine():
    scr = load("fu_screen")
    t = FakeVast(OFFERS)
    out = scr.screen(t, filter_fn=flt, reads=2, stable_reads=3, per_band=2, sleep=lambda s: None)
    got = {b: [c["machine_id"] for c in cs] for b, cs in out["candidates"].items()}
    assert got == {"below": ["11"], "at": ["12"], "current": ["13"]}          # 510 is in no band
    assert all(c["stable"] == "3/3" and c["stable_ok"] for cs in out["candidates"].values() for c in cs)
    pinned = [q for q in t.queries if "machine_id" in q]
    assert len(pinned) == 9 and {q["machine_id"]["eq"] for q in pinned} == {11, 12, 13}
    assert out["sample"].startswith("4 offers seen across 8 reads")             # its sample, never "the market"


def test_the_screen_never_reads_an_error_as_zero_offers():
    scr = load("fu_screen")
    with pytest.raises(RuntimeError, match="HTTP 429"):
        scr.screen(FakeVast(OFFERS, status=429), filter_fn=flt, reads=1, stable_reads=1, per_band=1,
                   sleep=lambda s: None)
