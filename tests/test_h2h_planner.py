"""bench/h2h_planner.py, the planner head-to-head's driver: Accelerate's table is read as printed, a receipt that is not
the split's is refused, and every term is graded measured / estimated."""
import hashlib
import importlib.util
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPLIT = json.loads((ROOT / "evidence" / "planner-split" / "SPLIT-v3.json").read_text())
#: estimate-memory's table as Accelerate 1.15.0 printed it for ERNIE-4.5-21B-A3B-PT (bench/h2h-planner/PREREG's smoke)
STDOUT = """Loading pretrained config for `baidu/ERNIE-4.5-21B-A3B-PT` from `transformers`...
┌──────────────────────────────────────────────────────┐
│Memory Usage for loading `baidu/ERNIE-4.5-21B-A3B-PT` │
├───────┬─────────────┬──────────┬─────────────────────┤
│ dtype │Largest Layer│Total Size│ Training using Adam │
├───────┼─────────────┼──────────┼─────────────────────┤
│float16│   1.48 GB   │ 40.65 GB │      162.61 GB      │
│  int4 │  378.83 MB  │ 10.16 GB │         N/A         │
└───────┴─────────────┴──────────┴─────────────────────┘
"""


def _tool():
    spec = importlib.util.spec_from_file_location("h2h_planner", ROOT / "bench" / "h2h_planner.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_accelerates_table_is_read_as_printed():
    t = _tool()
    s = t.parse_accelerate(STDOUT)
    assert set(s) == {"float16", "int4"}
    assert s["float16"]["printed"] == {"largest_layer": "1.48 GB", "total": "40.65 GB", "training_adam": "162.61 GB"}
    assert s["float16"]["total_bytes"] == round(40.65 * (1 << 30)) and s["int4"]["largest_layer_bytes"] == round(378.83 * (1 << 20))
    assert s["int4"]["training_adam_bytes"] is None                   # "N/A" stays absent, never zero


def test_a_table_that_is_not_there_is_refused():
    t = _tool()
    with pytest.raises(t.Refused, match="no size table"):
        t.parse_accelerate("Loading pretrained config...\nsomething else\n")


def test_a_receipt_that_is_not_the_splits_is_refused():
    t = _tool()
    row = SPLIT["held_out"][-1]
    with pytest.raises(t.Refused, match="not the split's sha256"):
        t.read_row(row, SPLIT["pinned_at"], lambda commit, path: b'{"not": "it"}')


def test_a_receipt_that_is_the_splits_is_read_at_the_pin():
    t = _tool()
    raw = b'{"comparison": {}}'
    row = {**SPLIT["held_out"][-1], "sha256": hashlib.sha256(raw).hexdigest()}
    seen = []
    assert t.read_row(row, SPLIT["pinned_at"], lambda commit, path: (seen.append((commit, path)), raw)[1]) == {"comparison": {}}
    assert seen == [(SPLIT["pinned_at"][row["repo"]], row["file"])]


def test_every_term_is_graded_measured_over_estimated():
    t = _tool()
    g = t.grade({"device_driver": {"estimated": 100, "measured": 80}, "host": {"estimated": 10, "measured": 11},
                 "device_allocator": {"estimated": 0, "measured": 5}, "note": "not a term"})
    assert g == {"device_driver": 0.8, "host": 1.1, "device_allocator": None}


def test_the_split_has_the_registered_21_rows():
    assert len(SPLIT["held_out"]) == 21 and SPLIT["pinned_at"]["experts4bit-qlora"].startswith("a4b2b98f")


def test_a_release_that_is_not_installed_reads_not_read_today():
    t = _tool()
    have = {"loggetta": "0.7.0", "experts4bit-qlora": "0.57.0", "grouped-nf4-gemm": None}.get
    with pytest.raises(t.Refused, match="not read today: experts4bit-qlora wants 0.58.0, installed 0.57.0; grouped-nf4-gemm"):
        t.check_versions({"loggetta": "0.7.0", "experts4bit-qlora": "0.58.0", "grouped-nf4-gemm": "0.51.0"}, have)
    assert t.check_versions({"loggetta": "0.7.0"}, have) == {"loggetta": "0.7.0"}
