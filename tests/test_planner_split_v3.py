"""evidence/planner-split/SPLIT-v3.json: version 3 of the frozen split carries version 2 unchanged and adds HO3's held-out rows, its
MISS rows included; its cohort counts and boxes are the rows'; every cohort's boxes ran on distinct machines, as the committed store
check records; and that check refuses a shared, missing or contradictory machine."""
import importlib.util
import json
import pathlib
import re
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1] / "evidence" / "planner-split"
V2 = json.loads((ROOT / "SPLIT-v2.json").read_text())
V3 = json.loads((ROOT / "SPLIT-v3.json").read_text())
MACH = json.loads((ROOT / "SPLIT-v3-machines.json").read_text())
HEX64 = re.compile(r"^[0-9a-f]{64}$")
HO3_BOXES = {"ho3-5090-r1": "RTX 5090", "ho3-a4000-1": "RTX A4000", "ho3-a4000-r2": "RTX A4000"}


def _checker():
    spec = importlib.util.spec_from_file_location("check_split_machines", ROOT / "check_split_machines.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _box(r):
    return r.get("box") or r["id"].split("/")[0]


def test_v3_carries_v2_unchanged():
    assert V3["version"] == 3 and V2["version"] == 2
    for k in ("schema", "rule", "derivation", "default_band"):
        assert V3[k] == V2[k], k
    assert V3["held_out"][: len(V2["held_out"])] == V2["held_out"]
    assert V3["pinned_at"]["loggetta"] == V2["pinned_at"]["loggetta"]
    assert V3["fills_next"][: len(V2["fills_next"])] == V2["fills_next"]


def test_ho3_rows_are_unseen_and_recorded_whatever_they_read():
    new = V3["held_out"][len(V2["held_out"]):]
    assert len(new) == 12 and len({r["file"] for r in new}) == 12
    assert {r["box"] for r in new} == set(HO3_BOXES)
    for r in new:
        assert r["tag"] == "UNSEEN" and r["repo"] == "experts4bit-qlora" and r["card"] == HO3_BOXES[r["box"]]
        assert r["file"].startswith(f"bench/ho3/receipts/{r['box']}/public/")
        assert HEX64.match(r["sha256"]) and HEX64.match(r["store_sha256"]) and r["sha256"] != r["store_sha256"]
        assert "inferred, not counted" in r["loss_branch"]
        host = r["placement"] == "host offload (experts on host)"
        assert host == ("host offload" in r["cohort"]) and host == ("host_over_estimate" in r)
        assert host == (r["model"] == "baidu/ERNIE-4.5-21B-A3B-PT" and r["card"] == "RTX A4000")
        assert r["outcome"] in ("COVERED", "HOST_UNDER")
        if host:
            assert (r["outcome"] == "HOST_UNDER") == (r["host_over_estimate"] > 1), r["id"]
    # the MISS rows are entered: ERNIE's host estimate was low on 3 of its 4 A4000 rows
    assert sorted(r["outcome"] for r in new if "host_over_estimate" in r) == ["COVERED", "HOST_UNDER", "HOST_UNDER", "HOST_UNDER"]
    assert all(r["outcome"] == "COVERED" for r in new if "host_over_estimate" not in r)


def test_cohort_counts_boxes_and_the_two_row_rule():
    counts, boxes = {}, {}
    for r in V3["held_out"]:
        counts[r["cohort"]] = counts.get(r["cohort"], 0) + 1
        boxes.setdefault(r["cohort"], set()).add(_box(r))
    assert {c: v["held_out_rows"] for c, v in V3["cohorts"].items()} == counts
    assert {c: v["boxes"] for c, v in V3["cohorts"].items()} == {c: len(b) for c, b in boxes.items()}
    for c, v in V3["cohorts"].items():
        assert (v["band"] == "unmeasured") == (v["held_out_rows"] < 2), c


def test_distinct_machines_is_the_committed_store_check():
    assert MACH["version"] == 3 and set(MACH["cohorts"]) == set(V3["cohorts"])
    for c, v in V3["cohorts"].items():
        m = MACH["cohorts"][c]
        assert v["distinct_machines"] is m["distinct_machines"] is True, c
        assert m["n_boxes"] == v["boxes"] == len(m["boxes"]) and m["n_machines"] == m["n_boxes"], c
        assert sorted({_box(r) for r in V3["held_out"] if r["cohort"] == c}) == m["boxes"], c


def test_the_store_check_names_no_machine():
    # boxes and counts only: no machine id leaves the store
    assert set(MACH) == {"schema", "split", "version", "source", "cohorts"}
    for m in MACH["cohorts"].values():
        assert set(m) == {"boxes", "n_boxes", "n_machines", "distinct_machines"}


def _store(tmp_path, boxes):
    """A fake store: ``boxes`` maps box -> (instance machine or None, receipt machine or None)."""
    for i, (box, (inst, rec)) in enumerate(boxes.items()):
        d = tmp_path / "receipts" / "experts4bit-qlora" / f"2026-10-2{i}" / box
        d.mkdir(parents=True)
        if inst is not None:
            (d / "instance.json").write_text(json.dumps({"machine_id": inst}))
        (d / "receipt.json").write_text(json.dumps({"environment": {"vast_machine_id": rec} if rec is not None else {}}))
    return tmp_path


def _split(*boxes):
    return {"version": 9, "held_out": [{"id": f"{b}/x/device/1024", "box": b, "repo": "experts4bit-qlora", "cohort": "C"} for b in boxes]}


def test_the_check_passes_distinct_machines_from_either_record(tmp_path):
    store = _store(tmp_path, {"a": (11, None), "b": (None, 12), "c": (13, 13)})
    assert _checker().check(_split("a", "b", "c"), str(store)) == {
        "C": {"boxes": ["a", "b", "c"], "n_boxes": 3, "n_machines": 3, "distinct_machines": True}}


def test_the_check_reads_a_shared_machine_as_not_distinct(tmp_path, capsys):
    store = _store(tmp_path, {"a": (11, None), "b": (None, 11)})
    assert _checker().check(_split("a", "b"), str(store))["C"]["distinct_machines"] is False
    split = tmp_path / "split.json"
    split.write_text(json.dumps(_split("a", "b")))
    mod = _checker()
    argv, sys.argv = sys.argv, ["check", str(split), str(store), "--out", str(tmp_path / "out.json")]
    try:
        assert mod.main() == 1
    finally:
        sys.argv = argv
    assert "11" not in (tmp_path / "out.json").read_text()          # the refusal never names the machine
    assert "11" not in capsys.readouterr().err


@pytest.mark.parametrize("boxes, why", [
    ({"a": (11, None)}, "0 run directories"),                        # b is not in the store
    ({"a": (11, None), "b": (None, None)}, "no recorded machine"),
    ({"a": (11, None), "b": (12, 13)}, "different machines"),
])
def test_the_check_refuses_a_box_it_cannot_read(tmp_path, boxes, why):
    store, mod = _store(tmp_path, boxes), _checker()
    with pytest.raises(mod.Unreadable, match=why):
        mod.check(_split("a", "b"), str(store))
