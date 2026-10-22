"""evidence/planner-split/SPLIT-v2.json: version 2 of the frozen split carries version 1 unchanged and adds HO2's held-out rows; its
cohort counts are the rows', and a cohort is eligible for a band only with at least 2 held-out rows."""
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1] / "evidence" / "planner-split"
V1 = json.loads((ROOT / "SPLIT.json").read_text())
V2 = json.loads((ROOT / "SPLIT-v2.json").read_text())
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def test_v2_carries_v1_unchanged():
    assert V2["version"] == 2 and V1["version"] == 1
    assert V2["derivation"] == V1["derivation"] and V2["rule"] == V1["rule"] and V2["schema"] == V1["schema"]
    assert V2["held_out"][: len(V1["held_out"])] == V1["held_out"]
    assert all(V2["cohorts"][c] == v for c, v in V1["cohorts"].items())


def test_ho2_rows_are_unseen_device_rows_cited_by_sha256():
    new = V2["held_out"][len(V1["held_out"]):]
    assert len(new) == 8
    for r in new:
        assert r["tag"] == "UNSEEN" and r["placement"] == "resident (device)" and r["repo"] == "experts4bit-qlora"
        assert r["file"].startswith("bench/ho2/receipts/") and "/public/" in r["file"]
        assert HEX64.match(r["sha256"]) and HEX64.match(r["store_sha256"]) and r["sha256"] != r["store_sha256"]
        assert "inferred, not counted" in r["loss_branch"]
    assert len({r["file"] for r in new}) == 8


def test_cohort_counts_match_the_rows_and_the_two_row_rule():
    counts = {}
    for r in V2["held_out"]:
        counts[r["cohort"]] = counts.get(r["cohort"], 0) + 1
    assert {c: v["held_out_rows"] for c, v in V2["cohorts"].items()} == counts
    for c, v in V2["cohorts"].items():
        assert (v["band"] == "unmeasured") == (v["held_out_rows"] < 2), c
