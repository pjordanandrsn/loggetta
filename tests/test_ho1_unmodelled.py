"""HO1's post-hoc detector: arms that padded without buckets where the planner assumes buckets are flagged from what
they recorded, not from their tags (evidence/2026-10-10-heldout-cards, addendum)."""
import importlib.util
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _rows_module():
    spec = importlib.util.spec_from_file_location("ho1_rows", os.path.join(ROOT, "bench", "ho1_rows.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_nine_over_plan_arms_are_flagged_and_the_eighteen_bucketed_are_not():
    m = _rows_module()
    fx = json.load(open(os.path.join(ROOT, "tests", "fixtures", "ho1", "qwen3-4096-fp32-arms.json")))
    flagged = {a["arm_receipt"] for a in fx["arms"] if m.unmodelled_settings(a["arm"], fx["top_k"])}
    expected = {a["arm_receipt"] for a in fx["arms"] if a["expect_unmodelled"]}
    assert len(expected) == 9 and len(fx["arms"]) == 27
    assert flagged == expected


def test_signals():
    m = _rows_module()
    base = {"arm": "fused", "seq": 2048, "micro_batch": 1}
    unbucketed = {**base, "lean_ab": {"lora_path_calls": {"padded": 5, "padded_bucketed": 0}}}
    assert m.unmodelled_settings(unbucketed, 8)                      # 16,384 routed rows: buckets assumed
    assert not m.unmodelled_settings(unbucketed, 4)                  # 8,192: a single padded block is what is priced
    assert not m.unmodelled_settings({**unbucketed, "arm": "reference"}, 8)
    assert not m.unmodelled_settings({**base, "lean_ab": {"lora_path_calls": {"padded": 0, "padded_bucketed": 7}}}, 8)
    old = {**base, "env": {"grouped-nf4-gemm": "0.41.0"}}
    assert m.unmodelled_settings(old, 8)                             # no path recorded, release without buckets
    assert not m.unmodelled_settings({**base, "env": {"grouped-nf4-gemm": "0.42.0"}}, 8)
