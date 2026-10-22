"""The receipt records which loss branch the training forwards took, COUNTED from experts4bit-qlora's chunked-LM-loss
counters over the run, with what the size gate read (threshold, chunk, head width). A row near the gate then states its
branch from its own counters, not by inference from its peak. CPU only: the trainer runs with CUDA stubbed."""
import pytest
from torch import nn

C = pytest.importorskip("experts4bit_qlora.engines.chunked_lm_loss")

from test_adapter_v2 import FACTS, plan_with, stubbed_trainer  # noqa: E402


def counting(monkeypatch, **calls):
    """A training loop that counts forwards the way the patched loss does."""
    def train(model, trainable, model_id, w, smi, meas, **k):
        for key, n in calls.items():
            C.CHUNKED_LM_LOSS_STATS[key] = C.CHUNKED_LM_LOSS_STATS.get(key, 0) + n
        return {"status": "OK", "measured": meas, "correctness": {}, "data": {}}
    return train


@pytest.mark.parametrize("calls, branch", [
    ({"chunked_calls": 20}, "chunked"),
    ({"small_calls": 20}, "stock"),                       # under the auto gate: stock
    ({"stock_calls": 20}, "stock"),
    ({"chunked_calls": 12, "small_calls": 8}, "mixed"),   # e.g. rows on both sides of the gate
    ({}, "not counted: no training forward reached the chunked-loss patch"),
])
def test_the_receipt_counts_the_branch_over_this_run_only(monkeypatch, calls, branch):
    C.CHUNKED_LM_LOSS_STATS["chunked_calls"] = C.CHUNKED_LM_LOSS_STATS.get("chunked_calls", 0) + 7   # an earlier run
    T = stubbed_trainer(monkeypatch, report=FACTS, train=counting(monkeypatch, **calls))
    out = T.run(plan_with(), adapter_dir=None, run_id="r")
    rec = out["engaged"]["chunked_lm_loss"]
    assert rec["branch"] == branch
    assert {k: v for k, v in rec["calls"].items() if v} == {k: v for k, v in calls.items() if v}   # the earlier 7 not counted
    assert rec["gate_bytes"] == C.chunked_lm_loss_min_bytes() and rec["chunk_tokens"] == C.chunked_lm_loss_requested()
    assert "error" not in rec


def test_the_receipt_records_the_head_width_the_gate_reads(monkeypatch):
    T = stubbed_trainer(monkeypatch, report=FACTS, train=counting(monkeypatch, chunked_calls=1))
    from experts4bit_qlora import recipe

    real = recipe.prepare_qlora_training
    def prepare(*a, **k):
        prep = real(*a, **k)
        prep.model.lm_head = nn.Linear(8, 65536, bias=False, device="meta")   # LFM2's head width
        return prep
    monkeypatch.setattr(recipe, "prepare_qlora_training", prepare)
    out = T.run(plan_with(), adapter_dir=None, run_id="r")
    assert out["engaged"]["chunked_lm_loss"]["logits_width"] == 65536


def test_recording_never_fails_a_run_that_trained(monkeypatch):
    T = stubbed_trainer(monkeypatch, report=FACTS, train=counting(monkeypatch, chunked_calls=1))
    monkeypatch.setattr(C, "chunked_lm_loss_min_bytes", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    out = T.run(plan_with(), adapter_dir=None, run_id="r")
    assert out["status"] == "OK" and out["engaged"]["chunked_lm_loss"]["error"] == "RuntimeError: boom"


def test_the_report_from_prepare_is_not_mutated(monkeypatch):
    report = dict(FACTS)
    T = stubbed_trainer(monkeypatch, report=report, train=counting(monkeypatch, chunked_calls=1))
    out = T.run(plan_with(), adapter_dir=None, run_id="r")
    assert "chunked_lm_loss" in out["engaged"] and "chunked_lm_loss" not in report
