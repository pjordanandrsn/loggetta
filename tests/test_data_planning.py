"""What a data profile means for a training plan: steps from epochs, whether the data covers the plan, which examples
packing splits, and a refusal before anything about the machine or the model is priced. Pure arithmetic."""
from types import SimpleNamespace

import pytest

from loggetta import Constraints, Workload, plan
from loggetta.data import TrainingData
from loggetta.planner import resolve_data
from test_execution import hw

DATA = TrainingData("train.jsonl").to_dict()


def profile(tokens=100_000, rows=1_000, histogram=((448, 512, 990), (4032, 4096, 9), (8192, 16384, 1)), **data):
    return {"schema": "data-profile/1", "rows": rows, "tokens": tokens, "histogram": [list(b) for b in histogram],
            "options": {**DATA, **data}, "format": "text", "split": "train", "source": {"kind": "local"},
            "lengths": {"min": 450, "p50": 500, "p90": 510, "p99": 4000, "max": 9000, "mean": 100.0}}


def test_without_a_profile_nothing_changes():
    w = Workload(data=DATA)
    assert resolve_data(w, None) == (w, [], [], None)


def test_epochs_become_steps():
    w, reasons, _, refusal = resolve_data(Workload(seq_len=512, micro_batch=2, grad_accum=4, data=DATA, epochs=2),
                                          profile(tokens=100_000))
    assert w.steps == 2 * 100_000 // 4096 == 48 and refusal is None
    assert reasons[0].startswith("steps 48: 2 epoch(s) of 100,000 tokens at 4,096 tokens per optimizer step")
    w, *_ = resolve_data(Workload(seq_len=4096, data=DATA, epochs=0.01), profile(tokens=100_000))
    assert w.steps == 1                                                          # never zero


def test_a_plan_that_reads_past_the_data_is_refused_with_what_would_fit():
    _, reasons, _, refusal = resolve_data(Workload(seq_len=512, steps=400, data=DATA), profile(tokens=100_000))
    assert "2.05 passes" in reasons[0]
    why, suggestions = refusal
    assert "repeating it is not allowed" in why[0]
    assert suggestions[0] == "--steps 195 reads it once" and "--repeat-data allows repeating it" in suggestions
    assert resolve_data(Workload(seq_len=512, steps=400, data={**DATA, "repeat": True}),
                        profile(tokens=100_000, repeat=True))[3] is None


def test_data_smaller_than_one_step_says_so():
    _, _, _, (why, suggestions) = resolve_data(Workload(seq_len=4096, grad_accum=8, data=DATA, epochs=1),
                                               profile(tokens=10_000))
    assert "less than one optimizer step reads (32,768" in why[0]
    assert suggestions[0] == "a shorter seq, micro-batch or grad-accum"


def test_examples_longer_than_seq_are_counted_exactly_on_bin_edges():
    _, _, warnings, _ = resolve_data(Workload(seq_len=512, steps=1, data=DATA), profile())
    assert warnings == ["10 of 1,000 examples are longer than seq 512 tokens: packing always splits them across rows, "
                        "so their later tokens train on a cut context"]
    _, _, warnings, _ = resolve_data(Workload(seq_len=500, steps=1, data=DATA), profile())
    assert warnings[0].startswith("10-1,000 of 1,000 examples")                 # 500 is inside a bin: a range
    assert resolve_data(Workload(seq_len=16384, steps=1, data=DATA), profile())[2] == []


def test_a_profile_for_other_options_is_a_caller_error():
    with pytest.raises(ValueError, match="other dataset options"):
        resolve_data(Workload(data=DATA), profile(shuffle=True))


def test_the_data_refusal_comes_before_the_machine_or_the_model():
    # a model no backend could load and no GPU: the data answer still comes first, and it is the plan's answer
    topo = SimpleNamespace(model="m", model_type="t", revision=None, convention=None, summary=lambda: "", n_layers=1,
                           expert_stacks=(), n_experts=0, top_k=None, expert_numel=0, dense_numel=0,
                           loader_refusal="not loadable", provenance={})
    p = plan(topo, hw(gpus=0), Workload(seq_len=512, steps=400, data=DATA), Constraints(),
             data_profile=profile(tokens=100_000))
    assert p.status == "refused" and "repeating it is not allowed" in p.refusal["reasons"][0]
    assert p.refusal["suggestions"][0] == "--steps 195 reads it once"
    assert p.data_profile["tokens"] == 100_000 and "100,000 tokens" in p.render()
