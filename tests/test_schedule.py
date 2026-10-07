"""The learning-rate schedule is a pure function of the workload: warmup, cosine decay to 10% of the peak, or constant."""
import json
import math
from dataclasses import replace

import pytest

from loggetta.plan import ExecutionPlan, Workload
from loggetta.schedule import describe, learning_rates, lr_factor, warmup_steps
from test_execution import a_plan


def test_cosine_warms_up_then_decays_to_the_floor():
    w = Workload(steps=100, learning_rate=1e-4, lr_schedule="cosine")
    lrs = learning_rates(w)
    assert warmup_steps(w) == 3                                         # 3% of the steps
    assert lrs[:3] == pytest.approx([1e-4 / 3, 2e-4 / 3, 1e-4])         # never zero, peak at the end of warmup
    assert lrs[3] == pytest.approx(1e-4) and lrs[-1] == pytest.approx(1e-5)
    assert all(a >= b for a, b in zip(lrs[3:], lrs[4:]))                # monotone decay
    assert lrs[3 + 48] == pytest.approx(1e-5 + 0.9e-4 * 0.5 * (1 + math.cos(math.pi * 48 / 96)))


def test_short_runs_and_explicit_warmup():
    assert learning_rates(Workload(steps=1, lr_schedule="cosine")) == [2e-4]       # one step: warmup 1, at the peak
    assert warmup_steps(Workload(steps=10, lr_schedule="cosine", warmup_steps=50)) == 10
    assert learning_rates(Workload(steps=4, warmup_steps=2)) == pytest.approx([1e-4, 2e-4, 2e-4, 2e-4])
    assert learning_rates(Workload(steps=4)) == [2e-4] * 4                          # constant, as before
    assert lr_factor(5, 10, 0, "cosine") == pytest.approx(0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * 5 / 9)))


def test_the_plan_says_it_and_old_plans_keep_their_shape():
    assert describe(Workload()) == "learning rate 0.0002"
    assert describe(Workload(steps=100, lr_schedule="cosine")) == ("learning rate 0.0002: linear warmup over 3 "
                                                                   "step(s), then cosine decay to 2e-05")
    old = a_plan().to_dict()
    assert "lr_schedule" not in old["workload"] and "warmup_steps" not in old["workload"]
    plan = replace(a_plan(), workload=Workload(lr_schedule="cosine", warmup_steps=2))
    assert ExecutionPlan.from_dict(json.loads(plan.to_json())).to_json() == plan.to_json()
    with pytest.raises(ValueError, match="lr_schedule must be one of"):
        Workload(lr_schedule="linear")
    with pytest.raises(ValueError, match="warmup_steps"):
        Workload(warmup_steps=-1)
