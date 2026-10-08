"""Prospective policy admission, frozen pricing and execution refusal; no GPU measurement."""
from dataclasses import replace
from pathlib import Path

import pytest

from loggetta import Constraints, Workload, plan
from loggetta import dense_policy
from loggetta.backends import dense, dense_train
from loggetta.execution import PlanNotExecutable, compare
from loggetta.plan import ExecutionPlan
from loggetta.planner import OVERHEAD_USES, licensed
from test_dense_plans import hw

FIXTURES = Path(__file__).parent / "fixtures" / "dq10"
SETUP = {"base": "nf4", "placement": "device", "r": 16, "alpha": 32, "adapter_dtype": "fp32",
         "targets": ["attn_in", "attn_out", "mlp_in", "mlp_out"], "attn_impl": "sdpa", "loss_chunk": 0}


@pytest.fixture
def subject(tmp_path):
    (tmp_path / "config.json").write_bytes((FIXTURES / "smollm3_3b.json").read_bytes())
    return dense.describe(str(tmp_path))


@pytest.fixture
def registered_hardware():
    hardware = hw(32)
    gpu = hardware.gpus[0]
    return replace(hardware, gpus=(replace(gpu, name="NVIDIA GeForce RTX 5090",
                                           driver=replace(gpu.driver, value="595.91.07")),))


@pytest.fixture(autouse=True)
def pinned_runtime(monkeypatch):
    monkeypatch.setattr(dense_policy, "runtime_versions", lambda: dense_policy.policy()[0]["runtime"])


def constraints(placement="device", **extra):
    return Constraints(allow_development_executor=True, dense_reserve_policy="dq10", allocator_profile="default",
                       fixed={**SETUP, "placement": placement}, **extra)


@pytest.mark.parametrize("placement", ["device", "stream"])
def test_actual_plan_pricing_and_receipt_allocator_exclusion(subject, registered_hardware, placement):
    workload = Workload(seq_len=4096, steps=2)
    p = plan(subject, registered_hardware, workload, constraints(placement), backends=(dense,))
    assert p.status == "feasible", p.render()
    default = plan(subject, registered_hardware, workload, Constraints(fixed={**SETUP, "placement": placement}),
                   backends=(dense,))
    e = compare(p, {})["device_allocator"]["estimated"]
    assert e == compare(default, {})["device_allocator"]["estimated"]
    payload, digest = dense_policy.policy()
    hypothesis = payload["hypotheses"][placement]
    n, d = hypothesis["reserve_numerator"], hypothesis["reserve_denominator"]
    charge = (e*n+d-1)//d
    tagged = [line for line in p.selected.lines if "DQ10" in line.name]
    assert len(tagged) == 2 and all(line.basis == "inferred" and digest in line.detail for line in tagged)
    assert p.selected.device_bytes == e + charge + hypothesis["context_bytes"]
    assert ExecutionPlan.from_dict(p.to_dict()).to_dict() == p.to_dict()


@pytest.mark.parametrize("change,reason", [
    ({"dense_reserve_policy": "unknown"}, "unknown dense reserve"),
    ({"allocator_profile": None}, "explicit default allocator"),
    ({"allocator_profile": "max_split_size_mb:128"}, "explicit default allocator"),
])
def test_explicit_invalid_selection_refuses_without_default_fallback(subject, registered_hardware, change, reason):
    c = replace(constraints(), **change)
    p = plan(subject, registered_hardware, Workload(seq_len=4096, steps=2), c, backends=(dense,))
    assert p.status == "refused" and reason in p.render()
    assert not any("fits" in text for text in p.refusal["suggestions"])


@pytest.mark.parametrize("change", [{"seq_len": 8192}, {"seq_len": 1024}, {"micro_batch": 2},
                                    {"steps": 3}, {"learning_rate": 1e-4}, {"lr_schedule": "cosine"}])
def test_outside_workload_refuses(subject, registered_hardware, change):
    workload = replace(Workload(seq_len=4096, steps=2), **change)
    p = plan(subject, registered_hardware, workload, constraints(), backends=(dense,))
    assert p.status == "refused" and "DQ10" in p.render()


def test_wrong_card_driver_runtime_config_and_setup_refuse(subject, registered_hardware, monkeypatch):
    gpu = registered_hardware.gpus[0]
    for changed in (replace(gpu, name="NVIDIA GeForce RTX 4090"),
                    replace(gpu, driver=replace(gpu.driver, value="different")),
                    replace(gpu, memory_total=replace(gpu.memory_total, value=24*2**30))):
        with pytest.raises(ValueError, match="GPU/card"):
            dense_policy.selection(subject, changed, SETUP, Workload(seq_len=4096, steps=2), constraints())
    with pytest.raises(ValueError, match="setup"):
        dense_policy.selection(subject, gpu, {**SETUP, "r": 8}, Workload(seq_len=4096, steps=2), constraints())
    config = Path(subject.model) / "config.json"
    config.write_bytes(config.read_bytes() + b" ")
    with pytest.raises(ValueError, match="unchanged registered"):
        dense_policy.selection(subject, gpu, SETUP, Workload(seq_len=4096, steps=2), constraints())
    monkeypatch.setattr(dense_policy, "runtime_versions", lambda: {})
    with pytest.raises(ValueError, match="runtime"):
        dense_policy.selection(subject, gpu, SETUP, Workload(seq_len=4096, steps=2), constraints())


def test_fresh_execution_rechecks_policy_and_one_byte_tampering_before_loading(subject, registered_hardware, monkeypatch):
    p = plan(subject, registered_hardware, Workload(seq_len=4096, steps=2), constraints(), backends=(dense,))
    import loggetta.hardware

    monkeypatch.setattr(loggetta.hardware, "probe", lambda: registered_hardware)
    assert dense_policy.validate_plan(p)["licensed"] is False
    index = next(i for i, line in enumerate(p.selected.lines) if line.name.startswith("allocator reserve"))
    changed = list(p.selected.lines)
    changed[index] = replace(changed[index], bytes=changed[index].bytes-1)
    tampered = replace(p, selected=replace(p.selected, lines=tuple(changed), device_bytes=p.selected.device_bytes-1))
    with pytest.raises(PlanNotExecutable, match="pricing"):
        dense_train.run(tampered)
    stripped = replace(p, constraints=replace(p.constraints, dense_reserve_policy=None))
    with pytest.raises(PlanNotExecutable, match="explicit policy"):
        dense_train.run(stripped)


def test_raw_dense_readings_license_no_generic_observation_use():
    for tag in ("dq7", "dq9", "dq10"):
        for use in OVERHEAD_USES:
            assert not licensed({tag: {}, "licensed_for": list(OVERHEAD_USES)}, use)
    for use in OVERHEAD_USES:
        assert not licensed({"plan": {"constraints": {"dense_reserve_policy": "dq10"}}}, use)


def test_optional_fields_omit_defaults_and_retain_old_wire(subject, registered_hardware):
    p = plan(subject, registered_hardware, Workload(seq_len=4096), Constraints(fixed=SETUP), backends=(dense,))
    assert "dense_reserve_policy" not in p.to_dict()["constraints"]
    assert "allocator_profile" not in p.to_dict()["constraints"]
    assert ExecutionPlan.from_dict(p.to_dict()).to_dict() == p.to_dict()
