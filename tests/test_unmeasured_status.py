"""A run whose device check had no driver sample says UNMEASURED, not OK, and the receipt says why (seen under WSL2,
where nvidia-smi lists no per-process memory: driver_samples was 0 and the status still read OK)."""
import pytest

from loggetta.backends import experts4bit_train as T
from loggetta.measure import DriverMemorySampler


def sampler(smi="/usr/bin/nvidia-smi", reads=0, errors=0, samples=0):
    s = DriverMemorySampler()
    s._smi, s.reads, s.read_errors, s.samples = smi, reads, errors, samples
    return s


@pytest.mark.parametrize("kw, why", [
    ({"smi": None}, "no nvidia-smi on PATH"),
    ({"reads": 40, "errors": 40}, "all 40 nvidia-smi reads failed"),
    ({"reads": 40}, "nvidia-smi answered 40 time(s) but never listed this process's GPU memory (WSL2"),
])
def test_an_unsampled_cuda_run_names_why(kw, why):
    text = T.driver_unmeasured(sampler(**kw), is_cuda=True)
    assert text.startswith(why) and "the plan's device estimate was not checked against it" in text


def test_a_sampled_run_or_a_cpu_run_is_not_unmeasured():
    assert T.driver_unmeasured(sampler(reads=40, samples=38), is_cuda=True) is None
    assert T.driver_unmeasured(sampler(smi=None), is_cuda=False) is None


@pytest.mark.parametrize("ok, meas, status", [
    (True, {}, "OK"),
    (True, {"driver_unmeasured": "no nvidia-smi on PATH: ..."}, "UNMEASURED"),
    (False, {"driver_unmeasured": "no nvidia-smi on PATH: ..."}, "ALARM"),     # an integrity failure outranks it
    (False, {}, "ALARM"),
])
def test_the_status_never_claims_what_was_not_measured(ok, meas, status):
    assert T.run_status(ok, meas) == status


def test_an_unmeasured_run_still_saves_its_adapter_and_counts_as_evidence():
    from loggetta import planner

    assert "UNMEASURED" in T.TRAINED_STATUSES and "ALARM" not in T.TRAINED_STATUSES
    assert "UNMEASURED" in planner.MEASURED_STATUSES


def test_the_summary_says_the_device_check_did_not_run():
    from loggetta.execution import summarize

    receipt = {"run_id": "r", "status": "UNMEASURED", "comparison": {},
               "measured": {"s_per_step_median": 1.0, "timed_steps": 1, "tokens_per_s": 1.0,
                            "driver_unmeasured": "nvidia-smi answered 40 time(s) but never listed ..."},
               "correctness": {"losses": [1.0, 0.5], "loss_first_third_mean": 1.0, "loss_last_third_mean": 0.5,
                               "all_finite": True, "frozen_expert_bytes_unchanged": True, "adapters_moved": True}}
    text = summarize(receipt)
    assert "Run r: UNMEASURED" in text
    assert "device check NOT RUN (status UNMEASURED): nvidia-smi answered 40 time(s)" in text


def test_the_sampler_counts_reads_and_failures(monkeypatch):
    s = DriverMemorySampler(interval=0.01)
    s._smi = "/usr/bin/nvidia-smi"
    calls = iter([OSError("gone"), None, None])

    def read():
        v = next(calls, None)
        if isinstance(v, Exception):
            raise v
        return v

    monkeypatch.setattr(s, "_read", read)
    with s:
        import time
        time.sleep(0.1)
    assert s.reads >= 3 and s.read_errors == 1 and s.samples == 0
    assert s.unmeasured_reason().startswith(f"nvidia-smi answered {s.reads} time(s)")


def _gpu():
    return type("G", (), {"name": "Test GPU", "driver": type("F", (), {"value": "575.64.05"})()})()


@pytest.mark.parametrize("obs_extra", [
    {"status": "UNMEASURED", "measured": {"driver_samples": 0}},                         # as written today: no field
    {"status": "UNMEASURED", "measured": {"driver_samples": 0, "cuda_context_bytes": 0}},  # a later edit writing 0
    {"measured": {"driver_samples": 0, "cuda_context_bytes": 0}},                        # status missing, samples 0
])
def test_a_run_with_no_driver_sample_never_teaches_the_cuda_context(obs_extra):
    """The planner falls back to its default context line; a receipt with no driver sample cannot teach a 0 GB one."""
    from loggetta import planner

    obs = {"run_id": "wsl", "hardware": {"gpu": {"name": "Test GPU", "driver": "575.64.05"}},
           **obs_extra, "measured": {"device_peak_bytes": 4 << 30, "device_reserved_peak_bytes": 5 << 30,
                                     **obs_extra["measured"]}}
    dev, _, _, _ = planner._overheads(None, _gpu(), [obs])
    assert dev[0].basis == "inferred" and dev[0].bytes == planner.DEFAULT_CUDA_CONTEXT
    assert "wsl" not in dev[0].detail


@pytest.mark.parametrize("status, code", [("OK", 0), ("UNMEASURED", 4), ("ALARM", 1), ("SAVE_FAILED", 1),
                                          ("SAVED_V1", 1)])
def test_unmeasured_has_its_own_documented_exit_code(status, code):
    from loggetta import cli

    assert cli.EXIT_CODES.get(status, 1) == code
    assert "4 trained, but the device check took no driver sample" in cli.EXIT_HELP


def test_the_help_lists_the_exit_codes(capsys):
    from loggetta import cli

    with pytest.raises(SystemExit):
        cli.main(["--help"])
    assert "4 trained, but the device check took no driver sample" in " ".join(capsys.readouterr().out.split())


def test_a_cited_unmeasured_receipt_says_so():
    from loggetta import planner

    assert planner.cite({"run_id": "w", "status": "UNMEASURED"}) == \
        "receipt w (UNMEASURED: no driver sample, its allocator peaks only)"


def test_an_unmeasured_run_that_fell_back_to_v1_still_reports_the_fallback(tmp_path, monkeypatch):
    """A /1 written where /2 was expected is always reported; UNMEASURED stays the status (it matters more)."""
    from test_adapter_v2 import FACTS, FakeSchema, plan_with, stubbed_trainer

    from loggetta.backends import experts4bit_adapters as A

    schema = FakeSchema()
    monkeypatch.setattr(A, "adapter_schema", lambda: schema)

    def train(model, trainable, model_id, w, smi, meas, **k):
        meas["driver_unmeasured"] = "nvidia-smi answered 3 time(s) but never listed this process's GPU memory"
        return {"status": "UNMEASURED", "measured": meas, "correctness": {}, "data": {}}

    T2 = stubbed_trainer(monkeypatch, report=FACTS, train=train)
    monkeypatch.setattr(A, "v2_fields", lambda *a, **k: (_ for _ in ()).throw(A.V2Unavailable("save-time reason")))
    out = T2.run(plan_with(), adapter_dir=str(tmp_path / "a"), run_id="r")
    assert out["status"] == "UNMEASURED"
    assert out["artifact_error"] == "saved as backend-adapter/1, not /2: save-time reason"
    assert out["artifacts"]["adapter"]["format"] == A.SCHEMA_V1
