"""The CLI saves data choices in plans and routes execution outputs without loading a model in these tests."""
import json
from dataclasses import replace

import loggetta
from loggetta.cli import main
from loggetta.execution import execute
from loggetta.backends import experts4bit
from loggetta.plan import ExecutionPlan
from test_execution import a_plan, hw
from test_training_data import Tokenizer


def stub_planner(monkeypatch):
    seen = {}
    def describe(*a, **k):
        seen["described"] = True
        return object()
    monkeypatch.setattr(loggetta, "describe_model", describe)
    monkeypatch.setattr(loggetta, "probe", hw)
    monkeypatch.setattr(loggetta.data, "load_tokenizer", lambda *a, **k: Tokenizer())
    def plan(topology, hardware, workload, constraints, *, observations, data_profile=None):
        seen.update(workload=workload, data_profile=data_profile)
        return replace(a_plan(), workload=workload, constraints=constraints, data_profile=data_profile)
    monkeypatch.setattr(loggetta, "plan", plan)
    return seen


def test_cli_saves_dataset_and_hyperparameters(monkeypatch, tmp_path):
    seen = stub_planner(monkeypatch)
    data = tmp_path / "train.jsonl"
    data.write_text('{"body": "some training data"}\n')
    output = tmp_path / "plan.json"
    assert main(["plan", "org/Model", "--dataset", str(data), "--format", "text", "--text-field", "body",
                 "--shuffle-data", "--repeat-data", "--learning-rate", "0.0001", "--out", str(output)]) == 0
    saved = ExecutionPlan.from_dict(json.loads(output.read_text()))
    assert saved.workload.data == seen["workload"].data
    assert saved.workload.data["source"] == str(data)
    assert saved.workload.data["repeat"] and saved.workload.data["shuffle"]
    assert saved.workload.data["text_field"] == "body"
    assert saved.workload.learning_rate == 0.0001


def test_cli_dataset_options_require_source(monkeypatch, capsys):
    stub_planner(monkeypatch)
    assert main(["plan", "org/Model", "--format", "chat"]) == 2
    assert "require --dataset" in capsys.readouterr().err


def test_cli_rejects_dataset_on_serving_plan(monkeypatch, capsys):
    stub_planner(monkeypatch)
    assert main(["plan", "org/Model", "--workload", "serve", "--dataset", "org/corpus"]) == 2
    assert "serving plan" in capsys.readouterr().err


def test_receipt_records_save_failure_instead_of_false_success(monkeypatch, tmp_path):
    seen = {}
    def run(plan, *, seed, log, adapter_dir):
        seen["adapter_dir"] = adapter_dir
        return {"status": "SAVE_FAILED", "measured": {}, "correctness": {}, "engaged": {}, "data": {},
                "artifacts": {}, "artifact_error": "disk full"}
    monkeypatch.setattr(experts4bit, "executor", lambda kind: run)
    adapter = tmp_path / "chosen-adapter"
    receipt = execute(a_plan(), hardware=hw(), out_dir=str(tmp_path / "runs"), adapter_dir=str(adapter),
                      prov={"sources": {}})
    assert seen["adapter_dir"] == str(adapter)
    assert receipt["status"] == "SAVE_FAILED" and receipt["artifact_error"] == "disk full"
    saved = list((tmp_path / "runs").glob("*.json"))
    assert len(saved) == 1
    assert json.loads(saved[0].read_text())["status"] == "SAVE_FAILED"


def test_cli_execute_forwards_output_without_replanning(monkeypatch, tmp_path):
    from loggetta import execution
    saved = tmp_path / "plan.json"
    saved.write_text(a_plan().to_json())
    seen = {}
    def run(p, **kwargs):
        seen.update(kwargs)
        return {"status": "OK", "run_id": "test-run"}
    monkeypatch.setattr(execution, "execute", run)
    monkeypatch.setattr(execution, "summarize", lambda r: "saved")
    assert main(["execute", str(saved), "--out", str(tmp_path), "--adapter-out", str(tmp_path / "adapter"),
                 "--seed", "17"]) == 0
    assert seen["adapter_dir"] == str(tmp_path / "adapter") and seen["seed"] == 17


def test_cli_profiles_the_dataset_before_planning_and_saves_the_profile(monkeypatch, tmp_path):
    seen = stub_planner(monkeypatch)
    data = tmp_path / "train.jsonl"
    data.write_text("".join(json.dumps({"text": "x" * n}) + "\n" for n in (10, 20, 30)))
    output = tmp_path / "plan.json"
    assert main(["plan", "org/Model", "--dataset", str(data), "--epochs", "2", "--out", str(output)]) == 0
    assert seen["data_profile"]["rows"] == 3 and seen["data_profile"]["tokens"] == 63
    assert seen["workload"].epochs == 2
    saved = ExecutionPlan.from_dict(json.loads(output.read_text()))
    assert saved.data_profile == seen["data_profile"]


def test_cli_a_bad_row_fails_before_the_model_is_described(monkeypatch, tmp_path, capsys):
    seen = stub_planner(monkeypatch)
    data = tmp_path / "train.jsonl"
    data.write_text(json.dumps({"text": "fine"}) + "\n" + json.dumps({"text": ""}) + "\n")
    assert main(["plan", "org/Model", "--dataset", str(data)]) == 2
    assert "dataset row 1" in capsys.readouterr().err and "described" not in seen


def test_cli_takes_steps_or_epochs_not_both(monkeypatch, tmp_path, capsys):
    stub_planner(monkeypatch)
    data = tmp_path / "train.jsonl"
    data.write_text(json.dumps({"text": "fine"}) + "\n")
    assert main(["plan", "org/Model", "--dataset", str(data), "--steps", "5", "--epochs", "1"]) == 2
    assert "not both" in capsys.readouterr().err


def test_cli_trains_with_warmup_and_cosine_by_default(monkeypatch, tmp_path):
    seen = stub_planner(monkeypatch)
    assert main(["plan", "org/Model"]) == 0
    assert seen["workload"].lr_schedule == "cosine" and seen["workload"].warmup_steps is None
    assert main(["plan", "org/Model", "--lr-schedule", "constant", "--warmup-steps", "5"]) == 0
    assert seen["workload"].lr_schedule == "constant" and seen["workload"].warmup_steps == 5
    assert main(["plan", "org/Model", "--workload", "serve"]) == 0
    assert seen["workload"].lr_schedule == "constant"                   # a server has no learning rate
