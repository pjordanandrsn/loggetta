"""Dense execution on tiny local safetensors checkpoints. Real PEFT and e4b engines; no network or GPU."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import transformers as tr

from loggetta import Constraints, Workload, execute, load_adapter, plan
from loggetta.backends import dense, dense_train
from loggetta.backends.dense_adapters import read_manifest, save_adapter
from loggetta.backends.dense_loader import load_base
from loggetta.data import TrainingData, prepare_data
from test_dense_plans import SMALL, hw
from test_training_data import Tokenizer, jsonl


@pytest.fixture
def checkpoint(tmp_path):
    torch.manual_seed(7)
    model = tr.LlamaForCausalLM(tr.LlamaConfig(**SMALL)).to(torch.bfloat16)
    directory = tmp_path / "base"
    model.save_pretrained(directory, safe_serialization=True)
    return directory, model


def make_plan(directory, **fixed):
    topology = dense.describe(str(directory))
    constraints = Constraints(fixed={"base": "bf16", "placement": "device", "r": 2, "alpha": 4,
                                     "loss_chunk": 0, **fixed})
    result = plan(topology, hw(24), Workload(seq_len=8, steps=2), constraints, backends=(dense,))
    assert result.status == "feasible", result.render()
    return result


def test_bounded_loader_reconstructs_exact_checkpoint_tensors(checkpoint):
    directory, original = checkpoint
    p = make_plan(directory)
    restored, topology, report = load_base(str(directory), p.selected.setup, device="cpu")
    assert topology.n_layers == 2 and not report["quantized_linears"]
    assert all(not parameter.requires_grad for parameter in restored.parameters())
    assert set(restored.state_dict()) == set(original.state_dict())
    for name, tensor in original.state_dict().items():
        assert torch.equal(tensor, restored.state_dict()[name]), name


def test_tied_checkpoint_loads_without_synthesizing_a_head(tmp_path):
    model = tr.Qwen2ForCausalLM(tr.Qwen2Config(**SMALL, tie_word_embeddings=True)).to(torch.bfloat16)
    directory = tmp_path / "tied"
    model.save_pretrained(directory)
    p = make_plan(directory)
    restored, _, _ = load_base(str(directory), p.selected.setup, device="cpu")
    assert restored.get_input_embeddings().weight is restored.get_output_embeddings().weight
    assert torch.equal(restored.get_input_embeddings().weight, model.get_input_embeddings().weight)


def test_missing_checkpoint_tensor_is_refused(checkpoint, tmp_path):
    from safetensors.torch import load_file, save_file

    directory, _ = checkpoint
    p = make_plan(directory)
    tensors = load_file(directory / "model.safetensors")
    tensors.pop("model.layers.0.self_attn.q_proj.weight")
    save_file(tensors, directory / "model.safetensors")
    with pytest.raises(ValueError, match="no unique tensor"):
        load_base(str(directory), p.selected.setup, device="cpu")


def test_packed_checkpoint_is_refused(checkpoint):
    from safetensors.torch import load_file, save_file

    directory, _ = checkpoint
    p = make_plan(directory)
    tensors = load_file(directory / "model.safetensors")
    name = "model.layers.0.self_attn.q_proj.weight"
    tensors[name] = tensors[name].to(torch.uint8)
    save_file(tensors, directory / "model.safetensors")
    with pytest.raises(ValueError, match="packed or scaled checkpoint formats are refused"):
        load_base(str(directory), p.selected.setup, device="cpu")


def test_conflicting_explicit_tied_head_is_refused(tmp_path):
    from safetensors.torch import load_file, save_file

    directory = tmp_path / "conflicting"
    model = tr.Qwen2ForCausalLM(tr.Qwen2Config(**SMALL, tie_word_embeddings=True)).to(torch.bfloat16)
    model.save_pretrained(directory)
    p = make_plan(directory)
    tensors = load_file(directory / "model.safetensors")
    tensors["lm_head.weight"] = tensors["model.embed_tokens.weight"] + 1
    save_file(tensors, directory / "model.safetensors")
    with pytest.raises(ValueError, match="configured tied head"):
        load_base(str(directory), p.selected.setup, device="cpu")


def test_executed_adapter_setup_equals_the_plan(checkpoint):
    directory, _ = checkpoint
    p = make_plan(directory, r=3, alpha=7, adapter_dtype="bf16", targets="attn_in,mlp_out")
    prepared = dense_train.prepare(p, device="cpu")
    assert prepared.report["setup"] == p.selected.setup
    assert len(prepared.report["adapter_targets"]) == 8
    assert all(param.dtype == torch.bfloat16 for param in prepared.trainable)
    from peft.tuners.lora.layer import LoraLayer
    modules = [module for module in prepared.model.modules() if isinstance(module, LoraLayer)]
    assert len(modules) == 8
    assert all(module.r["default"] == 3 and module.lora_alpha["default"] == 7 for module in modules)
    assert prepared.report["chunked_loss_engaged"] == 0


def stream_tiny_weights(monkeypatch):
    from experts4bit_qlora.engines import dense_offload

    original = dense_offload.enable_dense_offload
    monkeypatch.setattr(dense_offload, "enable_dense_offload",
                        lambda *args, **kwargs: original(*args, **kwargs, min_bytes=1))


def test_real_cpu_streamed_and_resident_losses_and_gradients_are_identical(checkpoint, monkeypatch):
    directory, _ = checkpoint
    stream_tiny_weights(monkeypatch)
    torch.manual_seed(13)
    resident = dense_train.prepare(make_plan(directory), device="cpu")
    torch.manual_seed(13)
    streamed = dense_train.prepare(make_plan(directory, placement="stream"), device="cpu")
    assert streamed.report["train_prefetch"] and streamed.report["dense_offload"]
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    for prepared in (resident, streamed):
        prepared.model.train()
    before = dense_train.frozen_digest(streamed.model)
    rloss = resident.model(input_ids=ids, labels=ids, use_cache=False).loss
    sloss = streamed.model(input_ids=ids, labels=ids, use_cache=False).loss
    rloss.backward()
    sloss.backward()
    assert torch.equal(rloss, sloss)
    assert all(torch.equal(r.grad, s.grad) for r, s in zip(resident.trainable, streamed.trainable))
    assert dense_train.frozen_digest(streamed.model) == before


def test_dense_training_receipt_integrity_and_peft_reload(checkpoint, tmp_path):
    from loggetta.backends.experts4bit_train import train_loop

    directory, _ = checkpoint
    p = make_plan(directory)
    prepared = dense_train.prepare(p, device="cpu")
    file = jsonl(tmp_path, [{"text": "abcdefghijklmno"}])
    spec = TrainingData(str(file))
    workload = replace(p.workload, data=spec.to_dict())
    sampler = SimpleNamespace(peak=0, interval=0, samples=0, anon_peak=0, shmem_peak=0, file_peak=0, required_peak=0)
    with prepare_data(Tokenizer(), 2, workload.seq_len, spec) as data:
        result = train_loop(prepared.model, prepared.trainable, str(directory), workload, sampler, {},
                            prepared_data=data, device="cpu", frozen_digest=dense_train.frozen_digest,
                            frozen_kind="dense")
    assert result["status"] == "OK" and result["correctness"]["frozen_dense_bytes_unchanged"]
    assert result["correctness"]["adapters_moved"]
    target = tmp_path / "adapter"
    save_adapter(prepared.model, Tokenizer(), target, p, data=result["data"], report=prepared.report, seed=0)
    manifest = read_manifest(target)
    assert manifest["backend"] == "dense" and manifest["format"] == "PEFT linear LoRA"
    assert not manifest["optimizer_state_included"]
    restored = load_adapter(target, device="cpu")
    prepared.model.eval()
    prepared.model.gradient_checkpointing_disable()
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        assert torch.equal(prepared.model(ids, use_cache=False).logits, restored(ids, use_cache=False).logits)
    weights = target / "adapter_model.safetensors"
    weights.write_bytes(weights.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_adapter(target, device="cpu")


def test_execute_dispatches_dense_and_preserves_setup(checkpoint, monkeypatch):
    from loggetta.execution import PlanNotExecutable

    directory, _ = checkpoint
    p = make_plan(directory)
    seen = {}

    def run(received, **kwargs):
        seen["plan"] = received
        return {"status": "OK", "measured": {}, "correctness": {}, "engaged": {"backend": "dense"}, "data": {}}

    with pytest.raises(PlanNotExecutable, match="--allow-development-executor"):
        execute(p, hardware=hw(24), prov={"sources": {}}, log=lambda *a: None)
    assert not seen
    p = replace(p, constraints=replace(p.constraints, allow_development_executor=True))
    monkeypatch.setattr(dense_train, "run", run)
    receipt = execute(p, hardware=hw(24), prov={"sources": {}}, log=lambda *a: None)
    assert seen["plan"] is p
    assert receipt["setup"] == p.selected.setup
    assert receipt["backend"] == "dense"
    assert receipt["plan"]["selected"]["backend"] == "dense"
    assert receipt["plan"]["constraints"]["allow_development_executor"] is True


def test_cli_saved_plan_development_opt_in_is_typed_and_default_is_omitted(checkpoint, tmp_path, monkeypatch):
    from loggetta.cli import main
    from loggetta import execution
    from loggetta.plan import ExecutionPlan

    directory, _ = checkpoint
    p = make_plan(directory)
    saved = tmp_path / "plan.json"
    saved.write_text(p.to_json())
    assert "allow_development_executor" not in p.to_dict()["constraints"]
    assert not ExecutionPlan.from_dict(p.to_dict()).constraints.allow_development_executor
    seen = []
    def run(received, **kwargs):
        seen.append(received.constraints.allow_development_executor)
        return {"run_id": "fixture", "status": "OK"}
    monkeypatch.setattr(execution, "execute", run)
    monkeypatch.setattr(execution, "summarize", lambda _: "fixture")
    assert main(["execute", str(saved)]) == 0
    assert main(["execute", str(saved), "--allow-development-executor"]) == 0
    assert seen == [False, True]


def test_dense_integrity_detects_a_frozen_weight_mutation(checkpoint):
    directory, _ = checkpoint
    prepared = dense_train.prepare(make_plan(directory), device="cpu")
    before = dense_train.frozen_digest(prepared.model)
    with torch.no_grad():
        parameter = next(p for p in prepared.model.base_model.model.model.layers[0].parameters() if not p.requires_grad)
        parameter.view(-1)[0] += 1
    assert dense_train.frozen_digest(prepared.model) != before


def test_chunked_loss_executes_the_planned_chunk_on_qwen3(tmp_path):
    directory = tmp_path / "qwen3"
    model = tr.Qwen3ForCausalLM(tr.Qwen3Config(**SMALL, head_dim=16)).to(torch.bfloat16)
    model.save_pretrained(directory)
    p = make_plan(directory, loss_chunk=4)
    prepared = dense_train.prepare(p, device="cpu")
    assert prepared.report["setup"] == p.selected.setup and prepared.report["chunked_loss_engaged"] == 1
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    loss = prepared.model(input_ids=ids, labels=ids, use_cache=False).loss
    loss.backward()
    assert torch.isfinite(loss)
    assert all(param.grad is not None and torch.isfinite(param.grad).all() for param in prepared.trainable)


def test_different_projection_shapes_are_not_called_uniform(monkeypatch):
    from accelerate import init_empty_weights

    config = tr.LlamaConfig(**SMALL)
    with init_empty_weights():
        tree = tr.LlamaForCausalLM(config)
        # Both classify as attn_in and have the same name, but their key widths differ.
        tree.model.layers[1].self_attn.k_proj = torch.nn.Linear(64, 64, bias=False)
    monkeypatch.setattr(tr.AutoModelForCausalLM, "from_config", lambda *a, **k: tree)
    topology = dense.describe(config)
    assert not topology.uniform
