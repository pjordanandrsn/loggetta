"""Actual dense inference-artifact boundary; CPU bf16 bases, no NF4/capacity claim."""
import json
from types import SimpleNamespace

import pytest
import torch
import transformers as tr
from peft.utils import get_peft_model_state_dict
from safetensors.torch import load_file, save_file

from loggetta import load_adapter
from loggetta.backends import dense_train
from loggetta.backends.dense_adapters import read_manifest, save_adapter
from loggetta.backends.experts4bit_adapters import MANIFEST
from loggetta.backends.experts4bit_train import train_loop
from loggetta.data import file_sha256
from test_dense_execution import make_plan, stream_tiny_weights
from test_dense_plans import SMALL
from test_training_data import Tokenizer


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _trained_artifact(tmp_path, monkeypatch, family, placement, adapter_dtype):
    torch.manual_seed(7)
    cls, config = (tr.LlamaForCausalLM, tr.LlamaConfig(**SMALL)) if family == "llama" else (
        tr.Qwen3ForCausalLM, tr.Qwen3Config(**SMALL, head_dim=16))
    directory = tmp_path / "base"
    cls(config).to(torch.bfloat16).save_pretrained(directory, safe_serialization=True)
    stream_tiny_weights(monkeypatch)
    plan = make_plan(directory, placement=placement, adapter_dtype=adapter_dtype)
    prepared = dense_train.prepare(plan, device="cpu")
    before = dense_train.frozen_digest(prepared.model)
    sampler = SimpleNamespace(peak=0, interval=0, samples=0, anon_peak=0, shmem_peak=0, file_peak=0, required_peak=0)
    data = SimpleNamespace(blocks=[[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]], info={})
    result = train_loop(prepared.model, prepared.trainable, str(directory), plan.workload, sampler, {},
                        prepared_data=data, device="cpu", frozen_digest=dense_train.frozen_digest, frozen_kind="dense")
    assert result["status"] == "OK" and result["correctness"]["adapters_moved"]
    assert dense_train.frozen_digest(prepared.model) == before
    assert any(torch.count_nonzero(p) for name, p in prepared.model.named_parameters() if "lora_B" in name)
    target = tmp_path / "adapter"
    save_adapter(prepared.model, Tokenizer(), target, plan, data=result["data"], report=prepared.report, seed=0)
    prepared.model.eval()
    prepared.model.gradient_checkpointing_disable()
    return prepared, target


@pytest.mark.parametrize("family", ("llama", "qwen3"))
@pytest.mark.parametrize("placement", ("device", "stream"))
@pytest.mark.parametrize("adapter_dtype", ("fp32", "bf16"))
def test_dense_adapter_storage_and_logits_are_bitwise_after_public_reload(
        tmp_path, monkeypatch, family, placement, adapter_dtype):
    prepared, target = _trained_artifact(tmp_path, monkeypatch, family, placement, adapter_dtype)
    manifest = read_manifest(target)
    assert manifest["setup"]["placement"] == placement and manifest["setup"]["adapter_dtype"] == adapter_dtype
    assert manifest["optimizer_state_included"] is False
    saved = load_file(str(target / "adapter_model.safetensors"))
    trained = get_peft_model_state_dict(prepared.model)
    restored = load_adapter(target, device="cpu")
    reloaded = get_peft_model_state_dict(restored)
    assert set(saved) == set(trained) == set(reloaded)
    dtype = torch.float32 if adapter_dtype == "fp32" else torch.bfloat16
    for key, tensor in saved.items():
        assert tensor.dtype == trained[key].dtype == reloaded[key].dtype == dtype
        assert torch.equal(tensor, trained[key]) and torch.equal(tensor, reloaded[key])
    if placement == "stream":
        handles = [getattr(module, "_dense_offload", None) for module in restored.modules()]
        handles = [handle for handle in handles if handle is not None]
        assert len(handles) == manifest["model"]["n_layers"] and sum(handle.bytes for handle in handles) > 0
    assert not restored.training and not any(parameter.requires_grad for parameter in restored.parameters())
    before = dense_train.frozen_digest(restored)
    for ids in (torch.tensor([[1, 2, 3, 4]]), torch.tensor([[8, 6, 4, 2], [1, 3, 5, 7]])):
        with torch.no_grad():
            assert torch.equal(prepared.model(ids, use_cache=False).logits, restored(ids, use_cache=False).logits)
    assert dense_train.frozen_digest(restored) == before


@pytest.mark.parametrize("family", ("llama", "qwen3"))
def test_missing_adapter_tensor_is_rejected_even_with_valid_recomputed_checksum(tmp_path, monkeypatch, family):
    _, target = _trained_artifact(tmp_path, monkeypatch, family, "stream", "fp32")
    weights = target / "adapter_model.safetensors"
    tensors = load_file(str(weights))
    del tensors[next(name for name in tensors if "lora_B" in name)]
    save_file(tensors, str(weights))
    manifest_path = target / MANIFEST
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][weights.name] = {"sha256": file_sha256(weights), "bytes": weights.stat().st_size}
    manifest_path.write_text(json.dumps(manifest) + "\n")
    read_manifest(target)  # The recomputed file checksum passes; semantic reload must refuse.
    with pytest.raises(ValueError, match="PEFT adapter reload differs"):
        load_adapter(target, device="cpu")
