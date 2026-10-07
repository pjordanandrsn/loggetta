"""Native adapter export/reload with actual runtime LoRA modules on CPU, no downloaded models."""
import copy
import hashlib
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from experts4bit_qlora.lora import LoRALinear, trainable_lora_param_ids
from loggetta.backends.experts4bit_adapters import load_adapter, load_adapter_weights, read_manifest, save_adapter
from loggetta.data import TrainingData, prepare_data
from loggetta.plan import Workload
from test_execution import a_plan
from test_training_data import Tokenizer, jsonl


class TinyModel(nn.Module):
    def __init__(self, rank=2):
        super().__init__()
        self.embedding = nn.Embedding(16, 8)
        self.projection = LoRALinear(nn.Linear(8, 16, bias=False), r=rank, alpha=4, dtype=torch.float32)
        expert, attention = trainable_lora_param_ids(self)
        for p in self.parameters():
            p.requires_grad_(id(p) in expert | attention)

    def forward(self, input_ids, labels=None):
        logits = self.projection(self.embedding(input_ids))
        loss = None if labels is None else F.cross_entropy(logits[:, :-1].reshape(-1, 16), labels[:, 1:].reshape(-1))
        return SimpleNamespace(logits=logits, loss=loss)


def frozen_digest(model):
    h = hashlib.sha256()
    for p in model.parameters():
        if not p.requires_grad:
            h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return {"frozen_cpu_fixture": h.hexdigest()}


def test_train_real_lora_export_reload_preserves_logits_and_frozen_base(tmp_path, monkeypatch):
    from loggetta.backends import experts4bit_train

    torch.manual_seed(12)
    model = TinyModel()
    fresh = copy.deepcopy(model)
    before = frozen_digest(model)
    file = jsonl(tmp_path, [{"text": "abcdefghijklmno"}])
    spec = TrainingData(str(file))
    w = Workload(seq_len=4, steps=2, data=spec.to_dict())
    plan = replace(a_plan(), workload=w)
    sampler = SimpleNamespace(peak=0, interval=0, samples=0, anon_peak=0, shmem_peak=0, file_peak=0, required_peak=0)
    monkeypatch.setattr(experts4bit_train, "_expert_digest", frozen_digest)
    with prepare_data(Tokenizer(), 2, 4, spec) as prepared:
        out = experts4bit_train.train_loop(model, [p for p in model.parameters() if p.requires_grad],
                                           "no-network", w, sampler, {}, prepared_data=prepared, device="cpu")
    assert out["status"] == "OK" and frozen_digest(model) == before
    path = tmp_path / "adapter"
    result = save_adapter(model, Tokenizer(), path, plan, data=out["data"], report={"commit": "abc123"}, seed=0)
    manifest = read_manifest(path)
    assert result["tensor_count"] == 2
    assert set(manifest["tensors"]) == {"projection.lora_A", "projection.lora_B"}
    assert manifest["resolved_model_revision"] == "abc123"
    assert not manifest["optimizer_state_included"]
    load_adapter_weights(fresh, path)
    assert frozen_digest(fresh) == before
    ids = torch.tensor([[1, 2, 3, 4]])
    assert torch.equal(model(ids).logits, fresh(ids).logits)
    for trained, restored in zip(model.parameters(), fresh.parameters()):
        assert torch.equal(trained, restored)


def make_artifact(tmp_path):
    model = TinyModel()
    with torch.no_grad():
        model.projection.lora_B.fill_(0.1)
    path = tmp_path / "adapter"
    save_adapter(model, Tokenizer(), path, a_plan(), data={}, report={"commit": "pinned-commit"}, seed=3)
    return model, path


def test_loader_rebuilds_recorded_recipe_and_revision(tmp_path, monkeypatch):
    from experts4bit_qlora import recipe
    model, path = make_artifact(tmp_path)
    fresh = copy.deepcopy(model)
    with torch.no_grad():
        fresh.projection.lora_B.zero_()
    seen = {}
    def prepare(model_id, setup, *, revision, device):
        seen.update(model=model_id, setup=setup.to_dict(), revision=revision, device=device)
        return SimpleNamespace(model=fresh)
    monkeypatch.setattr(recipe, "prepare_qlora_training", prepare)
    restored = load_adapter(path, device="cpu")
    assert seen["model"] == "org/Tiny-MoE" and seen["revision"] == "pinned-commit"
    assert seen["device"] == "cpu"
    assert not restored.training
    assert torch.equal(restored.projection.lora_B, model.projection.lora_B)


def test_no_overwrite_of_existing_directory(tmp_path):
    model, path = make_artifact(tmp_path)
    original = (path / "adapter.safetensors").read_bytes()
    with pytest.raises(FileExistsError, match="already exists"):
        save_adapter(model, Tokenizer(), path, a_plan(), data={}, report={}, seed=0)
    assert (path / "adapter.safetensors").read_bytes() == original


def test_corruption_rejected_before_parameters_are_modified(tmp_path):
    _, path = make_artifact(tmp_path)
    fresh = TinyModel()
    before = {k: p.detach().clone() for k, p in fresh.named_parameters()}
    weights = path / "adapter.safetensors"
    weights.write_bytes(weights.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_adapter_weights(fresh, path)
    assert all(torch.equal(p, before[k]) for k, p in fresh.named_parameters())


def test_shape_mismatch_rejected_before_any_copy(tmp_path):
    _, path = make_artifact(tmp_path)
    wrong_rank = TinyModel(rank=3)
    before = {k: p.detach().clone() for k, p in wrong_rank.named_parameters()}
    with pytest.raises(ValueError, match="shape/dtype mismatch"):
        load_adapter_weights(wrong_rank, path)
    assert all(torch.equal(p, before[k]) for k, p in wrong_rank.named_parameters())


def test_unknown_trainables_and_nonfinite_adapters_are_not_exported(tmp_path):
    model = TinyModel()
    model.embedding.weight.requires_grad_(True)
    with pytest.raises(ValueError, match="non-adapter trainables"):
        save_adapter(model, Tokenizer(), tmp_path / "bad", a_plan(), data={}, report={}, seed=0)
    model.embedding.weight.requires_grad_(False)
    with torch.no_grad():
        model.projection.lora_B.fill_(float("nan"))
    with pytest.raises(ValueError, match="non-finite"):
        save_adapter(model, Tokenizer(), tmp_path / "bad", a_plan(), data={}, report={}, seed=0)
    assert not (tmp_path / "bad").exists()


def test_invalid_custom_data_fails_before_gpu_or_model_load(tmp_path, monkeypatch):
    import transformers
    from experts4bit_qlora import recipe
    from loggetta.backends.experts4bit_train import run
    file = jsonl(tmp_path, [{"text": ""}])
    p = replace(a_plan(), workload=Workload(data=TrainingData(str(file)).to_dict()))
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: Tokenizer())
    def no_load(*a, **k):
        raise AssertionError("invalid data reached model or GPU initialization")
    monkeypatch.setattr(recipe, "prepare_qlora_training", no_load)
    monkeypatch.setattr(torch.cuda, "set_device", no_load)
    with pytest.raises(ValueError, match="non-empty text"):
        run(p, adapter_dir=str(tmp_path / "adapter"))
    assert not (tmp_path / "adapter").exists()


def test_export_covers_fused_expert_and_attention_adapters(tmp_path):
    from experts4bit_qlora import ExpertsLoRA, ExpertsNbit
    model = TinyModel()
    base = ExpertsNbit.from_float(torch.randn(2, 32, 8), torch.randn(2, 8, 16), quant_type="bf16")
    model.experts = ExpertsLoRA(base, r=2, alpha=4, dtype=torch.float32)
    expert, attention = trainable_lora_param_ids(model)
    for p in model.parameters():
        p.requires_grad_(id(p) in expert | attention)
    fresh = copy.deepcopy(model)
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.add_(0.25)
    before = frozen_digest(model)
    path = tmp_path / "mixed-adapter"
    result = save_adapter(model, Tokenizer(), path, a_plan(), data={}, report={}, seed=0)
    assert result["tensor_count"] == 6  # Four fused-expert tensors and two attention tensors.
    load_adapter_weights(fresh, path)
    assert frozen_digest(fresh) == before
    assert all(torch.equal(a, b) for a, b in zip(model.parameters(), fresh.parameters()))


def test_failed_export_does_not_leave_a_complete_looking_artifact(tmp_path):
    class BrokenTokenizer(Tokenizer):
        def save_pretrained(self, path):
            raise OSError("simulated storage failure")
    target = tmp_path / "failed-export"
    with pytest.raises(OSError, match="storage failure"):
        save_adapter(TinyModel(), BrokenTokenizer(), target, a_plan(), data={}, report={}, seed=0)
    assert not target.exists()


def test_accumulation_weights_micro_batches_by_their_trained_tokens(tmp_path, monkeypatch):
    """One optimizer step over the same rows must not depend on how they are split into micro-batches. With a loss mask
    the rows carry different numbers of trained tokens, so the step is the mean over all trained tokens: 1 x 2
    (micro-batch x grad-accum) and 2 x 1 must move the adapters identically. SGD keeps the update linear in the
    gradient, so the comparison is exact up to float rounding."""
    from loggetta.backends import experts4bit_train
    from test_training_data import chat, fast_tokenizer

    monkeypatch.setattr(experts4bit_train, "_expert_digest", frozen_digest)
    monkeypatch.setattr(torch.optim, "AdamW", torch.optim.SGD)
    rows = [chat(("user", "hello"), ("assistant", "hi there fine thanks ok")),
            chat(("user", "hello hello hello hello"), ("assistant", "ok"))]
    spec = TrainingData(str(jsonl(tmp_path, rows)), format="chat")
    sampler = SimpleNamespace(peak=0, interval=0, samples=0, anon_peak=0, shmem_peak=0, file_peak=0, required_peak=0)
    torch.manual_seed(3)
    base = TinyModel()
    with torch.no_grad():
        base.projection.lora_B.normal_()                                 # a non-zero B, so lora_A has a gradient too
    moved = []
    for micro_batch, accum in ((2, 1), (1, 2)):
        model = copy.deepcopy(base)
        w = Workload(seq_len=8, micro_batch=micro_batch, grad_accum=accum, steps=1, data=spec.to_dict(),
                     learning_rate=1.0)
        with prepare_data(fast_tokenizer(), 2, 8, spec) as prepared:
            counts = [int(prepared.mask[r, 1:].sum()) for r in range(2)]
            out = experts4bit_train.train_loop(model, [p for p in model.parameters() if p.requires_grad],
                                               "no-network", w, sampler, {}, prepared_data=prepared, device="cpu")
        c = out["correctness"]                  # (status ALARM is possible: B starts non-zero, so its norm may fall)
        assert c["all_finite"] and c["frozen_expert_bytes_unchanged"] and c["steps_without_trained_tokens"] == 0
        moved.append([p.detach().clone() for p in model.parameters() if p.requires_grad])
    assert counts[0] != counts[1]                                        # unequal rows: the case the weighting is for
    for a, b in zip(*moved):
        assert torch.allclose(a, b, atol=1e-5, rtol=0)
