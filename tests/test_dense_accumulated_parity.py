"""Dense recipe parity: CPU fp32, exact losses, adapter updates and AdamW state after every update.

The equality bar was fixed before execution: zero tolerance because both placements run the same operations in the
same order. Tiny local checkpoints use the real loader, then promote its bf16 tensors to fp32 before preparing LoRA
and streaming. This isolates recipe/offload semantics; it establishes no NF4, CUDA, memory-capacity or timing result.
"""
import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import transformers as tr

from loggetta import Workload
from loggetta.backends import dense_loader, dense_train
from loggetta.backends.experts4bit_train import train_loop
from test_dense_execution import make_plan, stream_tiny_weights
from test_dense_plans import SMALL


@pytest.fixture
def fp32_checkpoint(tmp_path, monkeypatch, request):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(7)
    family = request.param
    if family == "llama":
        model = tr.LlamaForCausalLM(tr.LlamaConfig(**SMALL))
    else:
        model = tr.Qwen3ForCausalLM(tr.Qwen3Config(**SMALL, head_dim=16))
    directory = tmp_path / family
    model.save_pretrained(directory, safe_serialization=True)
    load = dense_loader.load_base

    def load_fp32(*args, **kwargs):
        model, topology, report = load(*args, **kwargs)
        return model.float(), topology, report

    monkeypatch.setattr(dense_loader, "load_base", load_fp32)
    stream_tiny_weights(monkeypatch)
    yield directory
    torch.set_num_threads(previous_threads)


def _assert_equal(left, right, path="parity"):
    if isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right), path
    elif isinstance(left, dict):
        assert left.keys() == right.keys(), path
        for key in left:
            _assert_equal(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right), path
        for index, (a, b) in enumerate(zip(left, right)):
            _assert_equal(a, b, f"{path}[{index}]")
    else:
        assert left == right, path


def _run(directory, placement, monkeypatch, *, stale_home=False):
    torch.manual_seed(13)
    prepared = dense_train.prepare(make_plan(directory, placement=placement, loss_chunk=0), device="cpu")
    assert all(p.dtype == torch.float32 for p in prepared.trainable)
    assert prepared.report["chunked_loss_engaged"] == 0
    assert prepared.report["train_prefetch"] == (placement == "stream")
    handles = [module._dense_offload for module in prepared.model.modules() if hasattr(module, "_dense_offload")]
    if placement == "stream":
        assert len(handles) == SMALL["num_hidden_layers"]
        assert all(home.dtype == torch.float32 for handle in handles for _, _, _, home in handle.slots)
    if stale_home:
        # Mutation of an actual host home, before the stream stages it. No mock of the proof or training loop.
        home = next(home for _, attr, param, home in handles[0].slots
                    if param and attr == "weight" and home.ndim == 2 and home.shape[1] == SMALL["intermediate_size"])
        with torch.no_grad():
            home.fill_(0.125)
    initial = [p.detach().clone() for p in prepared.trainable]
    frozen = dense_train.frozen_digest(prepared.model)
    updates = []
    optimizer = torch.optim.AdamW

    class RecordingAdamW(optimizer):
        def step(self, *args, **kwargs):
            result = super().step(*args, **kwargs)
            updates.append({"adapters": [p.detach().clone() for p in prepared.trainable],
                            "optimizer": copy.deepcopy(self.state_dict())})
            return result

    # Per-step supervised target counts: (1, 0, 5), (0, 0, 0), (2, 6, 1). The first token never contributes.
    counts = (1, 0, 5, 0, 0, 0, 2, 6, 1)
    masks = np.zeros((9, 8), dtype=np.int64)
    for row, count in enumerate(counts):
        masks[row, 1:1 + count] = 1
    data = SimpleNamespace(blocks=np.arange(1, 73).reshape(9, 8) % SMALL["vocab_size"], mask=masks,
                           info={"source": "tiny local masked fixture"})
    sampler = SimpleNamespace(peak=0, interval=0, samples=0, anon_peak=0, shmem_peak=0, file_peak=0, required_peak=0)
    workload = Workload(seq_len=8, steps=3, micro_batch=1, grad_accum=3, optimizer="adamw", learning_rate=1e-3)
    with monkeypatch.context() as patch:
        patch.setattr(torch.optim, "AdamW", RecordingAdamW)
        result = train_loop(prepared.model, prepared.trainable, str(directory), workload, sampler, {}, seed=19,
                            warmup=0, log=lambda *args, **kwargs: None, prepared_data=data, device="cpu",
                            frozen_digest=dense_train.frozen_digest, frozen_kind="dense")
    assert result["status"] == "OK" and result["correctness"]["all_finite"]
    assert result["correctness"]["steps_without_trained_tokens"] == 1
    assert len(updates) == len(result["correctness"]["losses"]) == 2
    assert dense_train.frozen_digest(prepared.model) == frozen
    assert any(not torch.equal(before, after) for before, after in zip(initial, updates[-1]["adapters"]))
    for state in updates[-1]["optimizer"]["state"].values():
        assert int(state["step"]) == 2
    return {"losses": result["correctness"]["losses"], "updates": updates,
            "frozen": result["correctness"]["frozen_dense_digest"]}


@pytest.mark.parametrize("fp32_checkpoint", ("llama", "qwen3"), indirect=True)
def test_dense_streaming_preserves_every_accumulated_update_and_optimizer_state(fp32_checkpoint, monkeypatch):
    resident = _run(fp32_checkpoint, "device", monkeypatch)
    streamed = _run(fp32_checkpoint, "stream", monkeypatch)
    _assert_equal(resident, streamed)


@pytest.mark.parametrize("fp32_checkpoint", ("llama", "qwen3"), indirect=True)
def test_a_stale_streamed_host_home_fails_the_same_equality_bar(fp32_checkpoint, monkeypatch):
    resident = _run(fp32_checkpoint, "device", monkeypatch)
    stale = _run(fp32_checkpoint, "stream", monkeypatch, stale_home=True)
    with pytest.raises(AssertionError, match=r"parity\.(losses|updates)"):
        _assert_equal(resident, stale)
