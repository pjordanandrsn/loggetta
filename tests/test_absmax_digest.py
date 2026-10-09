"""The frozen-expert digest under experts4bit-qlora's double-quantized absmax.

Since experts4bit-qlora 0.49.0, ``enable_fast_train`` double-quantizes the frozen expert absmax by default for resident
training (``compress_expert_absmax_``), and leaves a guard under the old ``<which>_absmax`` name that raises on any use.
``_expert_digest`` read that name, so resident ``grouped_nf4`` training stopped before its first step. These tests build
the model exactly as a training run does -- ``prepare_qlora_training`` on a tiny random local MoE, which calls
``enable_fast_train`` -- on the CPU, with nothing downloaded.
"""
import hashlib

import pytest

torch = pytest.importorskip("torch")
tr = pytest.importorskip("transformers")
pytest.importorskip("experts4bit_qlora.absmax_dq", reason="needs experts4bit-qlora 0.49.0 or later")
pytest.importorskip("nf4_qlora", reason="needs grouped-nf4-gemm (enable_fast_train patches nothing without it)")

from loggetta.backends.experts4bit_train import _expert_digest  # noqa: E402


@pytest.fixture(scope="module")
def tiny_moe(tmp_path_factory):
    path = tmp_path_factory.mktemp("tiny-moe")
    cfg = tr.Qwen3MoeConfig(hidden_size=64, intermediate_size=128, moe_intermediate_size=128, num_experts=4,
                            num_experts_per_tok=2, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                            head_dim=16, vocab_size=32, max_position_embeddings=256, decoder_sparse_step=1)
    torch.manual_seed(7)
    tr.Qwen3MoeForCausalLM(cfg).save_pretrained(path)
    return str(path)


def _prepare(path, monkeypatch, absmax_dq=None):
    """The resident grouped_nf4 model a training run builds; ``absmax_dq`` None leaves experts4bit-qlora's default."""
    from experts4bit_qlora.engines.fast import FAST_TRAIN_STATS
    from experts4bit_qlora.recipe import QLoRASetup, prepare_qlora_training

    monkeypatch.delenv("OFFLOAD_EXPERTS", raising=False)
    monkeypatch.delenv("TRAIN_ARENA", raising=False)
    if absmax_dq is None:
        monkeypatch.delenv("E4B_ABSMAX_DQ", raising=False)
    else:
        monkeypatch.setenv("E4B_ABSMAX_DQ", absmax_dq)
    prep = prepare_qlora_training(path, QLoRASetup(expert_kernel="grouped_nf4", expert_residency="device"),
                                  device="cpu")
    return prep.model, FAST_TRAIN_STATS["absmax_dq"]


def _stacks(model):
    from experts4bit_qlora import ExpertsNbit

    return [m for m in model.modules() if isinstance(m, ExpertsNbit)]


def test_the_default_double_quantized_absmax_is_digested(tiny_moe, monkeypatch):
    model, dq = _prepare(tiny_moe, monkeypatch)
    assert dq["compressed"] > 0, dq                     # the default compressed it: this is the path that crashed
    first = _expert_digest(model)
    assert set(first) == {"stack[0]", "stack[-1]"} and all(len(v) == 64 for v in first.values())
    assert _expert_digest(model) == first               # the same stored bytes, the same digest


def test_the_digest_covers_every_stored_buffer_of_the_compressed_absmax(tiny_moe, monkeypatch):
    model, _ = _prepare(tiny_moe, monkeypatch)
    base = _stacks(model)[0]
    before = _expert_digest(model)["stack[0]"]
    for name in ("gate_up_absmax_q", "gate_up_absmax_s", "down_absmax_off", "down_absmax_code"):
        buf = base._buffers[name]
        saved = buf.clone()
        with torch.no_grad():
            buf.view(torch.uint8).view(-1)[0] ^= 1      # one stored bit
        assert _expert_digest(model)["stack[0]"] != before, name
        with torch.no_grad():
            buf.copy_(saved)
    assert _expert_digest(model)["stack[0]"] == before


def test_the_fp32_absmax_digest_is_unchanged(tiny_moe, monkeypatch):
    """With the compression off (``E4B_ABSMAX_DQ=0``, the workaround) the digest is the earlier formula exactly, so
    receipts written before the fix compare the same way."""
    model, dq = _prepare(tiny_moe, monkeypatch, absmax_dq="0")
    assert dq["compressed"] == 0
    stacks = _stacks(model)
    want = {}
    for key, base in (("stack[0]", stacks[0]), ("stack[-1]", stacks[-1])):
        sha = hashlib.sha256()
        for n in sorted(("gate_up_proj", "down_proj", "gate_up_absmax", "down_absmax")):
            t = getattr(base, n)
            if t is not None:
                sha.update(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
        want[key] = sha.hexdigest()
    assert _expert_digest(model) == want
