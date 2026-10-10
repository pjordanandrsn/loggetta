"""Cached generation after public adapter reload; actual CPU PEFT and streamed bases."""
import copy

import pytest
import torch
from transformers import GenerationConfig

from loggetta import load_adapter
from loggetta.backends import dense_train
from loggetta.backends.dense_adapters import read_manifest, save_adapter
from test_dense_accumulated_parity import _assert_equal
from test_dense_adapter_roundtrip import CPU_GUARD_PRESENT, _trained_artifact, cpu_threads  # noqa: F401
from test_dense_execution import make_plan
from test_training_data import Tokenizer


PROMPT = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])


def _pair(tmp_path, monkeypatch, family, adapter_dtype):
    prepared, target = _trained_artifact(tmp_path, monkeypatch, family, "device", adapter_dtype)
    manifest = read_manifest(target)
    streamed_target = tmp_path / "streamed-adapter"
    plan = make_plan(tmp_path / "base", placement="stream", adapter_dtype=adapter_dtype)
    save_adapter(prepared.model, Tokenizer(), streamed_target, plan, data=manifest["data"],
                 report=prepared.report, seed=0)
    resident = load_adapter(target, device="cpu")
    streamed = load_adapter(streamed_target, device="cpu")
    handles = [getattr(module, "_dense_offload", None) for module in streamed.modules()]
    handles = [handle for handle in handles if handle is not None]
    assert len(handles) == 2 and sum(handle.bytes for handle in handles) > 0
    assert resident.config.use_cache and streamed.config.use_cache
    return resident, streamed


def _cache_record(cache):
    return [{"keys": layer.keys.detach().clone(), "values": layer.values.detach().clone()} for layer in cache.layers]


def _decode(model, *, corrupt_cache=False):
    records = []
    with torch.no_grad():
        output = model(PROMPT, attention_mask=torch.ones_like(PROMPT), use_cache=True)
        cache = output.past_key_values
        assert cache.get_seq_length() == 4 and len(cache.layers) == 2
        records.append({"logits": output.logits.clone(), "cache": _cache_record(cache)})
        if corrupt_cache:
            # Same cached prefix length, wrong contents: checking just the length would miss this.
            cache.layers[0].values.zero_()
        tokens = [PROMPT.clone(), output.logits[:, -1].argmax(-1, keepdim=True)]
        for step in range(4):
            output = model(tokens[-1], attention_mask=torch.ones(2, 5 + step, dtype=torch.long),
                           past_key_values=cache, use_cache=True)
            assert output.past_key_values is cache and cache.get_seq_length() == 5 + step
            records.append({"logits": output.logits.clone(), "cache": _cache_record(cache)})
            tokens.append(output.logits[:, -1].argmax(-1, keepdim=True))
    return {"steps": records, "tokens": torch.cat(tokens, dim=1)}


@pytest.mark.parametrize("family", ("llama", "qwen3"))
@pytest.mark.parametrize("adapter_dtype", ("fp32", "bf16"))
@pytest.mark.xfail(not CPU_GUARD_PRESENT, strict=True, raises=(RuntimeError, ValueError),
                   reason="experts4bit-qlora#1539, fixed after 0.52.0")
def test_reloaded_dense_streaming_preserves_cached_decode_and_greedy_generate(
        tmp_path, monkeypatch, family, adapter_dtype):
    # Remove this feature-conditional marker when Loggetta's e4b floor reaches the fixing release.
    resident, streamed = _pair(tmp_path, monkeypatch, family, adapter_dtype)
    before = [dense_train.frozen_digest(model) for model in (resident, streamed)]
    expected = _decode(resident)
    _assert_equal(_decode(streamed), expected)
    config = GenerationConfig.from_model_config(resident.config)
    # None inherits Llama's model EOS; an empty list keeps this fixed-length control from ending early.
    config.do_sample, config.max_new_tokens, config.eos_token_id, config.pad_token_id = False, 5, [], 0
    with torch.no_grad():
        r = resident.generate(PROMPT, attention_mask=torch.ones_like(PROMPT), generation_config=copy.deepcopy(config))
        s = streamed.generate(PROMPT, attention_mask=torch.ones_like(PROMPT), generation_config=copy.deepcopy(config))
    assert torch.equal(r, s) and torch.equal(r, expected["tokens"])
    assert [dense_train.frozen_digest(model) for model in (resident, streamed)] == before


@pytest.mark.parametrize("family", ("llama", "qwen3"))
def test_a_corrupted_cached_prefix_fails_the_same_step_equality_bar(tmp_path, monkeypatch, family):
    resident, _ = _pair(tmp_path, monkeypatch, family, "fp32")
    expected = _decode(resident)
    mutant = _decode(resident, corrupt_cache=True)
    assert not torch.equal(expected["steps"][1]["logits"], mutant["steps"][1]["logits"])
    with pytest.raises(AssertionError, match="parity"):
        _assert_equal(mutant, expected)
