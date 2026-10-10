"""experts4bit-qlora's fused q/k/v training projection (P129, on by default since experts4bit-qlora#1560) costs device bytes
the plan must carry: each fused projection's absmax in fp32, 3 B per 64 q/k/v values over the nested statistics. Loggetta
defers to experts4bit-qlora's estimate, so the line is that estimate's; these tests pin that it reaches the plan under the
default, that ``E4B_TRAIN_FUSE_QKV=0`` drops it, and that the plan records the knob."""
import pytest

pytest.importorskip("transformers")
import transformers as tr  # noqa: E402

from loggetta.backends import experts4bit as e4b  # noqa: E402

LINE = "fused q/k/v training projection (fp32 absmax)"


def _fused_term_in_estimate():
    from experts4bit_qlora import recipe

    return hasattr(recipe, "_fused_qkv_absmax_bytes")


pytestmark = pytest.mark.skipif(not _fused_term_in_estimate(),
                                reason="the installed experts4bit-qlora predates the fused q/k/v term (#1560)")


def _qwen3_30b(layers=48):
    """Qwen3-30B-A3B's attention and expert shapes (config.json on the Hub), built offline."""
    return tr.Qwen3MoeConfig(hidden_size=2048, num_hidden_layers=layers, num_attention_heads=32, num_key_value_heads=4,
                             head_dim=128, num_experts=128, num_experts_per_tok=8, moe_intermediate_size=768,
                             intermediate_size=6144, vocab_size=151936, decoder_sparse_step=1, mlp_only_layers=[],
                             attention_bias=False)


def _qkv_values(cfg):
    h, d = cfg.hidden_size, cfg.head_dim
    return cfg.num_hidden_layers * h * (cfg.num_attention_heads + 2 * cfg.num_key_value_heads) * d


def _lines(topo, **setup):
    from loggetta import Workload

    base = dict(quant_type="nf4", blocksize=64, r=16, alpha=16, adapter_dtype="bf16", train_experts=True,
                train_attention=True, attn_4bit=True, expert_residency="device", pin=True, expert_kernel="grouped_nf4",
                dgrad=True, keep_moe_layers=0)
    lines, _, refusals = e4b.estimate(topo, {**base, **setup}, Workload(seq_len=512, micro_batch=1))
    assert not refusals
    return {ln[0]: ln for ln in lines}


@pytest.fixture(scope="module")
def qwen3():
    from loggetta import describe_model

    cfg = _qwen3_30b()
    return cfg, describe_model(cfg)


def test_the_default_plan_carries_the_fused_term(qwen3, monkeypatch):
    cfg, topo = qwen3
    monkeypatch.delenv("E4B_TRAIN_FUSE_QKV", raising=False)
    line = _lines(topo)[LINE]
    assert line[1] == "device" and line[2] == 3 * _qkv_values(cfg) // 64
    assert 23e6 < line[2] < 25e6                                   # ~24 MB on Qwen3-30B-A3B


def test_turning_the_knob_off_drops_it(qwen3, monkeypatch):
    _, topo = qwen3
    monkeypatch.setenv("E4B_TRAIN_FUSE_QKV", "0")
    assert LINE not in _lines(topo)


@pytest.mark.parametrize("setup", [dict(attn_4bit=False), dict(train_attention=False), dict(expert_kernel="reference")])
def test_only_where_the_run_fuses(qwen3, monkeypatch, setup):
    _, topo = qwen3
    monkeypatch.delenv("E4B_TRAIN_FUSE_QKV", raising=False)
    assert LINE not in _lines(topo, **setup)


def test_the_plan_records_the_knob(monkeypatch):
    from experts4bit_qlora import estimate_env

    monkeypatch.delenv("E4B_TRAIN_FUSE_QKV", raising=False)
    assert estimate_env()["E4B_TRAIN_FUSE_QKV"] is True
    monkeypatch.setenv("E4B_TRAIN_FUSE_QKV", "0")
    assert estimate_env()["E4B_TRAIN_FUSE_QKV"] is False
