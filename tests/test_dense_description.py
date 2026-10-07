"""The dense backend describes decoder-only models from config and a meta-device tree: LoRA targets by structural role,
not by a per-family list of names. Tiny configs of seven families, nothing downloaded, no weights read."""
import pytest

tr = pytest.importorskip("transformers")

from loggetta.backends import dense  # noqa: E402
from loggetta.backends.dense import DenseTopology, classify  # noqa: E402

SMALL = dict(hidden_size=64, intermediate_size=160, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
             vocab_size=128, max_position_embeddings=256)


def roles(t):
    return {r: names for r, names in t.targets.items() if names}


@pytest.mark.parametrize("make,expect", [
    (lambda: tr.Qwen3Config(**SMALL, head_dim=16),
     {"attn_in": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"], "attn_out": ["self_attn.o_proj"],
      "mlp_in": ["mlp.gate_proj", "mlp.up_proj"], "mlp_out": ["mlp.down_proj"]}),
    (lambda: tr.LlamaConfig(**SMALL),
     {"attn_in": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"], "attn_out": ["self_attn.o_proj"],
      "mlp_in": ["mlp.gate_proj", "mlp.up_proj"], "mlp_out": ["mlp.down_proj"]}),
    (lambda: tr.MistralConfig(**SMALL),
     {"attn_in": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"], "attn_out": ["self_attn.o_proj"],
      "mlp_in": ["mlp.gate_proj", "mlp.up_proj"], "mlp_out": ["mlp.down_proj"]}),
    (lambda: tr.Phi3Config(**SMALL, pad_token_id=0),         # fused projections: shapes still classify them
     {"attn_in": ["self_attn.qkv_proj"], "attn_out": ["self_attn.o_proj"], "mlp_in": ["mlp.gate_up_proj"],
      "mlp_out": ["mlp.down_proj"]}),
])
def test_every_projection_classifies_by_shape(make, expect):
    t = dense.describe(make())
    assert t.refusal is None and dense.refusal(t) is None
    assert roles(t) == expect and t.unclassified == () and t.uniform and t.n_layers == 2


def test_parameter_accounting_is_exact_on_llama():
    t = dense.describe(tr.LlamaConfig(**SMALL))                   # head_dim 16, kv width 32
    per_layer = 64 * 64 + 2 * 64 * 32 + 64 * 64 + 3 * 64 * 160
    assert t.linear_params == 2 * per_layer
    assert t.embedding_params == 128 * 64 and t.head_params == 128 * 64 and not t.tied_embeddings
    assert t.other_params == 2 * 2 * 64 + 64                      # two norms per layer + the final norm


def test_qwen2_bias_and_tied_head_are_described():
    t = dense.describe(tr.Qwen2Config(**SMALL, tie_word_embeddings=True))
    q = next(lin for lin in t.layer_linears if lin.name == "self_attn.q_proj")
    assert q.bias and q.role == "attn_in" and t.tied_embeddings and t.head_params == 0


def test_gemma2_softcapping_rules_out_sdpa():
    t = dense.describe(tr.Gemma2Config(**SMALL, head_dim=16))
    assert "sdpa" in t.attention_impls and "sdpa" not in t.attention_valid_impls
    assert any("softcapping" in n and "SDPA" in n for n in t.attention_notes)
    assert t.unclassified == ()


def test_gemma3_multimodal_checkpoint_is_described_by_its_text_tower():
    cfg = tr.Gemma3Config(text_config=dict(**SMALL, head_dim=16),
                          vision_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                                             num_attention_heads=2, image_size=28, patch_size=14))
    t = dense.describe(cfg)
    assert t.refusal is None and t.text_tower_of == "gemma3" and t.model_type == "gemma3_text"
    assert roles(t)["attn_out"] == ["self_attn.o_proj"] and t.unclassified == ()


def test_square_projections_are_told_apart_by_name_and_unknown_ones_stay_unclassified():
    widths = dict(hidden=64, inter=160, q_width=64, kv_width=32)
    assert classify("self_attn.q_proj", 64, 64, **widths) == "attn_in"     # 64 -> 64: also attn_out by shape
    assert classify("self_attn.o_proj", 64, 64, **widths) == "attn_out"
    assert classify("mixer.mystery", 64, 64, **widths) is None             # a tie no name settles
    assert classify("mlp.down_proj", 160, 64, **widths) == "mlp_out"       # shape alone decides


@pytest.mark.parametrize("make,match", [
    (lambda: tr.Qwen3MoeConfig(**SMALL, num_experts=4, moe_intermediate_size=32), "routed experts"),
    (lambda: tr.LlamaConfig(**SMALL, quantization_config={"quant_method": "gptq", "bits": 4}), "pre-quantized"),
    (lambda: tr.BertConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=4, intermediate_size=128,
                           vocab_size=128, is_decoder=True), "no decoder layers"),
])
def test_what_the_dense_backend_will_not_describe_is_refused_in_words(make, match):
    t = dense.describe(make())
    assert isinstance(t, DenseTopology) and match in t.refusal and match in dense.refusal(t)


def test_a_dense_model_reaches_the_planner_through_the_dense_backend():
    pytest.importorskip("experts4bit_qlora.recipe")
    from loggetta import Workload, describe_model, plan
    from test_execution import hw

    t = describe_model(tr.LlamaConfig(**SMALL))
    assert isinstance(t, DenseTopology)
    p = plan(t, hw(), Workload(seq_len=64))
    assert p.status == "feasible" and p.model["model_type"] == "llama" and p.selected.backend == "dense"
    moe = describe_model(tr.Qwen3MoeConfig(**SMALL, num_experts=4, moe_intermediate_size=32, head_dim=16))
    assert not isinstance(moe, DenseTopology)                              # a MoE model stays experts4bit's
