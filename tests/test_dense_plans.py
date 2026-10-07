"""Dense adapter-training plans: candidates, the itemized estimate against experts4bit-qlora's measured DQ4 peaks,
choices across card sizes, refusals, and plans that say they are planned only. Configs are built in code; nothing is
downloaded and no weight is read."""
import pytest

tr = pytest.importorskip("transformers")

from loggetta import Constraints, Workload, plan  # noqa: E402
from loggetta.backends import dense  # noqa: E402
from loggetta.hardware import GPU, Fact, HardwareProfile, Host  # noqa: E402

GiB = 1 << 30
QWEN3_32B = dict(hidden_size=5120, intermediate_size=25600, num_hidden_layers=64, num_attention_heads=64,
                 num_key_value_heads=8, head_dim=128, vocab_size=151936, max_position_embeddings=40960,
                 tie_word_embeddings=False)
QWEN3_14B = dict(QWEN3_32B, intermediate_size=17408, num_hidden_layers=40, num_attention_heads=40)
SMALL = dict(hidden_size=64, intermediate_size=160, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
             vocab_size=128, max_position_embeddings=256)
needs_offload = pytest.mark.skipif(dense._offload_plan() is None,
                                   reason="this experts4bit-qlora has no offload_plan (experts4bit-qlora#1312)")


def hw(gib, ram=100):
    r = lambda v: Fact(v, "reported")  # noqa: E731
    g = GPU(index=0, vendor="nvidia", name="Test GPU", uuid=None, compute_capability=r((8, 9)),
            memory_total=r(int(gib * GiB)), memory_free=r(int(gib * GiB)), driver=r("595.84"), pcie_gen_max=r(4),
            pcie_width_max=r(16), pcie_gen_current=r(4), pcie_width_current=r(16))
    host = Host(cpu_model=r("cpu"), cpus=r(16), memory_total=r(128 * GiB), memory_available=r(int(ram * GiB)),
                memory_limit=r(128 * GiB))
    return HardwareProfile(gpus=(g,), host=host, platform="Linux x86_64")


@pytest.fixture(scope="module")
def q14():
    return dense.describe(tr.Qwen3Config(**QWEN3_14B))


@pytest.fixture(scope="module")
def q32():
    return dense.describe(tr.Qwen3Config(**QWEN3_32B))


def allocator(c):
    return sum(ln.bytes for ln in c.lines if ln.where == "device"
               and not ln.name.startswith(("CUDA context", "allocator reserve")))


@pytest.mark.parametrize("placement,intercept", [("device", 19.45), pytest.param("stream", 5.37, marks=needs_offload)])
@pytest.mark.parametrize("seq", [2048, 4096, 7168, 14336])
def test_the_estimate_brackets_dq4s_measured_peaks_at_qwen3_32b(q32, placement, intercept, seq):
    """experts4bit-qlora DQ4 (dq4-5090-2), Qwen3-32B in NF4, PEFT LoRA r16 fp32 on all seven projections, chunked loss
    512, SDPA, micro-batch 1: the allocator peak was intercept + 1.2557 GiB per 1,024 tokens, resident and streamed.
    The estimate must not fall below it, and stays within 4% (resident) and 10% (streamed): its gradients and
    dequantization transient are summed as if they met the activation peak, which DQ4's 0.69 GiB shows they do not."""
    if q32.chunked_loss_refusal:            # DQ4 ran with the chunked loss; without it the bracket does not apply
        pytest.skip(f"this experts4bit-qlora cannot chunk Qwen3's loss: {q32.chunked_loss_refusal}")
    p = plan(q32, hw(200), Workload(seq_len=seq),
             Constraints(fixed={"base": "nf4", "placement": placement, "loss_chunk": 512}), backends=(dense,))
    measured = intercept + 1.2557 * seq / 1024
    est = allocator(p.selected) / GiB
    assert measured <= est <= measured * (1.04 if placement == "device" else 1.10)


def test_a_14b_on_a_24_gb_card_trains_nf4_resident_and_says_why(q14):
    p = plan(q14, hw(23.52), Workload(seq_len=4096), backends=(dense,))
    assert p.status == "feasible" and p.selected.setup["base"] == "nf4" and p.selected.setup["placement"] == "device"
    assert any(r.startswith("NF4 base (QLoRA): a bf16 base needs") for r in p.reasons)
    bf16 = next(c for c in p.alternatives if c.setup["base"] == "bf16")
    assert not bf16.feasible and bf16.rejected[0].startswith("device ")
    assert any(ln.name == "LoRA adapters" and "64,225,280 parameters" in ln.detail for ln in p.selected.lines)


def test_a_48_gb_card_keeps_the_base_exact(q14):
    p = plan(q14, hw(48), Workload(seq_len=4096), backends=(dense,))
    assert p.selected.setup["base"] == "bf16" and "bf16 base: the frozen weights stay exact (LoRA, not QLoRA)" in p.reasons


@needs_offload
def test_a_small_card_streams_the_frozen_layers_and_prices_the_host_and_the_link(q14):
    dense._late_bound_reason.cache_clear()
    p = plan(q14, hw(12), Workload(seq_len=4096), backends=(dense,))
    if dense._late_bound_reason() is not None:
        pytest.skip("the installed bitsandbytes cannot free offloaded weights here")
    s = p.selected
    assert s.setup["placement"] == "stream"
    lines = {ln.name: ln for ln in s.lines}
    assert lines["frozen decoder linears, pinned host homes"].where == "host"
    assert s.bounds["link_bytes_per_step"] == lines["frozen layers streamed per micro-batch"].bytes


def test_streaming_is_refused_when_offloaded_weights_would_not_be_freed(q14, monkeypatch):
    monkeypatch.setattr(dense, "_late_bound_reason", lambda: "bitsandbytes 9.9 differs from the mirrored sources")
    monkeypatch.setattr(dense, "_offload_plan", lambda: (lambda *a, **k: {}))
    p = plan(q14, hw(12), Workload(seq_len=4096), Constraints(fixed={"base": "nf4", "placement": "stream"}),
             backends=(dense,))
    assert p.status == "refused" and "streaming: bitsandbytes 9.9 differs" in p.refusal["reasons"][0]


def test_gemma2_softcapping_is_planned_with_eager_attention_and_its_scores_are_priced():
    t = dense.describe(tr.Gemma2Config(**SMALL, head_dim=16))
    p = plan(t, hw(12), Workload(seq_len=128), backends=(dense,))
    assert p.selected.setup["attn_impl"] in ("eager", "flash_attention_2", "flex_attention")
    if p.selected.setup["attn_impl"] == "eager":
        assert any(ln.name == "eager attention scores" for ln in p.selected.lines)
    held = plan(t, hw(12), Workload(seq_len=128), Constraints(fixed={"attn_impl": "sdpa"}), backends=(dense,))
    assert held.status == "refused" and "would change this model's semantics" in held.refusal["reasons"][0]


def test_full_logits_are_priced_where_the_chunked_loss_does_not_reach():
    t = dense.describe(tr.LlamaConfig(**SMALL))
    p = plan(t, hw(12), Workload(seq_len=128), backends=(dense,))
    assert p.selected.setup["loss_chunk"] == 0 and any("over full logits" in r for r in p.reasons)


@pytest.mark.parametrize("fixed,workload,match", [
    ({"base": "int8"}, Workload(seq_len=64), "8-bit bases are not priced yet"),
    ({"targets": "attn_in,router"}, Workload(seq_len=64), "targets must be a non-empty subset"),
    ({}, Workload(seq_len=512), "exceeds the model's 256 positions"),
])
def test_what_cannot_be_planned_is_refused_in_words(fixed, workload, match):
    t = dense.describe(tr.LlamaConfig(**SMALL))
    p = plan(t, hw(12), workload, Constraints(fixed=fixed), backends=(dense,))
    assert p.status == "refused" and any(match in r for r in p.refusal["reasons"])


def test_targets_choose_the_adapted_roles():
    t = dense.describe(tr.LlamaConfig(**SMALL))
    p = plan(t, hw(12), Workload(seq_len=64), Constraints(fixed={"targets": "attn_in,attn_out"}), backends=(dense,))
    lora = next(ln for ln in p.selected.lines if ln.name == "LoRA adapters")
    per_layer = 16 * ((64 + 64) + 2 * (64 + 32) + (64 + 64))                      # q, k, v, o at r=16
    assert f"{2 * per_layer:,} parameters" in lora.detail


def test_dense_plans_are_planned_only():
    from loggetta.execution import PlanNotExecutable, execute
    from test_execution import hw as small_hw

    t = dense.describe(tr.LlamaConfig(**SMALL))
    p = plan(t, small_hw(), Workload(seq_len=64), backends=(dense,))
    assert p.status == "feasible"
    with pytest.raises(PlanNotExecutable, match="planned only in this release"):
        execute(p, hardware=small_hw())


def test_a_moe_backends_receipts_do_not_set_a_dense_plans_reserve():
    pytest.importorskip("experts4bit_qlora.recipe")
    from loggetta import describe_model

    t = describe_model(tr.LlamaConfig(**SMALL))                                   # experts4bit refuses, dense admits
    moe_offload = {"run_id": "moe-offload", "status": "OK", "model": {"model": "x"}, "workload": {"kind": "train"},
                   "setup": {"expert_residency": "host", "expert_kernel": "grouped_nf4"},
                   "hardware": {"gpu": {"name": "Test GPU", "driver": "595.84"}},
                   "measured": {"cuda_context_bytes": 600 << 20, "device_peak_bytes": 10 * GiB,
                                "device_reserved_peak_bytes": int(13.9 * GiB)}}       # 39% slack, no plan recorded
    p = plan(t, hw(12), Workload(seq_len=64), observations=(moe_offload,))
    reserve = next(ln for ln in p.selected.lines if ln.name.startswith("allocator reserve"))
    assert p.selected.backend == "dense" and reserve.basis == "inferred" and "default 20%" in reserve.detail
    context = next(ln for ln in p.selected.lines if ln.name.startswith("CUDA context"))
    assert context.bytes == 600 << 20                                             # the GPU's context is shared
