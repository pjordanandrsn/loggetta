"""Dense adapter-training plans: candidates, the itemized estimate against experts4bit-qlora's measured DQ4 peaks,
choices across card sizes and refusals. Configs are built in code; nothing is
downloaded and no weight is read."""
import hashlib
import json
from pathlib import Path

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


@pytest.mark.parametrize("n", [1024, 1025, 16384, 16385, 32768])
def test_nf4_linear_bytes_include_every_retained_quant_state_tensor(n):
    import torch

    functional = pytest.importorskip("bitsandbytes.functional")
    packed, state = functional.quantize_4bit(torch.zeros(n, dtype=torch.bfloat16), quant_type="nf4",
                                           compress_statistics=True)
    tensors = (state.absmax, state.code, state.offset, state.state2.absmax, state.state2.code)
    retained = sum(tensor.numel() * tensor.element_size() for tensor in tensors)
    assert dense._linear_bytes(n, "nf4") == (packed.numel() * packed.element_size(), retained)
    # The nested codebook has private storage, so it belongs in every projection's charge.
    _, other = functional.quantize_4bit(torch.zeros(n, dtype=torch.bfloat16), quant_type="nf4",
                                       compress_statistics=True)
    assert state.state2.code.data_ptr() != other.state2.code.data_ptr()


@needs_offload
@pytest.mark.parametrize("base", ["nf4", "bf16"])
def test_small_frozen_weights_remain_priced_when_no_weight_reaches_streaming_threshold(base):
    topology = dense.describe(tr.LlamaConfig(**SMALL))
    setup = dict(base=base, placement="device", adapter_dtype="fp32", r=16, alpha=32,
                 targets=("attn_in", "attn_out", "mlp_in", "mlp_out"), loss_chunk=0, attn_impl="sdpa")
    resident, _, refused_r = dense.estimate(topology, setup, Workload(seq_len=64))
    streamed, _, refused_s = dense.estimate(topology, {**setup, "placement": "stream"}, Workload(seq_len=64))
    if refused_r or refused_s:
        pytest.skip(f"the installed runtime cannot price this base: {refused_r or refused_s}")
    def frozen_device(lines):
        return sum(row[2] for row in lines if row[1] == "device"
                   and row[0].startswith("frozen decoder linears"))
    # All packed weights in this tiny architecture are below 1 MiB. Nothing moves, so no frozen bytes disappear.
    assert frozen_device(resident) > 0 and frozen_device(streamed) == frozen_device(resident)
    assert next(row[2] for row in streamed if row[0] == "frozen decoder linears, pinned host homes") == 0
    # The retained frozen-weight charge must not absorb LoRA, which is already priced independently.
    for name in ("LoRA adapters", "adapter gradients", "optimizer state (adamw)"):
        assert next(row[2] for row in resident if row[0] == name) == next(row[2] for row in streamed if row[0] == name)


@needs_offload
def test_smollm3_keeps_36_mib_of_packed_kv_weights_on_device():
    if not hasattr(tr, "SmolLM3Config"):
        pytest.skip("SmolLM3 requires the registered transformers family")
    topology = dense.describe(tr.SmolLM3Config(hidden_size=2048, intermediate_size=11008, num_hidden_layers=36,
                                              num_attention_heads=16, num_key_value_heads=4, head_dim=128,
                                              vocab_size=128256, max_position_embeddings=65536))
    setup = dict(base="nf4", placement="stream", adapter_dtype="fp32", r=16, alpha=32,
                 targets=("attn_in", "attn_out", "mlp_in", "mlp_out"), loss_chunk=0, attn_impl="sdpa")
    lines, _, refused = dense.estimate(topology, setup, Workload(seq_len=512))
    if refused:
        pytest.skip(f"the installed runtime cannot price NF4 streaming: {refused}")
    assert next(row[2] for row in lines if row[0] == "frozen decoder linears, kept on device") == 36 << 20
    assert topology.n_layers == 36


# Snapshotted from the committed pre-fix estimator and the exact DQ9 config files.
# These are config-only estimates, with no model weights or hardware measurements.
DQ9_FIXTURES = Path(__file__).parent / "fixtures" / "dq9"
DQ9_REFERENCE = json.loads((DQ9_FIXTURES / "prior-estimates.json").read_text())
# Rows whose activation item now takes the larger branch as returned (see the file's "why"); the snapshot stays as taken.
DQ9_ACTIVATION_CORRECTION = json.loads((DQ9_FIXTURES / "activation-max-correction.json").read_text())["rows"]
#: the chunk coefficient the snapshot (and the correction above) were computed at
DQ9_BASELINE_CHUNK_BYTES_PER_LOGIT = 10


@needs_offload
@pytest.mark.parametrize("row", DQ9_REFERENCE["rows"],
                         ids=lambda row: f"{row['subject']}-{row['placement']}-{row['seq']}")
def test_kept_weight_correction_is_zero_on_every_dq9_subject_placement_and_rung(row, monkeypatch):
    # The snapshot was taken with experts4bit-qlora's chunk coefficient at 10 B per logit, its value before #1505. Pin it
    # (on the helper and on loggetta's fallback) so this stays a regression of the kept-weight correction at that
    # baseline; the coefficient's own movement is covered by the activation tests below.
    import experts4bit_qlora.engines.chunked_lm_loss as cll

    monkeypatch.setattr(cll, "CHUNK_BYTES_PER_LOGIT", DQ9_BASELINE_CHUNK_BYTES_PER_LOGIT)
    monkeypatch.setattr(dense, "CHUNK_LOSS_BYTES_PER_LOGIT", DQ9_BASELINE_CHUNK_BYTES_PER_LOGIT)
    config = DQ9_FIXTURES / (row["subject"] + ".json")
    assert hashlib.sha256(config.read_bytes()).hexdigest() == DQ9_REFERENCE["config_sha256"][row["subject"]]
    topology = dense.describe(str(config))
    # Every frozen NF4 code matrix reaches the engine's unchanged 1 MiB cutoff.
    assert min((lin.in_features * lin.out_features + 1) // 2 for lin in topology.layer_linears) >= 1 << 20
    setup = dict(base="nf4", placement=row["placement"], adapter_dtype="fp32", r=16, alpha=32,
                 targets=("attn_in", "attn_out", "mlp_in", "mlp_out"),
                 loss_chunk=0 if row["subject"] == "llama31_8b" else 512, attn_impl="sdpa")
    lines, _, refused = dense.estimate(topology, setup, Workload(seq_len=row["seq"], steps=2))
    assert not refused
    assert not any(line[0] == "frozen decoder linears, kept on device" for line in lines)
    # The separate quant_state correction adds only its source-derived codebooks and offset.
    # Every registered matrix has complete blocks, so there is no rounding correction here.
    quant_state_extra = topology.n_layers * len(topology.layer_linears) * (16 + 256 + 1) * 4
    correction = DQ9_ACTIVATION_CORRECTION.get(f"{row['subject']}-{row['placement']}-{row['seq']}")
    if correction:
        act = next(line for line in lines if line[0] == "activations")
        assert act[4].startswith(correction["branch"])
    activation_extra = correction["activation_correction_bytes"] if correction else 0
    assert sum(line[2] for line in lines if line[1] == "device") == row["device_bytes"] + quant_state_extra + activation_extra
    assert sum(line[2] for line in lines if line[1] == "device" and
               line[0].startswith("frozen decoder linears")) == row["frozen_device_bytes"] + quant_state_extra


def test_linear_biases_are_priced_once_with_the_other_parameters():
    import torch
    from accelerate import init_empty_weights

    config = tr.Qwen2Config(**SMALL)
    topology = dense.describe(config)
    with init_empty_weights():
        tree = tr.AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)
    # Count the real model's parameters independently of the topology's accounting. Qwen2 has q/k/v biases.
    assert any(lin.bias for lin in topology.layer_linears)
    excluded = {id(tree.get_input_embeddings().weight), id(tree.get_output_embeddings().weight)}
    excluded.update(id(mod.weight) for name, mod in tree.named_modules()
                    if isinstance(mod, torch.nn.Linear) and ".layers." in name)
    other_params = sum(p.numel() for p in tree.parameters() if id(p) not in excluded)
    assert other_params == topology.other_params
    result = plan(topology, hw(24), Workload(seq_len=128),
                  Constraints(fixed={"base": "bf16", "placement": "device"}), backends=(dense,))
    lines = {ln.name: ln for ln in result.selected.lines}
    assert lines["norms, biases and other parameters (bf16)"].bytes == 2 * other_params


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
def test_a_14gib_card_streams_the_frozen_layers_and_prices_the_host_and_the_link(q14):
    dense._late_bound_reason.cache_clear()
    # DQ7 now requires 2.4 GB streamed headroom; the former 12 GiB case is separately tested as refused.
    p = plan(q14, hw(14), Workload(seq_len=4096), backends=(dense,))
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


def test_dense_training_executor_is_available_and_serving_remains_planned_only():
    from loggetta.backends.dense_train import run

    assert "pending the registered CUDA proof" in dense.planned_only_reason("train")
    assert dense.executor("train") is run
    assert dense.executor("serve") is None


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


@needs_offload
def test_streamed_dq7_headroom_floor_is_mandatory_at_one_byte_boundary(q14, monkeypatch):
    from loggetta.execution import compare
    from loggetta.plan import ExecutionPlan

    monkeypatch.setattr(dense, "_late_bound_reason", lambda: None)
    fixed = {"base": "nf4", "placement": "stream", "loss_chunk": 512}
    c = Constraints(fixed=fixed, headroom=0)
    wide = plan(q14, hw(200), Workload(seq_len=4096), c, backends=(dense,))
    device = wide.selected.device_bytes
    required = max(dense.DQ7_STREAM_HEADROOM, -(-device//5))
    from dataclasses import replace
    exact = plan(q14, hw(200), wide.workload, replace(c, vram_budget=device+required), backends=(dense,))
    short = plan(q14, hw(200), wide.workload, replace(c, vram_budget=device+required-1), backends=(dense,))
    assert exact.status == "feasible" and short.status == "refused"
    assert exact.budget["headroom"] == required == short.budget["headroom"]
    assert exact.selected.bounds["device_headroom_bytes"] == required
    assert exact.selected.device_bytes == device
    assert any("2.4 GB" in w and "DQ7" in w for w in exact.warnings)
    assert any("2.4 GB" in w for w in short.warnings)
    assert compare(exact, {})["device_allocator"]["estimated"] == allocator(wide.selected)
    assert ExecutionPlan.from_dict(exact.to_dict()).to_dict() == exact.to_dict()
    assert "headroom" in exact.render() and "headroom" in short.render()
    high = plan(q14, hw(200), wide.workload, replace(c, headroom=required+1), backends=(dense,))
    assert high.budget["headroom"] == required+1


def test_dense_warnings_include_refusals_and_resident_plans(q14):
    resident = plan(q14, hw(48), Workload(seq_len=4096), backends=(dense,))
    assert resident.selected.setup["placement"] == "device"
    assert "device_headroom_bytes" not in resident.selected.bounds
    assert any("DQ7" in w and "Llama" in w for w in resident.warnings)
    refused = plan(q14, hw(1), Workload(seq_len=4096), backends=(dense,))
    assert refused.status == "refused" and any("DQ7" in w for w in refused.warnings)


@needs_offload
def test_headroom_bounds_without_link_probe_render_and_time_target(q14, monkeypatch):
    monkeypatch.setattr(dense, "_late_bound_reason", lambda: None)
    monkeypatch.setattr("loggetta.planner.link_bandwidth", lambda *a: (None, "inferred", "fixture"))
    p = plan(q14, hw(24), Workload(seq_len=4096),
             Constraints(fixed={"base": "nf4", "placement": "stream"}, headroom=0, target_s_per_step=100),
             backends=(dense,))
    assert p.status == "feasible"
    assert p.selected.bounds == {"device_headroom_bytes": max(dense.DQ7_STREAM_HEADROOM, -(-p.selected.device_bytes//5))}
    assert "transfer bound" not in p.render()


@needs_offload
def test_large_streamed_plans_receive_proportional_margin_and_scope_warning(q32, monkeypatch):
    monkeypatch.setattr(dense, "_late_bound_reason", lambda: None)
    p = plan(q32, hw(200), Workload(seq_len=8192),
             Constraints(fixed={"base": "nf4", "placement": "stream"}, headroom=0), backends=(dense,))
    assert p.budget["headroom"] == -(-p.selected.device_bytes//5) > dense.DQ7_STREAM_HEADROOM
    assert any("outside DQ7" in warning for warning in p.warnings)


@needs_offload
def test_step_count_and_schedule_do_not_create_memory_scope_warning(q14, monkeypatch):
    monkeypatch.setattr(dense, "_late_bound_reason", lambda: None)
    p = plan(q14, hw(24), Workload(seq_len=2048, steps=20, learning_rate=1e-4, lr_schedule="cosine"),
             Constraints(fixed={"base": "nf4", "placement": "stream"}, headroom=0), backends=(dense,))
    assert p.status == "feasible" and p.budget["headroom"] >= dense.DQ7_STREAM_HEADROOM
    assert any("DQ7" in warning for warning in p.warnings)
    assert not any("outside DQ7" in warning for warning in p.warnings)


def test_a_larger_loss_term_never_lowers_the_activation_estimate(q32, monkeypatch):
    """The activation item is the larger of its two branches as returned: layer inputs + one layer's recompute, or layer
    inputs + the loss's workspace, both scaled as returned. Comparing ``T x layer_work`` with the loss workspace before the
    coefficient scaled the layer branch let a larger loss term switch to the smaller branch: at Qwen3-32B and 2,048 tokens
    the estimate fell when experts4bit-qlora's chunk coefficient rose from 10 to 12 B per logit."""
    import experts4bit_qlora.engines.chunked_lm_loss as cll

    def activations(coef):
        monkeypatch.setattr(cll, "CHUNK_BYTES_PER_LOGIT", coef)
        p = plan(q32, hw(24), Workload(seq_len=2048, micro_batch=1), Constraints())
        assert p.selected is not None and p.selected.setup["loss_chunk"]
        return next(ln for ln in p.selected.lines if ln.name == "activations").bytes

    values = [activations(c) for c in (8, 10, 12, 16, 24)]
    assert values == sorted(values), values


def test_the_fallback_prices_like_the_helper_and_never_falls_as_its_coefficient_rises(q32, monkeypatch):
    """With an experts4bit-qlora that has no ``chunked_loss_bytes``, the dense estimate uses its own named coefficient
    (``CHUNK_LOSS_BYTES_PER_LOGIT``, 12). At the same coefficient it prices exactly as the helper does, and like the
    helper path its activation item never falls as the coefficient rises."""
    import sys

    import experts4bit_qlora.engines.chunked_lm_loss as cll

    def activations():
        p = plan(q32, hw(24), Workload(seq_len=2048, micro_batch=1), Constraints())
        return next(ln for ln in p.selected.lines if ln.name == "activations")

    monkeypatch.setattr(cll, "CHUNK_BYTES_PER_LOGIT", 12)
    helper = activations()
    monkeypatch.setitem(sys.modules, "experts4bit_qlora.engines.chunked_lm_loss", None)   # the import now fails
    assert dense.CHUNK_LOSS_BYTES_PER_LOGIT == 12
    fallback = activations()
    assert fallback.bytes == helper.bytes and "cannot state it" in fallback.detail
    values = []
    for coef in (8, 10, 12, 16, 24):
        monkeypatch.setattr(dense, "CHUNK_LOSS_BYTES_PER_LOGIT", coef)
        values.append(activations().bytes)
    assert values == sorted(values), values
