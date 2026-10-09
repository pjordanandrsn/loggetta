"""The grouped_nf4 MoE backward line (#43): the padded LoRA delta's bound, the fused workspaces, and the rules mirrored
from grouped-nf4-gemm, pinned against the installed package so the estimate cannot drift from the kernel silently."""
import json
from pathlib import Path

import pytest

from loggetta.backends import experts4bit as e4b
from loggetta.backends.experts4bit import _ladder_up, gnf4_padded_rows_bound

EVIDENCE = Path(__file__).parent.parent / "evidence" / "2026-10-09-a2000-moe-train-residual"
OLMOE = dict(n_experts=64, top_k=8, tokens=1024, hidden=2048, first_out=2048, intermediate=1024)


# --- pinned against grouped-nf4-gemm's kernel/nf4_qlora.py --------------------------------------------------------------

def _nf4_qlora():
    return pytest.importorskip("nf4_qlora", reason="needs grouped-nf4-gemm")


def test_the_mirrored_route_constants_are_the_installed_kernels():
    q = _nf4_qlora()
    assert e4b.GNF4_PAD_BYTES_LIMIT == q._PAD_BYTES_LIMIT
    assert e4b.GNF4_PAD_BUCKETS_MIN_ROWS == q._PAD_BUCKETS_AUTO_MIN_ROWS
    assert e4b.GNF4_PAD_BUCKET_RATIO == q._PAD_BUCKET_RATIO


def test_the_mirrored_ladder_is_the_installed_kernels():
    q = _nf4_qlora()
    if not hasattr(q, "_ladder_up"):
        pytest.skip("this grouped-nf4-gemm has no ladder")
    assert all(_ladder_up(n) == q._ladder_up(n) for n in range(0, 70000))


def test_the_single_block_ladder_engages_for_fp32_adapters_only(monkeypatch):
    """The bound ladders fp32 adapters, as ``NF4_QLORA_SINGLE_LADDER=auto`` (0.44.0's default) does. On a release with no
    single-block ladder the bound only overstates."""
    q = _nf4_qlora()
    if not hasattr(q, "_single_ladder_enabled"):
        pytest.skip("this grouped-nf4-gemm has no single-block ladder (before 0.44.0)")
    torch = pytest.importorskip("torch")
    monkeypatch.delenv("NF4_QLORA_SINGLE_LADDER", raising=False)
    assert q._single_ladder_enabled(torch.float32) and not q._single_ladder_enabled(torch.bfloat16)


def test_buckets_are_the_default_route_past_the_row_gate(monkeypatch):
    q = _nf4_qlora()
    monkeypatch.delenv("NF4_QLORA_PAD_BUCKETS", raising=False)
    monkeypatch.delenv("NF4_QLORA_PAD_BUCKETS_MIN_ROWS", raising=False)
    assert q._pad_buckets_enabled(e4b.GNF4_PAD_BUCKETS_MIN_ROWS)
    assert not q._pad_buckets_enabled(e4b.GNF4_PAD_BUCKETS_MIN_ROWS - 1)


# --- the bound ------------------------------------------------------------------------------------------------------

def test_the_ladder_rounds_up_by_under_a_quarter():
    for n in range(1, 5000):
        r = _ladder_up(n)
        assert r >= n and (r == n if n <= 4 else r < 1.25 * n)


def test_olmoe_single_block_bound_is_experts_times_tokens():
    rows, how = gnf4_padded_rows_bound(**OLMOE, adapter_dtype="bf16")
    assert rows == 64 * 1024 and "single block" in how


def test_the_bound_covers_the_rows_the_a2000_replay_saw():
    """#44's A2 and C replays: ``_lora_delta_padded``'s x for gate_up is ``[G x widest, H]`` in the adapter dtype (bf16)."""
    rows, _ = gnf4_padded_rows_bound(**OLMOE, adapter_dtype="bf16")
    for run in ("A2", "C"):
        groups = json.loads((EVIDENCE / run / "residual.json").read_text())["groups"]
        x = next(g for g in groups if g["frame"].endswith("_lora_delta_padded") and ":540 " in g["frame"])
        seen = max(int(s) for s in x["sizes"]) // (OLMOE["hidden"] * 2)
        assert seen <= rows, (run, seen, rows)


def test_fp32_adapters_ladder_at_most_one_and_a_half_times_the_single_block():
    for shape in (OLMOE, dict(OLMOE, n_experts=60, tokens=900), dict(OLMOE, n_experts=40, tokens=333, top_k=6)):
        single, _ = gnf4_padded_rows_bound(**shape, adapter_dtype="bf16")
        laddered, how = gnf4_padded_rows_bound(**shape, adapter_dtype="fp32")
        assert single <= laddered <= single * 25 / 16 + 1 and "ladder" in how


def test_calls_past_the_row_gate_are_bounded_by_the_buckets():
    shape = dict(OLMOE, tokens=4096)                       # 32,768 routed rows
    rows, how = gnf4_padded_rows_bound(**shape, adapter_dtype="bf16")
    assert rows == 2 * 4096 * 8 and "bucketed" in how
    assert gnf4_padded_rows_bound(**shape, adapter_dtype="fp32")[0] == rows      # the single-block ladder only


def test_the_pad_route_cap_bounds_a_block_the_loop_would_take():
    # 128 experts x 2,000 tokens at H = 4096: the single block would pass 2 GiB, so the auto route caps what is padded
    shape = dict(n_experts=128, top_k=8, tokens=2000, hidden=4096, first_out=3072, intermediate=1536)
    rows, how = gnf4_padded_rows_bound(**shape, adapter_dtype="bf16")
    cap = e4b.GNF4_PAD_BYTES_LIMIT // ((1536 + 4096) * 2)
    assert rows == cap < 128 * 2000 and "capped" in how
    assert gnf4_padded_rows_bound(**shape, adapter_dtype="fp32")[0] <= -(-cap * 25 // 16)


# --- the line in a plan's estimate ----------------------------------------------------------------------------------

tr = pytest.importorskip("transformers")
pytest.importorskip("experts4bit_qlora.recipe")
LINE = "grouped_nf4 MoE backward (above the activation item)"


@pytest.fixture(scope="module")
def topo():
    """An OLMoE-shaped model (E = 64, top-8, H = 2048, I = 1024) with a small vocabulary, so the backward branch, not
    the loss, is the larger activation term."""
    from loggetta import describe_model

    cfg = tr.OlmoeConfig(hidden_size=2048, intermediate_size=1024, num_experts=64, num_experts_per_tok=8,
                         num_hidden_layers=4, num_attention_heads=16, num_key_value_heads=16, vocab_size=8192,
                         max_position_embeddings=4096)
    return describe_model(cfg)


def _estimate(topo, **setup):
    from loggetta import Workload

    base = dict(quant_type="nf4", blocksize=64, r=8, alpha=16, adapter_dtype="bf16", train_experts=True,
                train_attention=True, attn_4bit=False, expert_residency="device", pin=True, expert_kernel="grouped_nf4",
                dgrad=True, keep_moe_layers=0)
    lines, unmodelled, refusals = e4b.estimate(topo, {**base, **setup}, Workload(seq_len=512, micro_batch=2))
    assert not refusals
    return {ln[0]: ln for ln in lines}, unmodelled


def test_the_grouped_kernel_prices_its_backward_branch(topo):
    lines, unmodelled = _estimate(topo)
    T, H, L = 1024, 2048, 4
    want = (L * T * H * 2 + 64 * 1024 * (2048 + 2048 + 1024) * 2 + 5 * T * 8 * 2048 * 2) - lines["activations"][2]
    assert lines[LINE][2] == want > 0 and lines[LINE][1] == "device" and lines[LINE][3] == "heuristic"
    assert any("NF4_QLORA_SINGLE_LADDER" in u for u in unmodelled)
    assert "adapter gradients" in lines                     # the expert LoRA gradients stay priced once, there


def test_the_reference_kernel_is_unchanged(topo):
    lines, unmodelled = _estimate(topo, expert_kernel="reference")
    assert LINE not in lines and not any("NF4_QLORA" in u for u in unmodelled)


def test_host_residency_prices_the_same_branch(topo):
    assert _estimate(topo, expert_residency="host")[0][LINE][2] == _estimate(topo)[0][LINE][2]


def test_fp32_adapters_price_the_padded_delta_at_four_bytes(topo):
    bf16, fp32 = _estimate(topo)[0][LINE][2], _estimate(topo, adapter_dtype="fp32")[0][LINE][2]
    assert fp32 - bf16 == 64 * 1024 * (2048 + 2048 + 1024) * 2


def test_no_line_when_the_loss_is_the_larger_term(topo):
    big_vocab = e4b.describe(tr.OlmoeConfig(hidden_size=2048, intermediate_size=1024, num_experts=64,
                                            num_experts_per_tok=8, num_hidden_layers=4, num_attention_heads=16,
                                            num_key_value_heads=16, vocab_size=200000, max_position_embeddings=4096))
    assert LINE not in _estimate(big_vocab)[0]


def test_a_plan_carries_the_line(topo):
    from loggetta import Constraints, Workload, plan
    from loggetta.hardware import GPU, Fact, HardwareProfile, Host

    r = lambda v: Fact(v, "reported")  # noqa: E731
    gpu = GPU(index=0, vendor="nvidia", name="Test GPU", uuid=None, compute_capability=r((8, 6)),
              memory_total=r(24 << 30), memory_free=r(24 << 30), driver=r("575.64.05"), pcie_gen_max=r(4),
              pcie_width_max=r(16), pcie_gen_current=r(4), pcie_width_current=r(16))
    host = Host(cpu_model=r("test cpu"), cpus=r(8), memory_total=r(64 << 30), memory_available=r(40 << 30),
                memory_limit=r(64 << 30))
    p = plan(topo, HardwareProfile(gpus=(gpu,), host=host, platform="Linux x86_64"), Workload(seq_len=512, micro_batch=2),
             Constraints(fixed={"expert_kernel": "grouped_nf4", "expert_residency": "device"}))
    if p.selected is None or p.selected.setup.get("expert_kernel") != "grouped_nf4":
        pytest.skip("grouped-nf4-gemm is not usable here, so no grouped_nf4 plan")
    assert any(ln.name == LINE for ln in p.selected.lines)
