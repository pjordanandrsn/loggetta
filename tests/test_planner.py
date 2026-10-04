"""Planner behaviour on real topologies (tiny configs through experts4bit-qlora's describe_moe) and stated hardware.

Hardware is an INPUT to the planner, so the tests state it (a 12 GiB sm_86 card, a 40 GiB host) instead of probing
the machine they run on; nothing the planner computes is mocked. Backend probing is real: the grouped kernel is a
candidate only where grouped-nf4-gemm imports.
"""
import json

import pytest

tr = pytest.importorskip("transformers")
pytest.importorskip("experts4bit_qlora.recipe")

from loggetta import Constraints, Workload, describe_model, plan  # noqa: E402
from loggetta.hardware import GPU, Fact, HardwareProfile, Host  # noqa: E402
from loggetta.plan import ExecutionPlan  # noqa: E402

GiB = 1 << 30


def gpu(free_gib=12.0, total_gib=12.0, cap=(8, 6), width=16):
    r = lambda v: Fact(v, "reported")  # noqa: E731
    return GPU(index=0, vendor="nvidia", name="Test GPU", uuid=None, compute_capability=r(cap),
               memory_total=r(int(total_gib * GiB)), memory_free=r(int(free_gib * GiB)), driver=r("575.64.05"),
               pcie_gen_max=r(4), pcie_width_max=r(16), pcie_gen_current=r(4), pcie_width_current=r(width))


def hw(free_gib=12.0, ram_gib=40.0, **kw):
    r = lambda v: Fact(v, "reported")  # noqa: E731
    host = Host(cpu_model=r("test cpu"), cpus=r(8), memory_total=r(64 * GiB), memory_available=r(int(ram_gib * GiB)),
                memory_limit=r(64 * GiB))
    return HardwareProfile(gpus=(gpu(free_gib, **kw),), host=host, platform="Linux x86_64")


@pytest.fixture(scope="module")
def topo():
    # big enough expert slab (~1.4 GiB NF4) that host residency saves a meaningful amount; meta-only, so free
    cfg = tr.Qwen3MoeConfig(hidden_size=1024, intermediate_size=2048, moe_intermediate_size=768, num_experts=128,
                            num_experts_per_tok=4, num_hidden_layers=8, num_attention_heads=8, num_key_value_heads=4,
                            head_dim=128, vocab_size=32000, max_position_embeddings=4096, decoder_sparse_step=1)
    return describe_model(cfg)


def _resident_bytes(topo, seq=512):
    p = plan(topo, hw(), Workload(seq_len=seq), Constraints(expert_residency=("device",), fixed={"attn_4bit": False}))
    return p.selected.device_bytes


def _between_host_and_resident(topo):
    """A device budget the cheapest host-backed setup fits and no resident setup does (headroom included)."""
    host = plan(topo, hw(), Workload(seq_len=512), Constraints(expert_residency=("host",),
                                                               objective="min_vram")).selected.device_bytes
    resident = plan(topo, hw(), Workload(seq_len=512), Constraints(expert_residency=("device",),
                                                                   objective="min_vram")).selected.device_bytes
    assert resident - host > 1 * GiB, "fixture too small for the test"
    budget = (host + resident) // 2
    return (budget + 512 * (1 << 20)) / GiB              # free memory such that budget - headroom sits between them


def test_roomy_card_keeps_everything_resident_on_the_fastest_kernel(topo):
    p = plan(topo, hw(), Workload(seq_len=512))
    assert p.status == "feasible"
    s = p.selected.setup
    assert s["expert_residency"] == "device" and not s["attn_4bit"]
    try:
        import nf4_route  # noqa: F401
        assert s["expert_kernel"] == "grouped_nf4"
    except ImportError:
        assert s["expert_kernel"] == "reference"
    assert any("resident on the device" in r for r in p.reasons)
    assert {ln.basis for ln in p.selected.lines} >= {"derived", "heuristic", "inferred"}
    assert p.performance["estimate"] is None                     # no fabricated speed


def test_plans_are_deterministic_and_round_trip(topo):
    a = plan(topo, hw(), Workload(seq_len=512))
    b = plan(topo, hw(), Workload(seq_len=512))
    assert a.to_json() == b.to_json()
    again = ExecutionPlan.from_dict(json.loads(a.to_json()))
    assert again.to_json() == a.to_json()
    assert "Selected" in a.render() and "Alternatives considered" in a.render()


def test_a_tight_card_moves_experts_to_the_host_and_says_why(topo):
    p = plan(topo, hw(free_gib=_between_host_and_resident(topo)), Workload(seq_len=512))
    assert p.status == "feasible"
    assert p.selected.setup["expert_residency"] == "host"
    assert any("host-backed" in r for r in p.reasons)
    losers = [a for a in p.alternatives if not a.feasible]
    assert losers and all(any("budget" in r for r in a.rejected) for a in losers)


def test_forbidding_offload_on_a_tight_card_is_refused_with_alternatives(topo):
    p = plan(topo, hw(free_gib=_between_host_and_resident(topo)), Workload(seq_len=512),
             Constraints(expert_residency=("device",)))
    assert p.status == "refused" and p.selected is None
    assert p.refusal["closest"] and p.refusal["reasons"]
    sugg = " | ".join(p.refusal["suggestions"])
    assert "more device memory" in sugg and "allow host-backed experts" in sugg
    assert "NOT FEASIBLE" in p.render()


def test_a_large_context_suggests_fewer_tokens(topo):
    big = plan(topo, hw(free_gib=_resident_bytes(topo, 512) / GiB + 0.6), Workload(seq_len=16384),
               Constraints(expert_residency=("device",), fixed={"attn_4bit": False}))
    assert big.status == "refused"
    assert any("reduce tokens per micro-batch" in s for s in big.refusal["suggestions"])


def test_the_host_budget_is_enforced(topo):
    need = _resident_bytes(topo)
    p = plan(topo, hw(free_gib=(need * 0.5) / GiB, ram_gib=1.0), Workload(seq_len=512))
    assert p.status == "refused"
    assert any("host" in r for r in p.refusal["reasons"])


def test_expert_mode_fixes_a_field(topo):
    p = plan(topo, hw(), Workload(seq_len=512), Constraints(fixed={"expert_kernel": "reference", "r": 16}))
    assert p.selected.setup["expert_kernel"] == "reference" and p.selected.setup["r"] == 16
    assert all(a.setup["expert_kernel"] == "reference" for a in p.alternatives)
    with pytest.raises(ValueError):
        plan(topo, hw(), Workload(seq_len=512), Constraints(fixed={"no_such_field": 1}))


def test_objectives_order_differently(topo):
    fast = plan(topo, hw(), Workload(seq_len=512))
    lean = plan(topo, hw(), Workload(seq_len=512), Constraints(objective="min_vram"))
    assert lean.selected.device_bytes <= fast.selected.device_bytes
    assert lean.selected.setup["expert_residency"] == "host"


def test_serving_is_refused_as_a_capability_not_a_crash(topo):
    p = plan(topo, hw(), Workload(kind="serve", context_len=4096, concurrency=1, phase="decode"))
    assert p.status == "refused" and "no installed backend plans 'serve'" in p.refusal["reasons"][0]


def test_a_model_the_loader_refuses_is_refused_with_its_reason():
    cfg = tr.LlamaConfig(hidden_size=256, intermediate_size=512, num_hidden_layers=2, num_attention_heads=4,
                         vocab_size=1000)
    p = plan(describe_model(cfg), hw(), Workload())
    assert p.status == "refused" and "Unsupported model_type='llama'" in p.refusal["reasons"][0]


def test_no_gpu_is_a_refusal(topo):
    h = hw()
    p = plan(topo, HardwareProfile(gpus=(), host=h.host, platform=h.platform), Workload())
    assert p.status == "refused" and "no GPU" in p.refusal["reasons"][0]


def test_a_measured_context_replaces_the_default(topo):
    receipt = {"schema": "execution-receipt/1", "run_id": "r1", "status": "OK",
               "hardware": {"gpu": {"name": "Test GPU", "driver": "575.64.05"}},
               "measured": {"cuda_context_bytes": 300 << 20, "host_baseline_bytes": 2 * GiB}}
    p = plan(topo, hw(), Workload(seq_len=512), observations=[receipt])
    ctx = next(ln for ln in p.selected.lines if ln.name.startswith("CUDA context"))
    assert ctx.basis == "measured" and ctx.bytes == 300 << 20 and "r1" in ctx.detail


def test_a_budget_above_free_memory_is_warned(topo):
    p = plan(topo, hw(free_gib=4.0), Workload(seq_len=512), Constraints(vram_budget=10 * GiB))
    assert any("exceeds what the driver reports free" in x for x in p.warnings)


def test_a_degraded_pcie_link_is_warned_when_experts_stream(topo):
    p = plan(topo, hw(width=8), Workload(seq_len=512), Constraints(expert_residency=("host",)))
    assert any("x8 of x16" in x for x in p.warnings)


def test_host_backed_plans_carry_a_transfer_lower_bound(topo):
    p = plan(topo, hw(), Workload(seq_len=512, grad_accum=4), Constraints(expert_residency=("host",)))
    b = p.selected.bounds
    staging = next(ln for ln in p.selected.lines if ln.where == "link")
    assert b["link_bytes_per_step"] == 4 * staging.bytes                       # per micro-batch x grad-accum
    assert b["link_gbps_basis"] == "inferred" and abs(b["link_gbps"] - 1.969 * 16) < 1e-9
    assert abs(b["s_per_step_lower_bound"] - b["link_bytes_per_step"] / (b["link_gbps"] * 1e9)) < 1e-12
    assert "transfer bound" in p.render()
    assert not plan(topo, hw(), Workload(seq_len=512)).selected.bounds        # resident: nothing crosses the link


def test_an_impossible_step_time_target_refuses_host_backed_setups(topo):
    p = plan(topo, hw(free_gib=_between_host_and_resident(topo)), Workload(seq_len=512),
             Constraints(target_s_per_step=1e-6))
    assert p.status == "refused"
    assert any("host-to-device traffic alone" in r for r in p.refusal["reasons"])


def test_a_measured_link_replaces_the_pcie_ceiling(topo):
    receipt = {"schema": "execution-receipt/1", "run_id": "r2", "status": "OK",
               "hardware": {"gpu": {"name": "Test GPU", "driver": "575.64.05"}}, "measured": {"link_h2d_gbps": 12.5}}
    p = plan(topo, hw(), Workload(seq_len=512), Constraints(expert_residency=("host",)), observations=[receipt])
    assert p.selected.bounds["link_gbps"] == 12.5 and p.selected.bounds["link_gbps_basis"] == "measured"


def test_reserve_slack_comes_from_the_matching_setup(topo):
    def rec(rid, residency, alloc, reserved):
        return {"schema": "execution-receipt/1", "run_id": rid, "status": "OK",
                "hardware": {"gpu": {"name": "Test GPU", "driver": "575.64.05"}},
                "setup": {"expert_residency": residency, "expert_kernel": "grouped_nf4"},
                "measured": {"cuda_context_bytes": 100 << 20, "device_peak_bytes": alloc, "device_reserved_peak_bytes": reserved}}
    obs = [rec("host-run", "host", 100, 140), rec("dev-run", "device", 100, 120)]
    p = plan(topo, hw(), Workload(seq_len=512), Constraints(fixed={"expert_kernel": "grouped_nf4"}), observations=obs)
    line = next(ln for ln in p.selected.lines if ln.name.startswith("allocator reserve"))
    assert p.selected.setup["expert_residency"] == "device"
    assert "dev-run" in line.detail and abs(line.bytes / sum(ln.bytes for ln in p.selected.lines if ln.where == "device"
                                                             and ln.basis in ("derived", "heuristic")) - 0.2) < 1e-3
