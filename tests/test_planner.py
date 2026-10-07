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


def _serve(topo, ctx, seqs, constraints=Constraints(), **hwkw):
    pytest.importorskip("experts4bit_qlora.serve_recipe")
    pytest.importorskip("fp8_paged_attn", reason="needs grouped-nf4-gemm")
    return plan(topo, hw(**hwkw), Workload(kind="serve", context_len=ctx, concurrency=seqs), constraints)


def _can_graph():
    """Whether the experts4bit-qlora under test can say where its decode graphs run (fused_append_unsupported)."""
    try:
        from experts4bit_qlora.engines.fp8_paged_kv import fused_append_unsupported  # noqa: F401
        return True
    except ImportError:
        return False


def test_serving_plans_the_paged_server_with_its_kv_pool(topo):
    p = _serve(topo, 4096, 8, cap=(12, 0))
    assert p.status == "feasible", p.render()
    s = p.selected.setup
    assert (s["placement"], s["max_seqs"], s["max_tokens_per_seq"], s["graphs"]) == ("all-vram", 8, 4096, _can_graph())
    kv = next(ln for ln in p.selected.lines if ln.name == "FP8 paged KV pool")
    assert kv.basis == "derived" and kv.where == "device"
    assert any("KV pool" in r for r in p.reasons) and not any("grouped_nf4" in r for r in p.reasons)
    assert "serve all-vram" in p.selected.label()
    assert "4096 tokens per sequence x 8 sequences" in p.render()
    assert any("E4B_PAGED_MAX_SEQS=8" in r and "E4B_PAGED_MAX_TOKENS_PER_SEQ=4096" in r for r in p.reasons)
    again = ExecutionPlan.from_dict(json.loads(p.to_json()))
    assert again.to_json() == p.to_json() == _serve(topo, 4096, 8, cap=(12, 0)).to_json()
    from loggetta.execution import PlanNotExecutable, execute
    with pytest.raises(PlanNotExecutable, match="planned only"):
        execute(p)


def test_serving_too_much_kv_is_refused_with_a_context_and_a_concurrency_that_fit(topo):
    p = _serve(topo, 131072, 64)
    assert p.status == "refused"
    sugg = p.refusal["suggestions"]
    ctx = next(s for s in sugg if s.startswith("context "))
    seqs = next(s for s in sugg if s.startswith("concurrency "))
    fit_ctx, fit_seqs = int(ctx.split()[1]), int(seqs.split()[1])
    assert fit_ctx < 131072 and fit_ctx % 16 == 0 and fit_seqs < 64
    assert _serve(topo, fit_ctx, 64).status == "feasible"           # the suggestions are plans, not guesses
    assert _serve(topo, fit_ctx + 16, 64).status == "refused"        # and the largest ones at block granularity
    assert _serve(topo, 131072, fit_seqs).status == "feasible"


def _can_solve():
    """Whether the experts4bit-qlora under test prices the solver's tiers, and its CPU tier can run here."""
    from experts4bit_qlora.serve_recipe import ServeSetup
    try:
        import cpu_grouped
        return "vram_gb" in ServeSetup.__dataclass_fields__ and cpu_grouped.cpu_kernels_available()
    except ImportError:
        return False


def _tier(p, prefix):
    return next((ln.bytes for ln in p.selected.lines if ln.name.startswith(prefix)), 0)


def test_serving_host_residency_plans_the_solver_tiers(topo):
    p = _serve(topo, 4096, 1, Constraints(expert_residency=("host",)))
    if not _can_solve():
        assert p.status == "refused"
        return
    assert p.status == "feasible" and p.selected.setup["placement"] == "solver"
    assert all(c.setup["placement"] == "solver" for c in p.alternatives)


def test_serving_fills_the_solver_tiers_when_all_vram_does_not_fit(topo):
    if not _can_solve():
        pytest.skip("the solver's tiers are not priced or cannot run here")
    need = _serve(topo, 4096, 1, cap=(12, 0)).selected.device_bytes
    p = _serve(topo, 4096, 1, cap=(12, 0), free_gib=0.8 * need / GiB)
    assert p.status == "feasible" and p.selected.setup["placement"] == "solver", p.render()
    vram, dram, nvme = _tier(p, "expert stacks, VRAM"), _tier(p, "expert stacks, DRAM"), _tier(p, "expert rows on NVMe")
    assert vram > 0 and dram > 0 and nvme == 0                       # 40 GiB of host takes every other row
    slack = p.budget["device"] - p.budget["headroom"] - p.selected.device_bytes
    assert 0 <= slack < 64 << 20                                      # the VRAM tier filled the device budget
    assert any("experts split across tiers" in r for r in p.reasons)
    assert any("E4B_PAGED_VRAM_GB=" in r and "E4B_PAGED_PLACEMENT=solver" in r for r in p.reasons)
    assert any(c.setup["placement"] == "all-vram" and not c.feasible for c in p.alternatives)


def test_serving_spills_to_nvme_when_the_host_is_short(topo):
    if not _can_solve():
        pytest.skip("the solver's tiers are not priced or cannot run here")
    need = _serve(topo, 4096, 1, cap=(12, 0)).selected.device_bytes
    p = _serve(topo, 4096, 1, cap=(12, 0), free_gib=0.8 * need / GiB, ram_gib=4.6)
    assert p.status == "feasible", p.render()
    assert _tier(p, "expert rows on NVMe") > 0 and _tier(p, "cold view") > 0
    assert p.selected.host_bytes + p.budget["host_headroom"] <= p.budget["host"]
    try:
        from experts4bit_qlora.serve_recipe import min_hot_rows  # noqa: F401
    except ImportError:
        return
    h = p.selected.setup["hot_rows"]                  # the fewest a cold layer can route: top-4 x 512, <= 128, <= NVMe rows
    view = next(ln.bytes for ln in p.selected.lines if ln.name.startswith("cold view"))   # h rows of the arena
    nvme_rows = _tier(p, "expert rows on NVMe") * h // view
    assert h == min(128, 4 * 512, nvme_rows) and any("cold tier: hot_rows" in r for r in p.reasons)


def test_forcing_device_residency_keeps_the_solver_out(topo):
    need = _serve(topo, 4096, 1, cap=(12, 0)).selected.device_bytes
    p = _serve(topo, 4096, 1, Constraints(expert_residency=("device",)), cap=(12, 0), free_gib=0.8 * need / GiB)
    assert p.status == "refused"
    assert all(c.setup["placement"] == "all-vram" for c in p.alternatives)


def test_serving_does_not_borrow_training_slack(topo):
    pytest.importorskip("fp8_paged_attn", reason="needs grouped-nf4-gemm")
    rec = {"run_id": "train-1", "status": "OK", "model": {"model": topo.model}, "workload": {"tokens_per_microbatch": 512},
           "setup": {"expert_residency": "device", "expert_kernel": "grouped_nf4"},
           "hardware": {"gpu": {"name": "Test GPU", "driver": "575.64.05"}},
           "measured": {"cuda_context_bytes": 300 << 20, "device_peak_bytes": 4 * GiB,
                        "device_reserved_peak_bytes": 6 * GiB}}          # 50% slack: a trainer's churn
    p = plan(topo, hw(), Workload(kind="serve", context_len=4096, concurrency=1), observations=[rec])
    res = next(ln for ln in p.selected.lines if ln.name.startswith("allocator reserve"))
    ctx = next(ln for ln in p.selected.lines if ln.name.startswith("CUDA context"))
    assert res.basis == "inferred" and "no serve receipt" in res.detail
    assert ctx.basis == "measured" and ctx.bytes == 300 << 20          # the context is the GPU's, not the workload's


def test_training_plans_do_not_mention_serving_kernels(topo):
    p = plan(topo, hw(), Workload(seq_len=512))
    assert not any("paged_fp8" in r for r in p.reasons)


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


def test_an_unmeasured_setup_gets_the_conservative_slack(topo):
    def rec(rid, residency, alloc, reserved):
        return {"schema": "execution-receipt/1", "run_id": rid, "status": "OK",
                "hardware": {"gpu": {"name": "Test GPU", "driver": "575.64.05"}},
                "setup": {"expert_residency": residency, "expert_kernel": "grouped_nf4"},
                "measured": {"cuda_context_bytes": 100 << 20, "device_peak_bytes": alloc, "device_reserved_peak_bytes": reserved}}
    obs = [rec("dev-run", "device", 100, 110), rec("host-run", "host", 100, 140)]
    p = plan(topo, hw(), Workload(seq_len=512), Constraints(fixed={"expert_kernel": "reference"}), observations=obs)
    line = next(ln for ln in p.selected.lines if ln.name.startswith("allocator reserve"))
    assert "conservative" in line.detail and "host-run" in line.detail


def test_slack_for_an_unmeasured_gpu_transfers_through_an_anchor(topo):
    def rec(rid, gpu_name, model, alloc, reserved):
        return {"schema": "execution-receipt/1", "run_id": rid, "status": "OK", "model": {"model": model},
                "hardware": {"gpu": {"name": gpu_name, "driver": "x"}},
                "setup": {"expert_residency": "device", "expert_kernel": "grouped_nf4"},
                "measured": {"device_peak_bytes": alloc, "device_reserved_peak_bytes": reserved}}
    model = topo.model
    obs = [rec("big-on-5090", "RTX 5090", model, 100, 110),           # this model, measured elsewhere: 0.10
           rec("anchor-on-5090", "RTX 5090", "anchor", 100, 115),      # anchor there: 0.15
           rec("anchor-here", "Test GPU", "anchor", 100, 130)]         # anchor here: 0.30 -> transferred 0.10 x 0.30 / 0.15
    p = plan(topo, hw(), Workload(seq_len=512), Constraints(fixed={"expert_kernel": "grouped_nf4", "attn_4bit": False},
                                                            expert_residency=("device",)), observations=obs)
    line = next(ln for ln in p.selected.lines if ln.name.startswith("allocator reserve"))
    alloc = sum(ln.bytes for ln in p.selected.lines if ln.where == "device" and ln.basis in ("derived", "heuristic")
                and not ln.name.startswith("allocator reserve"))
    assert line.basis == "heuristic" and "transferred" in line.detail and "big-on-5090" in line.detail
    assert abs(line.bytes / alloc - 0.20) < 1e-3


def test_serving_slack_is_not_transferred_through_an_anchor():
    """Serving slack is 0.06-1.5%: an anchor ratio of two such measurements is noise (a ~14x ratio turned OLMoE's 2.8% on
    an RTX 5090 into 4.0 GiB of reserve on an RTX 4090). A serve plan takes this GPU's same-setup slack instead."""
    from loggetta.planner import reserve_fraction
    g = gpu()

    def rec(rid, gpu_name, model, frac):
        return {"run_id": rid, "status": "OK", "model": {"model": model}, "workload": {"kind": "serve"},
                "setup": {"placement": "all-vram", "graphs": True}, "hardware": {"gpu": {"name": gpu_name}},
                "measured": {"device_peak_bytes": 100 * GiB, "device_reserved_peak_bytes": int(100 * GiB * (1 + frac))}}
    obs = [rec("mine-on-5090", "RTX 5090", "m", 0.028), rec("anchor-on-5090", "RTX 5090", "anchor", 0.0006),
           rec("anchor-here", g.name, "anchor", 0.0083)]
    f, basis, src = reserve_fraction(g, {"placement": "all-vram", "graphs": True}, obs, (0.2, "inferred", "default"),
                                     model="m", kind="serve", key=("placement", "graphs"))
    assert abs(f - 0.0083) < 1e-6 and basis == "measured" and "anchor-here" in src and "transferred" not in src


def test_serving_below_sm89_plans_eager_decode_and_says_why(topo):
    p = _serve(topo, 4096, 8)                                 # the stated test card is sm_86
    assert p.status == "feasible"
    assert all(c.setup["graphs"] is False for c in (p.selected,) + p.alternatives)
    assert any("paged_graphs not usable" in r for r in p.reasons)
    assert any("E4B_PAGED_GRAPHS=0" in r for r in p.reasons)


def test_serve_slack_prefers_the_receipt_with_the_whole_setup(topo):
    pytest.importorskip("fp8_paged_attn", reason="needs grouped-nf4-gemm")
    if not _can_graph():
        pytest.skip("this experts4bit-qlora plans no decode graphs")
    base = _serve(topo, 4096, 4, cap=(12, 0)).selected.setup

    def rec(rid, graphs, reserved):
        return {"run_id": rid, "status": "OK", "model": {"model": topo.model}, "workload": {"kind": "serve"},
                "setup": {**base, "graphs": graphs, "buckets": list(base["buckets"])},
                "hardware": {"gpu": {"name": "Test GPU", "driver": "575.64.05"}},
                "measured": {"cuda_context_bytes": 200 << 20, "device_peak_bytes": 4 * GiB,
                             "device_reserved_peak_bytes": reserved}}
    obs = [rec("eager", False, int(4.4 * GiB)), rec("graphs", True, int(5.0 * GiB))]
    p = plan(topo, hw(cap=(12, 0)), Workload(kind="serve", context_len=4096, concurrency=4), observations=obs)
    by = {c.setup["graphs"]: next(ln for ln in c.lines if ln.name.startswith("allocator reserve"))
          for c in (p.selected,) + p.alternatives if c.setup["placement"] == "all-vram"}
    assert by[True].basis == by[False].basis == "measured"
    assert "receipt graphs" in by[True].detail and "receipt eager" in by[False].detail


def test_serving_plans_the_prefill_graph_off_unless_asked(topo):
    from experts4bit_qlora.serve_recipe import ServeSetup
    if "prefill_graph" not in ServeSetup.__dataclass_fields__:
        pytest.skip("this experts4bit-qlora's ServeSetup has no prefill_graph")
    p = _serve(topo, 4096, 8, cap=(12, 0))
    assert p.selected.setup["prefill_graph"] == "0"
    assert any("prefill graph off" in r for r in p.reasons)
    assert any("E4B_PAGED_PREFILL_GRAPH=0" in r for r in p.reasons)
    asked = _serve(topo, 4096, 8, Constraints(fixed={"prefill_graph": "auto"}), cap=(12, 0))
    assert asked.selected.setup["prefill_graph"] == "auto" and "prefill graph auto" in asked.selected.label()
    assert any("prefill graph" in u for u in asked.selected.unmodelled)


def test_a_solver_receipts_slack_does_not_reach_all_vram_plans(topo):
    if not _can_solve():
        pytest.skip("the solver's tiers are not priced or cannot run here")
    base = _serve(topo, 4096, 1, cap=(12, 0)).selected.setup

    def rec(rid, placement, reserved):
        return {"run_id": rid, "status": "OK", "model": {"model": topo.model}, "workload": {"kind": "serve"},
                "setup": {**base, "placement": placement, "graphs": False, "buckets": list(base["buckets"])},
                "hardware": {"gpu": {"name": "Test GPU", "driver": "575.64.05"}},
                "measured": {"device_peak_bytes": 4 * GiB, "device_reserved_peak_bytes": reserved}}
    obs = [rec("tiers", "solver", int(4.6 * GiB)), rec("resident", "all-vram", int(4.04 * GiB))]
    p = plan(topo, hw(cap=(12, 0)), Workload(kind="serve", context_len=4096, concurrency=1),
             Constraints(fixed={"graphs": False}), observations=obs)
    by = {c.setup["placement"]: next(ln for ln in c.lines if ln.name.startswith("allocator reserve"))
          for c in (p.selected,) + p.alternatives}
    assert "receipt resident" in by["all-vram"].detail and "receipt tiers" in by["solver"].detail


def test_serve_plans_learn_the_residual_and_host_growth_from_this_models_receipts(topo):
    first = _serve(topo, 4096, 1, cap=(12, 0))
    planner_lines = ("allocator reserve", "CUDA context", "allocator residual")   # what the planner adds itself
    setup, est = first.selected.setup, sum(ln.bytes for ln in first.selected.lines if ln.where == "device"
                                          and not ln.name.startswith(planner_lines))
    rec = {"run_id": "seen", "status": "OK", "model": {"model": topo.model}, "workload": {"kind": "serve"},
           "setup": {**setup, "buckets": list(setup["buckets"]), "torch_threads": "2"},
           "hardware": {"gpu": {"name": "Other GPU", "driver": "1"}},
           "measured": {"device_peak_bytes": est + (200 << 20), "device_reserved_peak_bytes": est + (210 << 20),
                        "host_anon_after_load_bytes": 2 * GiB, "host_anon_peak_bytes": 2 * GiB + (700 << 20)}}
    p = plan(topo, hw(cap=(12, 0)), Workload(kind="serve", context_len=4096, concurrency=1),
             Constraints(fixed={"graphs": setup["graphs"]}), observations=[rec])
    lines = {ln.name: ln for ln in p.selected.lines}
    res = lines["allocator residual (runtime buffers no item prices)"]
    assert res.basis == "measured" and abs(res.bytes - (200 << 20)) < (1 << 20) and "receipt seen" in res.detail
    assert lines["host growth while serving"].bytes == 700 << 20
    other = plan(topo, hw(cap=(12, 0)), Workload(kind="serve", context_len=4096, concurrency=1),
                 observations=[{**rec, "model": {"model": "someone/else"}}])
    assert not any(ln.name.startswith(("allocator residual", "host growth")) for ln in other.selected.lines)


def test_a_receipt_licensed_for_some_uses_teaches_only_those(topo):
    """Lane SV6 (experts4bit-qlora#1275) licenses its receipts as same-setup evidence for the allocator reserve and the
    CUDA context on its card class, and nothing more: its host growth and residual are not learned. A receipt without
    ``licensed_for`` teaches every use, as before."""
    from loggetta.planner import licensed
    first = _serve(topo, 4096, 1, cap=(12, 0))
    planner_lines = ("allocator reserve", "CUDA context", "allocator residual")
    setup, est = first.selected.setup, sum(ln.bytes for ln in first.selected.lines if ln.where == "device"
                                          and not ln.name.startswith(planner_lines))
    rec = {"run_id": "scoped", "status": "OK", "model": {"model": topo.model}, "workload": {"kind": "serve"},
           "setup": {**setup, "buckets": list(setup["buckets"])}, "hardware": {"gpu": {"name": "Test GPU", "driver": "1"}},
           "measured": {"device_peak_bytes": est + (200 << 20), "device_reserved_peak_bytes": est + (400 << 20),
                        "host_anon_after_load_bytes": 2 * GiB, "host_anon_serving_peak_bytes": 2 * GiB + (700 << 20)},
           "licensed_for": ["reserve", "context"]}
    assert licensed(rec, "reserve") and not licensed(rec, "host_growth") and licensed({}, "host_growth")
    run = (topo, hw(cap=(12, 0)), Workload(kind="serve", context_len=4096, concurrency=1),
           Constraints(fixed={"graphs": setup["graphs"]}))
    lines = {ln.name: ln for ln in plan(*run, observations=[rec]).selected.lines}
    assert "receipt scoped" in lines["allocator reserve (cached, unallocated blocks)"].detail
    assert not any(n.startswith(("allocator residual", "host growth")) for n in lines)
    full = {k: v for k, v in rec.items() if k != "licensed_for"}
    lines = {ln.name: ln for ln in plan(*run, observations=[full]).selected.lines}
    assert lines["host growth while serving"].bytes == 700 << 20 and "allocator residual" in " ".join(lines)


def test_reserve_matches_the_whole_setup_then_its_shape_then_its_key_fields():
    """Tier budgets move a few one-time allocations, not the workload's transient ones: a receipt that differs from the
    candidate only in the fields the planner sizes (lanes SV4 and SV6: 4.1% and 3.8% at two VRAM tiers of one 8 x 8192
    shape) is preferred to a same-key receipt of another shape (6.7% at 4 x 4096 with an NVMe tier). A receipt scoped by
    ``licensed_for`` counts only for its own whole setup (lane SV6's licence)."""
    from loggetta.planner import reserve_fraction
    g = gpu()
    base = {"placement": "solver", "max_seqs": 8, "max_tokens_per_seq": 8192, "graphs": False, "vram_gb": 12.6,
            "dram_gb": 15.2, "hot_rows": 1}

    def rec(rid, setup, frac, scope=None):
        r = {"run_id": rid, "status": "OK", "model": {"model": "m"}, "workload": {"kind": "serve"}, "setup": setup,
             "hardware": {"gpu": {"name": g.name}},
             "measured": {"device_peak_bytes": 100 * GiB, "device_reserved_peak_bytes": int(100 * GiB * (1 + frac))}}
        return {**r, "licensed_for": scope} if scope else r

    shape = rec("shape", {**base, "vram_gb": 10.9, "hot_rows": 4}, 0.041)
    other = rec("other", {**base, "max_seqs": 4, "max_tokens_per_seq": 4096, "vram_gb": 8.0}, 0.067)
    kw = dict(default=(0.2, "inferred", "default"), model="m", kind="serve", key=("placement", "graphs"))
    budgets = ("vram_gb", "dram_gb", "hot_rows")
    f, basis, src = reserve_fraction(g, base, [shape, other], budget_fields=budgets, **kw)
    assert abs(f - 0.041) < 1e-6 and basis == "measured" and "receipt shape" in src and "same shape" in src
    f, _, src = reserve_fraction(g, base, [shape, other], **kw)            # a backend that names no budget fields
    assert abs(f - 0.067) < 1e-6 and "same key fields" in src
    f, _, src = reserve_fraction(g, base, [shape, other, rec("whole", dict(base), 0.03)], budget_fields=budgets, **kw)
    assert abs(f - 0.03) < 1e-6 and "this GPU, setup and model" in src
    scoped_shape = rec("scoped", {**base, "vram_gb": 10.9}, 0.01, scope=["reserve", "context"])
    f, _, src = reserve_fraction(g, base, [scoped_shape, other], budget_fields=budgets, **kw)
    assert abs(f - 0.067) < 1e-6 and "receipt other" in src                # scoped: not borrowed for another setup
    scoped_whole = rec("scoped-whole", dict(base), 0.01, scope=["reserve", "context"])
    f, _, src = reserve_fraction(g, base, [scoped_whole, other], budget_fields=budgets, **kw)
    assert abs(f - 0.01) < 1e-6 and "receipt scoped-whole" in src


def test_every_serve_receipt_on_file_names_bulk_kv():
    """A serve plan's setup names ``bulk_kv`` (experts4bit-qlora #1247), so a receipt that does not can never match one
    whole or by shape. bench/annotate_bulk_kv.py recorded each from the code its run used; the seat's 2026-10-06 family
    runs recorded no commit and stay without it. A new import must record the field."""
    import glob
    import json
    import os
    root = os.path.join(os.path.dirname(__file__), "..", "evidence")
    unknown = {"2026-10-06-a2000-family-serve"}
    missing = []
    for f in glob.glob(os.path.join(root, "*", "*.json")):
        r = json.load(open(f))
        if r.get("schema") == "execution-receipt/1" and r.get("workload", {}).get("kind") == "serve" \
                and "bulk_kv" not in (r.get("setup") or {}) and os.path.basename(os.path.dirname(f)) not in unknown:
            missing.append(os.path.relpath(f, root))
    assert not missing, missing


def test_serve_slack_is_borrowed_across_cards_only_for_serving(topo):
    def rec(rid, kind, setup):
        return {"run_id": rid, "status": "OK", "model": {"model": "other/model"}, "workload": {"kind": kind},
                "setup": setup, "hardware": {"gpu": {"name": "Another GPU", "driver": "1"}},
                "measured": {"device_peak_bytes": 4 * GiB, "device_reserved_peak_bytes": int(4.08 * GiB)}}
    serve = _serve(topo, 4096, 1, cap=(12, 0)).selected.setup
    obs = [rec("served-elsewhere", "serve", {**serve, "buckets": list(serve["buckets"])})]
    p = plan(topo, hw(cap=(12, 0)), Workload(kind="serve", context_len=4096, concurrency=1),
             Constraints(fixed={"graphs": serve["graphs"]}), observations=obs)
    res = next(ln for ln in p.selected.lines if ln.name.startswith("allocator reserve"))
    assert res.basis == "heuristic" and "served-elsewhere" in res.detail and "Another GPU" in res.detail
    trained = plan(topo, hw(), Workload(seq_len=512), observations=[rec("trained-elsewhere", "train", {
        "expert_residency": "device", "expert_kernel": "grouped_nf4"})])
    tres = next(ln for ln in trained.selected.lines if ln.name.startswith("allocator reserve"))
    assert tres.basis == "inferred"


def _has_int4_levers():
    try:
        from experts4bit_qlora.serve_recipe import ServeSetup
    except ImportError:
        return False
    return "exp_int4" in ServeSetup.__dataclass_fields__


def test_the_int4_levers_are_planned_only_when_fixed(topo):
    if not _has_int4_levers():
        pytest.skip("the experts4bit-qlora under test does not price the int4 serve levers")
    base = _serve(topo, 4096, 4, cap=(12, 0))
    assert not base.selected.setup["exp_int4"] and not base.selected.setup["attn_int4"]
    assert any("int4 levers off" in r for r in base.reasons)
    p = _serve(topo, 4096, 4, Constraints(fixed={"exp_int4": True, "attn_int4": True}), cap=(12, 0))
    assert p.status == "feasible", p.render()
    names = {ln.name for ln in p.selected.lines}
    assert "int4 expert stores (all VRAM; the NF4 stacks freed)" in names and not any(
        n.startswith("frozen expert stacks") for n in names)
    assert "attention projections' bf16 copy (kept from the first prefill)" in names
    assert "int4 experts" in p.selected.label() and "int4 attention" in p.selected.label()
    assert any("source checkpoint" in r for r in p.reasons) and any("not a memory one" in r for r in p.reasons)
    assert any("E4B_SERVE_EXP_INT4=1" in r and "E4B_SERVE_ATTN_INT4=1" in r for r in p.reasons)
    assert p.selected.host_bytes > base.selected.host_bytes            # the repack's fp32 layer read
    assert all(c.setup.get("placement") == "all-vram" for c in [p.selected])


def test_receipts_from_before_the_int4_levers_match_the_levers_off(topo):
    if not _has_int4_levers():
        pytest.skip("the experts4bit-qlora under test does not price the int4 serve levers")
    setup = _serve(topo, 4096, 1, cap=(12, 0)).selected.setup
    old = {k: v for k, v in setup.items() if k not in ("exp_int4", "attn_int4")}   # written before the fields existed
    rec = {"run_id": "before-int4", "status": "OK", "model": {"model": topo.model}, "workload": {"kind": "serve"},
           "setup": {**old, "buckets": list(old["buckets"])}, "hardware": {"gpu": {"name": "Test GPU", "driver": "1"}},
           "measured": {"device_peak_bytes": 4 * GiB, "device_reserved_peak_bytes": int(4.04 * GiB)}}
    p = plan(topo, hw(cap=(12, 0)), Workload(kind="serve", context_len=4096, concurrency=1),
             Constraints(fixed={"graphs": setup["graphs"]}), observations=[rec])
    res = next(ln for ln in p.selected.lines if ln.name.startswith("allocator reserve"))
    assert res.basis == "measured" and "before-int4" in res.detail and "this GPU, setup and model" in res.detail
    q = plan(topo, hw(cap=(12, 0)), Workload(kind="serve", context_len=4096, concurrency=1),
             Constraints(fixed={"graphs": setup["graphs"], "exp_int4": True}), observations=[rec])
    qres = next(ln for ln in q.selected.lines if ln.name.startswith("allocator reserve"))
    assert "this GPU, setup and model" not in qres.detail             # an NF4 receipt is not the int4 setup's


def test_host_growth_is_serving_s_own_peak_where_the_receipt_has_one(topo):
    """A load can peak above everything after it (the int4 repack hands its host buffers back before serving): growth
    while serving is measured from the serving phase's own peak when the receipt records one."""
    first = _serve(topo, 4096, 1, cap=(12, 0))
    setup = first.selected.setup
    rec = {"run_id": "load-peaked", "status": "OK", "model": {"model": topo.model}, "workload": {"kind": "serve"},
           "setup": {**setup, "buckets": list(setup["buckets"])}, "hardware": {"gpu": {"name": "Other GPU", "driver": "1"}},
           "measured": {"device_peak_bytes": 1, "device_reserved_peak_bytes": 1, "host_anon_after_load_bytes": GiB,
                        "host_anon_peak_bytes": 7 * GiB, "host_anon_serving_peak_bytes": GiB + (300 << 20)}}
    p = plan(topo, hw(cap=(12, 0)), Workload(kind="serve", context_len=4096, concurrency=1),
             Constraints(fixed={"graphs": setup["graphs"]}), observations=[rec])
    assert next(ln for ln in p.selected.lines if ln.name == "host growth while serving").bytes == 300 << 20
    old = {**rec, "measured": {k: v for k, v in rec["measured"].items() if k != "host_anon_serving_peak_bytes"}}
    q = plan(topo, hw(cap=(12, 0)), Workload(kind="serve", context_len=4096, concurrency=1),
             Constraints(fixed={"graphs": setup["graphs"]}), observations=[old])
    assert next(ln for ln in q.selected.lines if ln.name == "host growth while serving").bytes == 6 * GiB


def test_serve_plans_cap_the_decode_buckets_at_the_sequences():
    from loggetta.backends.experts4bit import usable_buckets
    assert usable_buckets(1, (1, 2, 4, 8, 16)) == (1,)
    assert usable_buckets(8, (1, 2, 4, 8, 16)) == (1, 2, 4, 8)
    assert usable_buckets(12, (1, 2, 4, 8, 16)) == (1, 2, 4, 8, 12)
    assert usable_buckets(32, (1, 2, 4, 8, 16)) == (1, 2, 4, 8, 16)       # wider steps run in chunks of the largest
    if not _can_graph():
        pytest.skip("decode graphs are not planned by the experts4bit-qlora under test")
    topo = describe_model(tr.Qwen3MoeConfig(
        hidden_size=1024, intermediate_size=2048, moe_intermediate_size=768, num_experts=128, num_experts_per_tok=4,
        num_hidden_layers=8, num_attention_heads=8, num_key_value_heads=4, head_dim=128, vocab_size=32000,
        max_position_embeddings=4096, decoder_sparse_step=1))
    one = _serve(topo, 4096, 1, cap=(12, 0))
    assert one.selected.setup["buckets"] == (1,) and any("buckets [1]" in r for r in one.reasons)
    fixed = _serve(topo, 4096, 1, Constraints(fixed={"buckets": (1, 2, 4, 8, 16)}), cap=(12, 0))
    assert tuple(fixed.selected.setup["buckets"]) == (1, 2, 4, 8, 16)
    try:                                    # experts4bit-qlora#1234: the server drops unusable buckets itself
        from experts4bit_qlora.serve_recipe import usable_buckets  # noqa: F401
        server_clamps = True
    except ImportError:
        server_clamps = False
    if server_clamps:
        assert one.selected.device_bytes == fixed.selected.device_bytes            # priced as the server captures
    else:
        assert one.selected.device_bytes < fixed.selected.device_bytes             # fewer scratch slots


_OOM = ("CUDA out of memory. Tried to allocate 32.00 MiB. GPU 0 has a total capacity of 11.20 GiB of which 9.75 MiB is "
        "free. Of the allocated memory 10.59 GiB is allocated by PyTorch")


def test_an_out_of_memory_receipt_caps_the_budget_at_what_a_process_gets(topo):
    """Lane SV5: an RTX 4090 reports 24,564 MiB and gave the process 23.52 GiB; a plan sized against 24 GiB ran out of
    memory. A receipt's out-of-memory message names the capacity, and the budget is capped there for that GPU class."""
    from loggetta.planner import usable_capacity
    rec = {"run_id": "oomed", "status": "OOM", "model": {"model": "x"}, "workload": {"kind": "serve"}, "setup": {},
           "hardware": {"gpu": {"name": "Test GPU"}}, "measured": {"device_peak_bytes": 10 * GiB}, "error": _OOM}
    cap = usable_capacity(gpu(), [rec])
    assert cap == (int(11.20 * GiB), "oomed") and usable_capacity(gpu(), []) is None
    p = plan(topo, hw(), Workload(seq_len=512), observations=[rec])
    assert p.budget["device"] == int(11.20 * GiB) and "receipt oomed" in p.budget["device_source"]


def test_an_out_of_memory_receipt_is_a_lower_bound_the_plan_respects(topo):
    """The allocator peak at the failure plus the allocation that failed is a floor on that setup's need; where today's
    estimate is below it, the gap is charged as a residual."""
    first = _serve(topo, 4096, 1, cap=(12, 0))
    setup = first.selected.setup
    est = sum(ln.bytes for ln in first.selected.lines if ln.where == "device"
              and not ln.name.startswith(("allocator reserve", "CUDA context", "allocator residual")))
    rec = {"run_id": "oomed", "status": "OOM", "model": {"model": topo.model}, "workload": {"kind": "serve"},
           "setup": {**setup, "buckets": list(setup["buckets"])}, "hardware": {"gpu": {"name": "Other GPU"}},
           "measured": {"device_peak_bytes": est + (300 << 20)}, "error": _OOM}
    p = plan(topo, hw(cap=(12, 0)), Workload(kind="serve", context_len=4096, concurrency=1),
             Constraints(fixed={"graphs": setup["graphs"]}), observations=[rec])
    res = next(ln for ln in p.selected.lines if ln.name.startswith("allocator residual"))
    assert abs(res.bytes - ((300 << 20) + (32 << 20))) < (1 << 20) and "receipt oomed" in res.detail


def test_a_data_profile_resolves_epochs_and_travels_with_the_plan(topo):
    from loggetta.data import TrainingData

    data = TrainingData("train.jsonl").to_dict()
    profile = {"schema": "data-profile/1", "rows": 200, "tokens": 51_200, "histogram": [[192, 256, 200]],
               "options": data, "format": "text", "split": "train", "source": {"kind": "local", "sha256": "ab" * 32},
               "lengths": {"min": 256, "p50": 256, "p90": 256, "p99": 256, "max": 256, "mean": 256.0}}
    p = plan(topo, hw(), Workload(seq_len=512, data=data, epochs=1), Constraints(), data_profile=profile)
    assert p.status == "feasible" and p.workload.steps == 100 and p.workload.epochs == 1
    assert any(r.startswith("steps 100: 1 epoch(s) of 51,200 tokens") for r in p.reasons)
    assert "200 examples, 51,200 tokens" in p.render()
    assert ExecutionPlan.from_dict(json.loads(p.to_json())).to_json() == p.to_json()
    assert p.to_json() == plan(topo, hw(), Workload(seq_len=512, data=data, epochs=1), Constraints(),
                               data_profile=profile).to_json()


def test_isolated_packing_is_priced_in_the_plan(topo):
    from loggetta.data import TrainingData

    data = TrainingData("chat.jsonl", format="chat").to_dict()
    profile = {"schema": "data-profile/1", "rows": 300, "tokens": 90_000, "histogram": [[256, 320, 300]],
               "options": data, "format": "chat", "split": "train", "source": {"kind": "local", "sha256": "cd" * 32},
               "lengths": {"min": 300, "p50": 300, "p90": 300, "p99": 300, "max": 300, "mean": 300.0},
               "loss_mode": "assistant", "loss_tokens": 30_000, "loss": "assistant tokens only", "packing_mode": "isolated",
               "packed": {"seq_len": 1024, "rows": 100, "efficiency": 0.88, "truncated_examples": 0, "truncated_tokens": 0,
                          "truncated_loss_tokens": 0, "window": 64}}
    p = plan(topo, hw(), Workload(seq_len=1024, data=data, epochs=1), Constraints(), data_profile=profile)
    assert p.status == "feasible" and p.workload.steps == 100
    mask = next(ln for ln in p.selected.lines if ln.name == "packed-example attention masks")
    assert mask.bytes == 1024 * 1024 * 3 and mask.where == "device"
    assert any(r.startswith("isolated packing: 100 rows of 1,024 tokens, 88% filled") for r in p.reasons)
    assert "isolated packing" in p.render()
