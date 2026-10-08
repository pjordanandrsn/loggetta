"""The single-stream serve objective (concurrency 1): which B=1 levers a default server runs for this model family, as
the INSTALLED experts4bit-qlora resolves them, and a decode figure quoted only from a receipt of exactly this setup.

loggetta does not predict throughput. A figure is shown only when a serve receipt matches model, GPU name and driver,
setup and the experts4bit-qlora / grouped-nf4-gemm versions exactly; otherwise nothing is shown and the planner's own
statement stands.
"""
import types

import pytest

from loggetta.measure import decode_step_b1

GiB = 1 << 30


# ------------------------------------------------------------------ the step trace reader

def _row(step_ms, decode_rows=1, bucket=1, prefill_tokens=0, gpu=None):
    r = {"step_ms": step_ms, "decode_rows": decode_rows, "bucket": bucket, "seg": {}}
    if prefill_tokens:
        r["prefill_tokens"] = prefill_tokens
    if gpu is not None:
        r["gpu"] = gpu
    return r


def test_decode_step_b1_reads_one_row_decode_steps_only():
    rows = [_row(40.0, decode_rows=0, prefill_tokens=512),         # the prompt's prefill
            _row(9.0, prefill_tokens=16),                          # a step that also prefilled: not a decode step
            _row(4.6, gpu={"dec_prep": 0.1, "dec_issue": 4.3}),
            _row(4.8, gpu={"dec_prep": 0.1, "dec_issue": 4.5}),
            _row(4.7),                                            # its events not resolved: host time only
            _row(7.0, decode_rows=2, bucket=2)]                   # two rows: not single-stream
    b1 = decode_step_b1(rows)
    assert b1["decode_steps_b1"] == 3 and b1["decode_step_ms_b1"] == 4.7
    assert b1["decode_device_ms_b1"] == pytest.approx(4.3)        # median of 4.2 and 4.4
    assert b1["decode_buckets_b1"] == [1]


def test_decode_step_b1_is_none_without_a_single_stream_step():
    assert decode_step_b1([_row(40.0, decode_rows=0, prefill_tokens=512), _row(7.0, decode_rows=2)]) is None
    assert decode_step_b1([]) is None


def test_eager_decode_steps_count_with_no_bucket():
    b1 = decode_step_b1([_row(9.1, bucket=None), _row(9.3, bucket=None)])
    assert b1["decode_step_ms_b1"] == pytest.approx(9.2) and b1["decode_buckets_b1"] == [None]


# ------------------------------------------------------------------ the levers, read from experts4bit-qlora

e4b = pytest.importorskip("loggetta.backends.experts4bit")


def _serve_paged():
    return pytest.importorskip("experts4bit_qlora.serve_paged")


def _family_default(monkeypatch, sp):
    """experts4bit-qlora's family-scoped default (lane P115, e4b#1361) on whatever version is installed: the four
    knobs, the unset marker, the per-family resolution and the reads it names."""
    knobs = ("E4B_PAGED_FUSE_QKV", "E4B_FUSE_T1_GLUE", "E4B_FUSE_T1_GLUE_R2", "E4B_FUSE_ROUTER_EPI")
    allow = {"qwen3_moe": "P115 Phase D: SANE at T == 1 (#1379)"}
    unlicensed = {"gpt_oss": "P115 Phase C: SANE argmax agreement 0.924 < 0.95 (#1342)"}

    def resolve(modes, model_type):
        on = model_type in allow
        return ({k: ("auto" if on else "0") if v == "default" else v for k, v in modes.items()},
                {k: ("default-allowlisted" if on else "default-off") for k in modes})
    for name, val in (("FUSION_KNOBS", knobs), ("FUSION_UNSET", "default"), ("resolve_fusion_modes", resolve),
                      ("FUSION_DEFAULT_FAMILIES", allow), ("FUSION_UNLICENSED", unlicensed)):
        monkeypatch.setattr(sp, name, val, raising=False)


def test_without_a_family_scoped_default_the_stack_is_off_unless_set(monkeypatch):
    sp = _serve_paged()
    monkeypatch.delattr(sp, "resolve_fusion_modes", raising=False)
    line = e4b.single_stream_levers(types.SimpleNamespace(model_type="qwen3_moe"))
    assert "off unless set" in line and "no family-scoped fusion default" in line


def test_an_allowlisted_family_runs_the_stack_by_default_and_names_its_read(monkeypatch):
    sp = _serve_paged()
    _family_default(monkeypatch, sp)
    line = e4b.single_stream_levers(types.SimpleNamespace(model_type="qwen3_moe"))
    assert "on by default for qwen3_moe" in line and "#1379" in line and "E4B_PAGED_FUSE_QKV=0" in line


def test_any_other_family_is_off_by_default_and_says_which_read_it_lacks(monkeypatch):
    sp = _serve_paged()
    _family_default(monkeypatch, sp)
    gptoss = e4b.single_stream_levers(types.SimpleNamespace(model_type="gpt_oss"))
    assert "off by default for gpt_oss" in gptoss and "0.924" in gptoss
    other = e4b.single_stream_levers(types.SimpleNamespace(model_type="olmoe"))
    assert "off by default for olmoe" in other and "no registered read" in other


# ------------------------------------------------------------------ a figure only from a receipt of this same setup

SETUP = {"placement": "all-vram", "max_seqs": 1, "max_tokens_per_seq": 4096, "graphs": True, "buckets": [1]}
VERS = {"experts4bit-qlora": "0.50.0", "grouped-nf4-gemm": "0.43.0"}


def _gpu(name="RTX 5090", driver="595.71.05"):
    return types.SimpleNamespace(name=name, driver=types.SimpleNamespace(value=driver))


def _receipt(**over):
    rec = {"run_id": "serve-q3-c4096x1", "status": "OK", "model": {"model": "Qwen/Qwen3-30B-A3B"},
           "workload": {"kind": "serve", "context_len": 4096, "concurrency": 1, "prompt_tokens": 1024,
                        "new_tokens": 64},
           "setup": dict(SETUP), "hardware": {"gpu": {"name": "RTX 5090", "driver": "595.71.05"}},
           "provenance": {"versions": dict(VERS)},
           "measured": {"decode_step_ms_b1": 4.64, "decode_device_ms_b1": 4.31, "decode_steps_b1": 63}}
    for k, v in over.items():
        rec[k] = v
    return rec


def _perf(observations, setup=SETUP, concurrency=1, kind="serve", gpu=None, versions=VERS):
    topo = types.SimpleNamespace(model="Qwen/Qwen3-30B-A3B")
    wl = types.SimpleNamespace(kind=kind, concurrency=concurrency)
    return e4b.performance(topo, dict(setup), wl, gpu or _gpu(), observations,
                           types.SimpleNamespace(versions=dict(versions)))


def test_a_receipt_of_exactly_this_setup_is_quoted_as_measured():
    p = _perf([_receipt()])
    assert p["estimate"]["basis"] == "measured-same-setup" and p["estimate"]["decode_step_ms_b1"] == 4.64
    assert p["estimate"]["tokens_per_s_b1"] == pytest.approx(215.5, abs=0.1)
    assert p["estimate"]["receipt"] == "serve-q3-c4096x1"
    assert "measured on this setup" in p["statement"] and "serve-q3-c4096x1" in p["statement"]
    assert "after a 1024-token prompt over 64 new tokens" in p["statement"]     # where in the KV it was measured
    assert (p["estimate"]["prompt_tokens"], p["estimate"]["new_tokens"]) == (1024, 64)


def test_a_receipt_without_its_lengths_is_still_quoted_without_them():
    rec = _receipt(workload={"kind": "serve", "context_len": 4096, "concurrency": 1})
    p = _perf([rec])
    assert p["estimate"]["basis"] == "measured-same-setup" and "token prompt" not in p["statement"]
    assert p["estimate"]["prompt_tokens"] is None


@pytest.mark.parametrize("change", [
    {"hardware": {"gpu": {"name": "RTX 4090", "driver": "595.71.05"}}},       # another GPU
    {"hardware": {"gpu": {"name": "RTX 5090", "driver": "575.64.05"}}},       # another driver
    {"provenance": {"versions": {**VERS, "experts4bit-qlora": "0.49.0"}}},    # another e4b: other defaults
    {"provenance": {"versions": {**VERS, "grouped-nf4-gemm": "0.42.0"}}},     # another kernel release
    {"setup": {**SETUP, "graphs": False}},                                   # another setup
    {"model": {"model": "Qwen/Qwen3.6-35B-A3B"}},                            # another model
    {"workload": {"kind": "serve", "context_len": 4096, "concurrency": 4}},  # not one sequence
    {"status": "REFUSED"},
    {"measured": {"tokens_per_s": 210.0}},                                   # no decode step on record
])
def test_anything_short_of_the_same_setup_shows_nothing(change):
    assert _perf([_receipt(**change)]) is None


def test_no_figure_for_other_workloads():
    assert _perf([_receipt()], concurrency=4, setup={**SETUP, "max_seqs": 4}) is None
    assert _perf([_receipt()], kind="train") is None
    assert _perf([]) is None


# ------------------------------------------------------------------ through the planner

@pytest.fixture(scope="module")
def topo():
    tr = pytest.importorskip("transformers")           # here, not at module level: the tests above need neither
    pytest.importorskip("experts4bit_qlora.arch.topology")
    from loggetta import describe_model
    cfg = tr.Qwen3MoeConfig(hidden_size=1024, intermediate_size=2048, moe_intermediate_size=768, num_experts=128,
                            num_experts_per_tok=4, num_hidden_layers=8, num_attention_heads=8, num_key_value_heads=4,
                            head_dim=128, vocab_size=32000, max_position_embeddings=4096, decoder_sparse_step=1)
    return describe_model(cfg)


def _hw():
    from loggetta.hardware import GPU, Fact, HardwareProfile, Host
    r = lambda v: Fact(v, "reported")  # noqa: E731
    g = GPU(index=0, vendor="nvidia", name="Test GPU", uuid=None, compute_capability=r((8, 6)),
            memory_total=r(12 * GiB), memory_free=r(12 * GiB), driver=r("575.64.05"),
            pcie_gen_max=r(4), pcie_width_max=r(16), pcie_gen_current=r(4), pcie_width_current=r(16))
    host = Host(cpu_model=r("test cpu"), cpus=r(8), memory_total=r(64 * GiB), memory_available=r(40 * GiB),
                memory_limit=r(64 * GiB))
    return HardwareProfile(gpus=(g,), host=host, platform="Linux x86_64")


def _serve_plan(topo, observations=()):
    from loggetta import Workload, plan
    pytest.importorskip("experts4bit_qlora.serve_recipe")
    pytest.importorskip("fp8_paged_attn", reason="needs grouped-nf4-gemm")
    return plan(topo, _hw(), Workload(kind="serve", context_len=4096, concurrency=1), observations=list(observations))


def test_a_single_stream_plan_states_its_levers_and_quotes_only_a_matching_receipt(topo):
    p = _serve_plan(topo)
    assert p.status == "feasible" and p.selected.setup["max_seqs"] == 1
    assert any(r.startswith("single stream: the B=1 fused stack") for r in p.reasons)
    assert p.performance["estimate"] is None and p.performance["statement"].startswith("not predicted")
    rec = {"run_id": "serve-tiny-c4096x1", "status": "OK", "model": {"model": topo.model},
           "workload": {"kind": "serve", "context_len": 4096, "concurrency": 1}, "setup": dict(p.selected.setup),
           "hardware": {"gpu": {"name": "Test GPU", "driver": "575.64.05"}},
           "provenance": {"versions": dict(p.provenance["backend_versions"])},
           "measured": {"decode_step_ms_b1": 5.0, "decode_steps_b1": 63}}
    q = _serve_plan(topo, [rec])
    assert q.performance["estimate"]["basis"] == "measured-same-setup"
    assert q.performance["estimate"]["tokens_per_s_b1"] == 200.0 and "serve-tiny-c4096x1" in q.performance["statement"]
    other = _serve_plan(topo, [{**rec, "provenance": {"versions": {"experts4bit-qlora": "0.0.0"}}}])
    assert other.performance["estimate"] is None
