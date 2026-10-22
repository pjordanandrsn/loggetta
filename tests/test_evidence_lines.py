"""What a reader of the plan's total is told about its evidence: which overhead terms rest on receipts and which are
defaults, the streamed admission margin beside a streamed total (#90), and how a cited receipt is named (its counted
loss branch, a SAVED_V1 status). With no receipts on file, a plan cites no run id at all."""
import re

import pytest

from loggetta import planner
from loggetta.plan import MemoryLine

GiB = 1 << 30


class Sel:
    def __init__(self, lines, device_bytes=3 * GiB):
        self.lines, self.device_bytes = tuple(lines), device_bytes


def line(name, basis, where="device"):
    return MemoryLine(name, where, GiB // 2, basis, "detail")


def test_the_evidence_line_names_measured_terms_and_defaults():
    sel = Sel([line("CUDA context + library workspaces", "measured"),
               line("allocator reserve (cached, unallocated blocks)", "inferred"),
               line("process baseline (torch, CUDA, libraries, model objects)", "measured", "host")])
    assert planner.evidence_reasons(sel, 0) == [
        "estimate evidence: CUDA context, host baseline measured from receipts on file; allocator reserve a default "
        "(each line above names its receipt or default)"]


def test_with_no_receipt_every_overhead_term_says_it_is_a_default():
    sel = Sel([line("CUDA context + library workspaces (DQ10)", "inferred"),
               line("allocator reserve (DQ10 training peak slack)", "policy")])
    (text,) = planner.evidence_reasons(sel, 0)
    assert text.startswith("estimate evidence: no overhead term measured on this machine yet; CUDA context, "
                           "allocator reserve defaults")


def test_a_streamed_plan_states_its_admission_margin_beside_the_total():
    sel = Sel([], device_bytes=10 * GiB)
    (text,) = planner.evidence_reasons(sel, int(2.4e9))
    assert text.startswith("streamed: 10.00 GiB plan total, and a 2.24 GiB admission margin kept free above it "
                           "(the larger of 2.24 GiB, i.e. 2.4 GB, and 20% of the plan)")
    assert "the total alone is not what the process will use" in text
    assert planner.evidence_link(planner.INSAMPLE_EVIDENCE) in text


@pytest.mark.parametrize("version, ref", [("0.6.3", "v0.6.3"), ("1.0.0", "v1.0.0"),
                                          ("0.6.3.dev2", "main"), ("0.6.3+local.abc", "main"),
                                          ("0.6.3.dev0+g1a2b3c", "main"), ("0.7.0rc1", "main"), ("0.7.0b2", "main")])
def test_the_link_pins_a_release_tag_and_a_dev_build_links_main(monkeypatch, version, ref):
    import importlib.metadata as md

    class Meta(dict):
        def get_all(self, key):
            return ["Source, https://example.test/org/pkg"] if key == "Project-URL" else None

    monkeypatch.setattr(md, "metadata", lambda name: Meta(Version=version))
    assert planner.evidence_link("evidence/x") == f"https://example.test/org/pkg/tree/{ref}/evidence/x"


def test_the_evidence_link_opens_for_a_pip_user():
    """From the installed metadata's Source URL, so the link works outside a checkout and the code names no repository."""
    from importlib.metadata import metadata

    meta = metadata("loggetta")
    source = next(u.split(",", 1)[1].strip() for u in meta.get_all("Project-URL") if u.split(",", 1)[0].strip() == "Source")
    assert planner.evidence_link("evidence/x") == f"{source}/tree/v{meta['Version']}/evidence/x"   # a tag, never main
    assert planner.evidence_reasons(Sel([]), 0) == []          # resident: no margin line, nothing invented


@pytest.mark.parametrize("obs, text", [
    ({"run_id": "a"}, "receipt a"),
    ({"run_id": "b", "status": "SAVED_V1"}, "receipt b (SAVED_V1: its adapter was kept as backend-adapter/1)"),
    ({"run_id": "c", "engaged": {"chunked_lm_loss": {"branch": "chunked"}}}, "receipt c (chunked LM loss, counted)"),
    ({"run_id": "d", "status": "SAVED_V1", "engaged": {"chunked_lm_loss": {"branch": "mixed"}}},
     "receipt d (mixed LM loss, counted; SAVED_V1: its adapter was kept as backend-adapter/1)"),
    ({"run_id": "e", "engaged": {"chunked_lm_loss": {"branch": "not counted: no training forward reached the patch"}}},
     "receipt e"),                                              # nothing counted, nothing claimed
])
def test_a_cited_receipt_names_its_counted_branch_and_a_saved_v1_status(obs, text):
    assert planner.cite(obs) == text


def test_a_saved_v1_receipt_is_named_where_the_planner_reads_its_overheads():
    gpu = type("G", (), {"name": "Test GPU", "driver": type("F", (), {"value": "575.64.05"})()})()
    obs = {"run_id": "r1", "status": "SAVED_V1", "hardware": {"gpu": {"name": "Test GPU", "driver": "575.64.05"}},
           "measured": {"cuda_context_bytes": 600 << 20, "device_peak_bytes": 4 * GiB,
                        "device_reserved_peak_bytes": 5 * GiB}}
    dev, host, frac, meta = planner._overheads(None, gpu, [obs])
    assert dev[0].basis == "measured" and "receipt r1 (SAVED_V1: its adapter was kept as backend-adapter/1)" in dev[0].detail
    assert meta[0] == "measured" and "SAVED_V1" in meta[1]


def test_with_no_receipts_a_plan_cites_no_run_id():
    """Every run id a plan cites comes from the receipts it was given; with none, it names no receipt at all."""
    tr = pytest.importorskip("transformers")
    pytest.importorskip("experts4bit_qlora.recipe")
    from test_planner import hw

    from loggetta import Workload, describe_model, plan

    cfg = tr.Qwen3MoeConfig(hidden_size=1024, intermediate_size=2048, moe_intermediate_size=768, num_experts=128,
                            num_experts_per_tok=4, num_hidden_layers=8, num_attention_heads=8, num_key_value_heads=4,
                            head_dim=128, vocab_size=32000, max_position_embeddings=4096, decoder_sparse_step=1)
    p = plan(describe_model(cfg), hw(), Workload(seq_len=512))
    text = p.render(verbose=True)
    assert re.findall(r"receipt (?!on |or |for )\S+", text) == []
    assert any(r.startswith("estimate evidence: no overhead term measured on this machine yet") for r in p.reasons)


def test_the_linked_evidence_is_in_the_tree_the_release_tag_will_hold():
    """The link points at this release's tag; the folder must be in the tree that tag is cut from."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    assert (root / planner.INSAMPLE_EVIDENCE).is_dir()
