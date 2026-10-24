"""The planner's held-out band (loggetta/bands.py, loggetta/planner_bands.json, bench/build_planner_bands.py): the packaged
bands are the split's held-out rows and recompute from their own bytes; a band needs 2 rows from 2 distinct machines; the
rendered text reads as an observation, never a prediction, and names the miss; the build refuses a receipt that is not the
split's; and the band never changes the plan."""
import dataclasses
import hashlib
import importlib.util
import json
import pathlib
import re

import pytest

from loggetta import bands as B
from loggetta.hardware import GPU, Fact, HardwareProfile, Host

ROOT = pathlib.Path(__file__).resolve().parents[1]
BANDS = json.loads((ROOT / "loggetta" / "planner_bands.json").read_text())
SPLIT = json.loads((ROOT / "evidence" / "planner-split" / BANDS["split"]).read_text())
PLACEMENT = {"resident (device)": "device", "host offload (experts on host)": "host"}
GiB, MiB = 1 << 30, 1 << 20
NEVER = re.compile(r"\b(expected|typical|predicted|prediction)\b", re.I)


def _builder():
    spec = importlib.util.spec_from_file_location("build_planner_bands", ROOT / "bench" / "build_planner_bands.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------------------------------- the packaged bands


def test_the_bands_are_the_splits_cohorts_and_rows():
    assert BANDS["split_version"] == SPLIT["version"] and BANDS["pinned_at"] == SPLIT["pinned_at"]
    assert set(BANDS["cohorts"]) == set(SPLIT["cohorts"])
    for name, c in BANDS["cohorts"].items():
        s = SPLIT["cohorts"][name]
        assert (c["held_out_rows"], c["boxes"], c["distinct_machines"]) == (s["held_out_rows"], s["boxes"], s["distinct_machines"])
        rows = [r for r in SPLIT["held_out"] if r["cohort"] == name]
        assert {(r["model"], r["card"], PLACEMENT[r["placement"]]) for r in rows} == {(c["model"], c["card"], c["placement"])}
        if c["band"] is not None:
            assert [(r["id"], r["sha256"], r["file"]) for r in c["rows"]] == [(r["id"], r["sha256"], r["file"]) for r in rows]


def test_a_band_needs_two_rows_from_two_distinct_machines():
    for name, c in BANDS["cohorts"].items():
        eligible = c["held_out_rows"] >= 2 and (c["boxes"] or 0) >= 2 and c["distinct_machines"] is True
        assert (c["band"] is not None) == eligible, name
        if not eligible:
            assert c["why"] == ("unmeasured" if c["held_out_rows"] < 2 else "one box"), name
    assert sum(c["band"] is not None for c in BANDS["cohorts"].values()) == 4


def test_every_band_recomputes_from_its_own_bytes():
    for name, c in BANDS["cohorts"].items():
        if c["band"] is None:
            continue
        for t, v in c["band"].items():
            ratios = []
            for r in c["rows"]:
                term = r["terms"][t]
                assert term["ratio"] == term["measured"] / term["estimated"], (name, r["id"], t)
                ratios.append(term["ratio"])
            assert (v["min"], v["max"], v["rows_over"]) == (min(ratios), max(ratios), sum(x > 1 for x in ratios)), (name, t)
        assert ("host" in c["band"]) == (c["placement"] == "host"), name


def test_the_host_offload_band_carries_the_miss():
    c = BANDS["cohorts"]["ERNIE-4.5-21B-A3B-PT / host offload / RTX A4000"]
    assert c["band"]["host"]["rows_over"] == 3 and round(c["band"]["host"]["max"], 4) == 1.0548
    assert all(c["band"][t]["rows_over"] == 0 for t in ("device_driver", "device_allocator"))


# ---------------------------------------------------------------------------------------------- the rendered text


CASES = [("baidu/ERNIE-4.5-21B-A3B-PT", "NVIDIA RTX A4000", "host"), ("LiquidAI/LFM2-8B-A1B", "NVIDIA RTX A4000", "device"),
         ("baidu/ERNIE-4.5-21B-A3B-PT", "NVIDIA GeForce RTX 5090", "device"), ("LiquidAI/LFM2-8B-A1B", "NVIDIA GeForce RTX 5090", "device"),
         ("LiquidAI/LFM2-8B-A1B", "NVIDIA GeForce RTX 4090", "device"), ("Qwen/Qwen3-14B", "NVIDIA GeForce RTX 5090", "device"),
         ("Qwen/Qwen3-30B-A3B", "NVIDIA GeForce RTX 5090", "device"), ("baidu/ERNIE-4.5-21B-A3B-PT", "NVIDIA RTX A4000", "device")]


@pytest.mark.parametrize("model, gpu, residency", CASES)
def test_the_text_is_an_observation_never_a_prediction(model, gpu, residency):
    b = B.lookup(model, gpu, residency)
    text = "\n".join(B.render(b) + B.warnings(b))
    assert not NEVER.search(text), text
    if b["status"] == "band":
        n, m = b["held_out_rows"], b["machines"]
        assert f"observed on {n} held-out rows from {m} machines" in text and n >= 2 and m >= 2
        for w in B.warnings(b):
            assert re.search(rf"ran LOW on \d+ of {n} held-out rows", w) and "a further machine may fall outside" in w


def test_the_host_offload_plan_shows_its_miss():
    b = B.lookup("baidu/ERNIE-4.5-21B-A3B-PT", "NVIDIA RTX A4000", "host")
    (line,) = B.render(b)
    assert "observed on 4 held-out rows from 2 machines" in line and "host 0.996-1.055" in line
    (w,) = B.warnings(b)
    assert "host (host peak / host estimate) estimate ran LOW on 3 of 4 held-out rows" in w and "808 MiB over" in w


def test_a_one_box_cohort_prints_no_band_by_name():
    b = B.lookup("LiquidAI/LFM2-8B-A1B", "NVIDIA GeForce RTX 4090", "device")
    assert b["status"] == "one box" and B.warnings(b) == []
    assert B.render(b) == [f"  {'held-out band':44s} held-out rows from one box only: no band (2 rows, {BANDS['split']})"]


@pytest.mark.parametrize("model, gpu, residency", [
    ("Qwen/Qwen3-30B-A3B", "NVIDIA GeForce RTX 5090", "device"),       # a model no row measured
    ("baidu/ERNIE-4.5-21B-A3B-PT", "NVIDIA RTX A4000", "device"),       # the right model and card, the other placement
    ("baidu/ERNIE-4.5-21B-A3B-PT", "NVIDIA RTX A6000", "host"),         # the right model and placement, another card
    ("baidu/ERNIE-4.5-21B-A3B-PT", None, "host"),                        # no GPU
    ("baidu/ERNIE-4.5-21B-A3B", "NVIDIA RTX A4000", "host"),             # a sibling repository id: no generalizing
])
def test_anything_else_is_unmeasured(model, gpu, residency):
    b = B.lookup(model, gpu, residency)
    assert b["status"] == "unmeasured" and B.warnings(b) == [] and B.render(b) == [f"  {'held-out band':44s} {B.UNMEASURED}"]


def test_card_names_normalize_to_the_splits():
    assert B.card_name("NVIDIA GeForce RTX 5090") == "RTX 5090" and B.card_name("NVIDIA RTX A4000") == "RTX A4000"
    assert B.card_name("RTX 4090") == "RTX 4090" and B.card_name(None) is None


# ------------------------------------------------------------------------------------------------------ the build


def _receipt(model="m/M", residency="device", drv=(100, 80), alloc=(90, 85), host=None):
    c = {"device_driver": {"estimated": drv[0], "measured": drv[1]}, "device_allocator": {"estimated": alloc[0], "measured": alloc[1]}}
    if host:
        c["host"] = {"estimated": host[0], "measured": host[1]}
    return json.dumps({"model": {"model": model}, "setup": {"expert_residency": residency}, "comparison": c}).encode()


def _split(files, boxes=("b1", "b2"), distinct=True, placement="resident (device)"):
    rows = [{"id": f"{bx}/m/x/{i}", "box": bx, "repo": "r", "file": f, "sha256": hashlib.sha256(raw).hexdigest(), "model": "m/M",
             "card": "RTX X", "placement": placement, "seq": 1024, "cohort": "C"} for i, ((f, raw), bx) in enumerate(zip(files, boxes))]
    return {"version": 9, "pinned_at": {"r": "c0ffee"}, "held_out": rows,
            "cohorts": {"C": {"held_out_rows": len(rows), "boxes": len(set(boxes)), "distinct_machines": distinct}}}


def test_the_build_reads_at_the_pinned_commit_and_summarizes():
    files = [("a.json", _receipt(drv=(100, 80))), ("b.json", _receipt(drv=(100, 90)))]
    seen = []
    store = dict(files)
    doc = _builder().build(_split(files), {"r": lambda c, p: (seen.append(c), store[p])[1]})
    assert set(seen) == {"c0ffee"}
    band = doc["cohorts"]["C"]["band"]
    assert band["device_driver"] == {"min": 0.8, "max": 0.9, "rows_over": 0} and "host" not in band


def test_the_build_refuses_a_receipt_that_is_not_the_splits():
    files = [("a.json", _receipt()), ("b.json", _receipt())]
    split = _split(files)
    mod = _builder()
    with pytest.raises(mod.Refused, match="not the split's sha256"):
        mod.build(split, {"r": lambda c, p: dict(files)[p] + b" "})


@pytest.mark.parametrize("bad, why", [
    (_receipt(residency="host", host=(10, 9)), "the receipt records"),          # the receipt's placement is not the split's
    (_receipt(model="m/Other"), "the receipt records"),
    (json.dumps({"model": {"model": "m/M"}, "setup": {"expert_residency": "device"}, "comparison": {}}).encode(), "no device_driver"),
])
def test_the_build_refuses_a_receipt_it_cannot_read(bad, why):
    files = [("a.json", _receipt()), ("b.json", bad)]
    mod = _builder()
    with pytest.raises(mod.Refused, match=why):
        mod.build(_split(files), {"r": lambda c, p: dict(files)[p]})


def test_the_build_gives_one_box_no_band():
    files = [("a.json", _receipt()), ("b.json", _receipt())]
    doc = _builder().build(_split(files, boxes=("b1", "b1")), {"r": lambda c, p: dict(files)[p]})
    assert doc["cohorts"]["C"]["band"] is None and doc["cohorts"]["C"]["why"] == "one box"


# ------------------------------------------------------------------------------------- the band changes nothing


@pytest.fixture(scope="module")
def topo():
    tr = pytest.importorskip("transformers")
    pytest.importorskip("experts4bit_qlora.recipe")
    from loggetta import describe_model

    cfg = tr.Qwen3MoeConfig(hidden_size=1024, intermediate_size=2048, moe_intermediate_size=768, num_experts=128,
                            num_experts_per_tok=4, num_hidden_layers=8, num_attention_heads=8, num_key_value_heads=4,
                            head_dim=128, vocab_size=32000, max_position_embeddings=4096, decoder_sparse_step=1)
    return describe_model(cfg)


def _hw():
    r = lambda v: Fact(v, "reported")  # noqa: E731
    g = GPU(index=0, vendor="nvidia", name="NVIDIA RTX A4000", uuid=None, compute_capability=r((8, 6)),
            memory_total=r(16 * GiB), memory_free=r(16 * GiB), driver=r("575.64.05"), pcie_gen_max=r(4),
            pcie_gen_current=r(4), pcie_width_max=r(16), pcie_width_current=r(16))
    h = Host(cpu_model=r("test cpu"), cpus=r(8), memory_total=r(64 * GiB), memory_available=r(60 * GiB), memory_limit=r(None))
    return HardwareProfile(gpus=(g,), host=h, platform="Linux x86_64")


def test_the_band_never_changes_the_plan(topo, monkeypatch):
    from loggetta import Workload, plan

    monkeypatch.setattr(B, "table", lambda: {**BANDS, "cohorts": {}})
    bare = plan(topo, _hw(), Workload(seq_len=512))
    assert bare.status == "feasible" and bare.band["status"] == "unmeasured"
    model, residency = bare.model.get("model"), bare.selected.setup.get("expert_residency")
    fake = json.loads(json.dumps(BANDS["cohorts"]["ERNIE-4.5-21B-A3B-PT / host offload / RTX A4000"]))
    fake.update(model=model, placement=residency)
    monkeypatch.setattr(B, "table", lambda: {**BANDS, "cohorts": {"X": fake}})
    banded = plan(topo, _hw(), Workload(seq_len=512))
    assert banded.band["status"] == "band" and any("ran LOW on 3 of 4" in w for w in banded.warnings)
    strip = lambda p: {k: v for k, v in p.to_dict().items() if k not in ("band", "warnings")}  # noqa: E731
    assert strip(banded) == strip(bare)                                       # selection, bytes, budget: unchanged
    assert [w for w in banded.warnings if "held-out rows" not in w] == list(bare.warnings)
    again = type(banded).from_dict(json.loads(banded.to_json()))
    assert again.to_json() == banded.to_json() and again.band == banded.band   # a saved plan keeps what the user saw
    assert "observed on 4 held-out rows from 2 machines" in banded.render()


def test_a_plan_without_a_band_round_trips_without_one(topo):
    from loggetta import Workload, plan

    old = dataclasses.replace(plan(topo, _hw(), Workload(seq_len=512)), band=None)   # a plan saved before L2
    assert "band" not in old.to_dict()
    assert type(old).from_dict(json.loads(old.to_json())).to_json() == old.to_json()
    assert "held-out band" not in old.render()
