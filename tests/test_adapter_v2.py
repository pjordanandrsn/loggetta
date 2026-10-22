"""backend-adapter/2: the writer records what the base was quantized to, and the reader checks it (PA-2).

Two halves. The first runs on every installed experts4bit-qlora: a stand-in ``adapter_schema`` pins what the writer
takes from where (the report, the residency, the scopes) and what it refuses before anything is written. The second
needs a release that ships ``adapter_schema`` (it skips before then): real NF4 expert stacks on the CPU, the real
fingerprint, and experts4bit-qlora's own verifier reading what Loggetta wrote.
"""
import json
import re
import sys
import types
from dataclasses import replace

import pytest
import torch
from torch import nn

from experts4bit_qlora.lora import ExpertsLoRA, LoRALinear, trainable_lora_param_ids
from loggetta.backends import experts4bit_adapters as A
from test_execution import a_plan
from test_training_data import Tokenizer

FACTS = {"expert_absmax": {"form": "double-quantized", "stacks": 1, "record": {"compressed": 1}},
         "train_fused_qkv": 0, "attn_quant": {"form": "bf16", "quant_type": None, "blocksize": None, "nested": None},
         "versions": {"bitsandbytes": "0.50.2", "torch": "2.14.1", "grouped-nf4-gemm": None, "transformers": "5.20.2",
                      "experts4bit-qlora": "0.56.1"}}
V2_KEYS = ("expert_absmax", "train_fused_qkv", "attn_quant", "versions", "lora_scale_applied", "fingerprints",
           "run_id", "created_at")


def plan_with(**setup):
    p = a_plan()
    # r and alpha as every real plan records them (QLoRASetup's fields), matching the LoRA modules built below: the
    # reader checks the applied scale against alpha/r
    return replace(p, selected=replace(p.selected, setup={**p.selected.setup, "r": 2, "alpha": 4, **setup}))


def trainable(model):
    expert, attention = trainable_lora_param_ids(model)
    for p in model.parameters():
        p.requires_grad_(id(p) in expert | attention)
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.add_(0.1)
    return model


class Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = LoRALinear(nn.Linear(16, 16, bias=False), r=2, alpha=4, dtype=torch.float32)
        self.o_proj = LoRALinear(nn.Linear(16, 16, bias=False), r=2, alpha=4, dtype=torch.float32)


def bf16_stack_model(attention=True):
    """A model the stand-in schema can fingerprint: an ExpertsLoRA (bf16 stack) and, optionally, attention LoRA."""
    from experts4bit_qlora import ExpertsNbit

    m = nn.Module()
    m.experts = ExpertsLoRA(ExpertsNbit.from_float(torch.randn(2, 32, 8), torch.randn(2, 8, 16), quant_type="bf16"),
                            r=2, alpha=4, dtype=torch.float32)
    if attention:
        m.self_attn = Attn()
    return trainable(m)


class FakeSchema(types.SimpleNamespace):
    """A stand-in for experts4bit_qlora.adapter_schema: it records every fingerprint call."""

    def __init__(self, fail_source=None, extra_field=None):
        fields = dict.fromkeys(V2_KEYS, "")
        if extra_field:
            fields[extra_field] = ""
        super().__init__(V2_FIELDS=fields, KNOWN_SCHEMES=("qfp-v1",), calls=[], fail_source=fail_source)

    def quant_fingerprint(self, model, scope, *, source="resident"):
        self.calls.append((scope, source))
        if source == self.fail_source:
            raise ValueError(f"quant_fingerprint: expert stack experts is evicted by expert offload; source={source!r}")
        return {"scheme_id": "qfp-v1", "scope": scope, "source": source, "modules": 2, "sha256": "a" * 64,
                "bytes_hashed": 1, "seconds": 0.01}

    def lora_scale_applied(self, model):
        return {"attention": 2.0, "experts": 2.0}


@pytest.fixture
def fake(monkeypatch):
    schema = FakeSchema()
    monkeypatch.setattr(A, "adapter_schema", lambda: schema)
    return schema


# ------------------------------------------------------------------------------------------------- the writer


def test_v2_records_every_field_from_the_runtime_and_the_report(tmp_path, fake):
    out = A.save_adapter(bf16_stack_model(), Tokenizer(), tmp_path / "a", plan_with(expert_residency="device"),
                         data={}, report={"commit": "c0ffee", **FACTS}, seed=0, run_id="run-1")
    m = json.loads((tmp_path / "a" / A.MANIFEST).read_text())
    assert out["format"] == m["schema"] == A.SCHEMA_V2
    assert set(V2_KEYS) <= set(m)
    for k in A.TRAINING_FACTS:                                   # copied from prepare_qlora_training's report as is
        assert m[k] == FACTS[k], k
    assert m["lora_scale_applied"] == {"attention": 2.0, "experts": 2.0}
    assert m["run_id"] == "run-1" and re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", m["created_at"])
    assert sorted(m["fingerprints"]) == ["attention", "experts"]
    assert sorted(fake.calls) == [("attention", "resident"), ("experts", "resident")]
    from safetensors import safe_open
    with safe_open(str(tmp_path / "a" / A.WEIGHTS), "pt") as f:
        assert f.metadata()["format"] == A.SCHEMA_V2


@pytest.mark.parametrize("residency, source", [("device", "resident"), ("host", "offload_homes")])
def test_the_fingerprint_source_is_the_recorded_residency(tmp_path, fake, residency, source):
    A.save_adapter(bf16_stack_model(attention=False), Tokenizer(), tmp_path / "a", plan_with(expert_residency=residency),
                   data={}, report=FACTS, seed=0, run_id="r")
    assert fake.calls == [("experts", source)]                   # experts only: this adapter carries no attention LoRA
    assert sorted(json.loads((tmp_path / "a" / A.MANIFEST).read_text())["fingerprints"]) == ["experts"]


def test_a_device_run_whose_stacks_are_evicted_is_refused_and_writes_nothing(tmp_path, monkeypatch):
    schema = FakeSchema(fail_source="resident")
    monkeypatch.setattr(A, "adapter_schema", lambda: schema)
    with pytest.raises(ValueError, match="evicted"):
        A.save_adapter(bf16_stack_model(), Tokenizer(), tmp_path / "a", plan_with(expert_residency="device"),
                       data={}, report=FACTS, seed=0, run_id="r")
    assert not (tmp_path / "a").exists()


@pytest.mark.parametrize("bad, match", [
    ({"report": {k: v for k, v in FACTS.items() if k != "expert_absmax"}}, r"lacks \['expert_absmax'\]"),
    ({"residency": None}, "unrecorded residency"),
    ({"residency": "disk"}, "unrecorded residency"),
])
def test_a_fact_that_cannot_be_recorded_refuses_before_the_directory_exists(tmp_path, fake, bad, match):
    plan = plan_with(expert_residency=bad.get("residency", "device"))
    with pytest.raises(ValueError, match=match):
        A.save_adapter(bf16_stack_model(), Tokenizer(), tmp_path / "a", plan, data={}, report=bad.get("report", FACTS),
                       seed=0, run_id="r")
    assert not (tmp_path / "a").exists()
    assert fake.calls == []                                      # refused before any fingerprint was taken


@pytest.mark.parametrize("run_id", [None, ""])
def test_v2_without_a_run_id_is_refused_before_the_directory_exists(tmp_path, fake, run_id):
    """A reader checks only that run_id is PRESENT, so a null would pass every verifier: the writer refuses it."""
    with pytest.raises(ValueError, match="records the run that produced the adapter"):
        A.save_adapter(bf16_stack_model(), Tokenizer(), tmp_path / "a", plan_with(), data={}, report=FACTS, seed=0,
                       run_id=run_id)
    assert not (tmp_path / "a").exists() and fake.calls == []


def test_a_schema_field_this_writer_does_not_record_is_refused_not_omitted(tmp_path, monkeypatch):
    schema = FakeSchema(extra_field="arena_index")
    monkeypatch.setattr(A, "adapter_schema", lambda: schema)
    with pytest.raises(ValueError, match=r"does not record: \['arena_index'\]"):
        A.save_adapter(bf16_stack_model(), Tokenizer(), tmp_path / "a", plan_with(), data={}, report=FACTS, seed=0, run_id="r")
    assert not (tmp_path / "a").exists()


def test_without_the_schema_the_writer_stays_on_v1_and_v2_is_refused_by_name(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "adapter_schema", lambda: None)
    out = A.save_adapter(bf16_stack_model(), Tokenizer(), tmp_path / "a", plan_with(), data={}, report={}, seed=0)
    m = json.loads((tmp_path / "a" / A.MANIFEST).read_text())
    assert out["format"] == m["schema"] == A.SCHEMA_V1 and not set(V2_KEYS) & set(m)
    with pytest.raises(ValueError, match="predates it"):
        A.save_adapter(bf16_stack_model(), Tokenizer(), tmp_path / "b", plan_with(), data={}, report=FACTS, seed=0,
                       schema=A.SCHEMA_V2)
    assert not (tmp_path / "b").exists()
    with pytest.raises(ValueError, match="unknown adapter schema"):
        A.save_adapter(bf16_stack_model(), Tokenizer(), tmp_path / "c", plan_with(), data={}, report=FACTS, seed=0,
                       schema="backend-adapter/3")


def test_adapter_schema_reports_a_missing_module_as_none(monkeypatch):
    import experts4bit_qlora

    # an older release: no submodule. Once another test imported it, `from pkg import mod` reads the package attribute,
    # so both the attribute and the sys.modules entry go
    monkeypatch.delattr(experts4bit_qlora, "adapter_schema", raising=False)
    monkeypatch.setitem(sys.modules, "experts4bit_qlora.adapter_schema", None)       # import raises ImportError
    assert A.adapter_schema() is None


# ------------------------------------------------------------------------------------------------- the reader


def written_v2(tmp_path, fake):
    path = tmp_path / "a"
    A.save_adapter(bf16_stack_model(), Tokenizer(), path, plan_with(), data={}, report=FACTS, seed=0, run_id="r")
    return path


def test_read_manifest_hands_v2_to_the_runtimes_own_verifier(tmp_path, fake, monkeypatch):
    path = written_v2(tmp_path, fake)
    seen = []

    class Refused(ValueError):
        pass

    def verify(d):
        seen.append(str(d))
        if "refuse" in seen[-1]:
            raise Refused("adapter refused (7): fingerprints lacks the ['experts'] scope(s)")

    monkeypatch.setitem(sys.modules, "experts4bit_qlora.adapter_dir",
                        types.SimpleNamespace(verify_adapter_dir=verify, AdapterRefused=Refused))
    assert A.read_manifest(path)["schema"] == A.SCHEMA_V2 and seen == [str(path)]
    other = tmp_path / "refuse"
    path.rename(other)
    with pytest.raises(ValueError, match=r"adapter refused \(7\)"):
        A.read_manifest(other)


def test_reading_v2_without_the_runtimes_verifier_is_refused_by_name(tmp_path, fake, monkeypatch):
    path = written_v2(tmp_path, fake)
    monkeypatch.setitem(sys.modules, "experts4bit_qlora.adapter_dir", None)
    with pytest.raises(ValueError, match="predates it"):
        A.read_manifest(path)


def test_verify_base_compares_each_recorded_scope_and_names_a_mismatch(tmp_path, fake):
    m = json.loads((written_v2(tmp_path, fake) / A.MANIFEST).read_text())
    model = bf16_stack_model()
    assert {s: r["match"] for s, r in A.verify_base(model, m).items()} == {"attention": True, "experts": True}
    m["fingerprints"]["experts"]["sha256"] = "b" * 64
    with pytest.raises(ValueError, match=r"\['experts'\] fingerprint\(s\) do not match"):
        A.verify_base(model, m)
    m["fingerprints"]["experts"]["scheme_id"] = "qfp-v9"
    with pytest.raises(ValueError, match="qfp-v9"):
        A.verify_base(model, m)
    assert A.verify_base(model, {"schema": A.SCHEMA_V1}) == {}


def test_load_adapter_checks_the_rebuilt_base_before_loading_any_tensor(tmp_path, fake, monkeypatch):
    from experts4bit_qlora import recipe

    path = written_v2(tmp_path, fake)
    monkeypatch.setattr(A, "read_manifest", lambda d: json.loads((path / A.MANIFEST).read_text()))
    rebuilt = bf16_stack_model()
    monkeypatch.setattr(recipe, "prepare_qlora_training", lambda *a, **k: types.SimpleNamespace(model=rebuilt))
    order = []
    monkeypatch.setattr(A, "verify_base", lambda model, manifest: order.append("verify") or {})
    monkeypatch.setattr(A, "load_adapter_weights", lambda model, d: order.append("load") or {})
    A.load_adapter(path, device="cpu")
    assert order == ["verify", "load"]


# ------------------------------------------------------------------------------------------------- inspect


def test_describe_prints_every_v2_fact_and_says_v1_records_none(tmp_path, fake):
    v2 = json.loads((written_v2(tmp_path, fake) / A.MANIFEST).read_text())
    text = "\n".join(A.describe(v2))
    for want in ("schema backend-adapter/2", "expert absmax: double-quantized (1 double-quantized stack(s))",
                 "attention base: bf16", "fused q/k/v in training: 0", "bitsandbytes 0.50.2",
                 "fingerprint experts: qfp-v1 " + "a" * 64 + " (2 modules, source resident", "run r, written"):
        assert want in text, want
    v1 = {k: v for k, v in v2.items() if k not in V2_KEYS}
    v1["schema"] = A.SCHEMA_V1
    assert "base facts: none recorded (backend-adapter/1)" in "\n".join(A.describe(v1))


def test_inspect_adapter_verifies_then_prints_and_refuses_a_damaged_directory(tmp_path, fake, monkeypatch, capsys):
    from loggetta import cli

    path = written_v2(tmp_path, fake)
    monkeypatch.setitem(sys.modules, "experts4bit_qlora.adapter_dir",
                        types.SimpleNamespace(verify_adapter_dir=lambda d: None, AdapterRefused=ValueError))
    assert cli.main(["inspect", "--adapter", str(path)]) == 0
    assert "expert absmax: double-quantized" in capsys.readouterr().out
    (path / A.WEIGHTS).write_bytes((path / A.WEIGHTS).read_bytes() + b"x")
    assert cli.main(["inspect", "--adapter", str(path)]) == 1
    assert "checksum mismatch" in capsys.readouterr().err


def test_every_real_executor_takes_the_run_id_execute_passes():
    """execute() passes run_id to every executor, unconditionally: deciding by its signature would drop it silently
    behind a wrapper that hides the parameter. So each executor must accept it."""
    import inspect

    from loggetta.backends import dense, experts4bit

    for backend in (experts4bit, dense):
        run = backend.executor("train")
        assert "run_id" in inspect.signature(run).parameters, backend.__name__


# ------------------------------------------------------------------------------------------------- end to end (real schema)


def real_moe(seed=5):
    """Real NF4 expert storage on the CPU (bitsandbytes quantizes here), attention LoRA over plain linears."""
    from experts4bit_qlora._vendor.experts import Experts4bit

    torch.manual_seed(seed)
    m = nn.Module()
    m.experts = ExpertsLoRA(Experts4bit.from_float(torch.randn(4, 128, 64, dtype=torch.bfloat16),
                                                   torch.randn(4, 64, 64, dtype=torch.bfloat16), quant_type="nf4",
                                                   blocksize=64), r=2, alpha=4)
    m.self_attn = Attn()
    return trainable(m)


@pytest.fixture
def real():
    schema = pytest.importorskip("experts4bit_qlora.adapter_schema")
    pytest.importorskip("experts4bit_qlora.adapter_dir")
    pytest.importorskip("bitsandbytes")
    return schema


def test_end_to_end_the_runtimes_verifier_accepts_what_loggetta_writes(tmp_path, real):
    from experts4bit_qlora.adapter_dir import verify_adapter_dir

    model = real_moe()
    report = {"commit": "0" * 40, **real.resolved_training_facts(model, None, fast_stats=None, qkv_stats=None)}
    out = A.save_adapter(model, Tokenizer(), tmp_path / "a", plan_with(expert_residency="device"), data={},
                         report=report, seed=0, run_id="e2e")
    ad = verify_adapter_dir(tmp_path / "a")                       # e4b's own reader, unmodified
    m = ad.manifest
    assert out["format"] == m["schema"] == A.SCHEMA_V2 and sorted(m["fingerprints"]) == ["attention", "experts"]
    assert m["fingerprints"]["experts"]["modules"] == 2 and m["fingerprints"]["experts"]["source"] == "resident"
    assert m["expert_absmax"]["form"] == "fp32"                   # nothing compressed it here
    assert A.read_manifest(tmp_path / "a")["run_id"] == "e2e"


def test_end_to_end_the_rebuilt_base_matches_and_a_changed_recipe_does_not(tmp_path, real):
    from experts4bit_qlora.absmax_dq import compress_expert_absmax_

    model = real_moe()
    report = {"commit": "0" * 40, **real.resolved_training_facts(model, None, fast_stats=None, qkv_stats=None)}
    A.save_adapter(model, Tokenizer(), tmp_path / "a", plan_with(), data={}, report=report, seed=0, run_id="r")
    manifest = A.read_manifest(tmp_path / "a")
    rebuilt = real_moe()                                          # the same base, built again
    assert all(r["match"] for r in A.verify_base(rebuilt, manifest).values())
    compress_expert_absmax_(rebuilt)                              # the other absmax form (ADAPTER_PORTABILITY F1)
    with pytest.raises(ValueError, match=r"\['experts'\] fingerprint\(s\) do not match"):
        A.verify_base(rebuilt, manifest)


def test_end_to_end_an_offload_run_fingerprints_from_the_host_homes(tmp_path, real):
    model = real_moe()
    base = model.experts.base
    home = {n: getattr(base, n).detach().clone() for n in ("gate_up_proj", "down_proj", "gate_up_absmax", "down_absmax")}
    for n in ("gate_up_proj", "down_proj"):
        base._parameters[n].data = torch.empty(0, dtype=home[n].dtype)
    for n in ("gate_up_absmax", "down_absmax"):
        base._buffers[n] = torch.empty(0, dtype=home[n].dtype)
    model.experts._offload = types.SimpleNamespace(base=base, home=home)
    report = {"commit": "0" * 40, **real.resolved_training_facts(model, None, fast_stats=None, qkv_stats=None)}
    with pytest.raises(ValueError, match="evicted"):              # recorded as device, but evicted: refused, nothing written
        A.save_adapter(model, Tokenizer(), tmp_path / "dev", plan_with(expert_residency="device"), data={},
                       report=report, seed=0, run_id="r")
    assert not (tmp_path / "dev").exists()
    A.save_adapter(model, Tokenizer(), tmp_path / "host", plan_with(expert_residency="host"), data={}, report=report,
                   seed=0, run_id="r")
    fp = A.read_manifest(tmp_path / "host")["fingerprints"]["experts"]
    assert fp["source"] == "offload_homes"
    assert fp["sha256"] == real.quant_fingerprint(real_moe(), "experts")["sha256"]   # same base, resident: same hash


def test_end_to_end_one_base_written_in_both_absmax_forms_fingerprints_differently(tmp_path, real):
    """The property the server-side check relies on: the same weights, written once with the fp32 absmax and once
    after training's double quantization (ADAPTER_PORTABILITY F1), record different expert fingerprints and say so."""
    from experts4bit_qlora.absmax_dq import compress_expert_absmax_

    fp32 = real_moe()
    report = {"commit": "0" * 40, **real.resolved_training_facts(fp32, None, fast_stats=None, qkv_stats=None)}
    A.save_adapter(fp32, Tokenizer(), tmp_path / "fp32", plan_with(), data={}, report=report, seed=0, run_id="fp32")
    dq = real_moe()
    n = compress_expert_absmax_(dq)                                # returns how many stacks it compressed
    assert n == 1
    rec = {"compressed": n}                                       # FAST_TRAIN_STATS["absmax_dq"]'s shape
    report = {"commit": "0" * 40,
              **real.resolved_training_facts(dq, None, fast_stats={"absmax_dq": rec}, qkv_stats=None)}
    A.save_adapter(dq, Tokenizer(), tmp_path / "dq", plan_with(), data={}, report=report, seed=0, run_id="dq")
    a, b = A.read_manifest(tmp_path / "fp32"), A.read_manifest(tmp_path / "dq")
    assert a["expert_absmax"]["form"] == "fp32" and b["expert_absmax"]["form"] == "double-quantized"
    assert a["fingerprints"]["experts"]["sha256"] != b["fingerprints"]["experts"]["sha256"]
    assert a["fingerprints"]["attention"]["sha256"] == b["fingerprints"]["attention"]["sha256"]   # attention untouched
