"""The planner with a backend that is not experts4bit-qlora: its own topology type, its own setup fields, its own
admission. Pure Python (no torch, no backend packages), so it shows the planner is policy over whatever a backend
answers: it plans, refuses, suggests, warns and learns from receipts without reading a single setup field itself."""
import json
from dataclasses import dataclass, field

import pytest

from loggetta import Constraints, Workload, describe_model, plan
from loggetta.hardware import GPU, Fact, HardwareProfile, Host
from loggetta.model import NoModelProvider
from loggetta.plan import ExecutionPlan

GiB = 1 << 30


def gpu(free_gib=24.0, total_gib=24.0, width=16):
    r = lambda v: Fact(v, "reported")  # noqa: E731
    return GPU(index=0, vendor="nvidia", name="Test GPU", uuid=None, compute_capability=r((8, 9)),
               memory_total=r(int(total_gib * GiB)), memory_free=r(int(free_gib * GiB)), driver=r("575.64.05"),
               pcie_gen_max=r(4), pcie_width_max=r(16), pcie_gen_current=r(4), pcie_width_current=r(width))


def hw(free_gib=24.0, ram_gib=40.0, **kw):
    r = lambda v: Fact(v, "reported")  # noqa: E731
    host = Host(cpu_model=r("test cpu"), cpus=r(8), memory_total=r(64 * GiB), memory_available=r(int(ram_gib * GiB)),
                memory_limit=r(64 * GiB))
    return HardwareProfile(gpus=(gpu(free_gib, **kw),), host=host, platform="Linux x86_64")


@dataclass(frozen=True)
class Shape:
    """The toy backend's own description of a model: nothing like experts4bit-qlora's topology."""

    model: str
    params: int
    why_not: str | None = None


@dataclass
class Status:
    available: bool = True
    reason: str = ""
    versions: dict = field(default_factory=lambda: {"toy": "1"})
    kernels: dict = field(default_factory=dict)


class Toy:
    """A dense-style backend: frozen weights in bf16 or 4-bit, resident or streamed from the host."""

    NAME = "toy"
    WORKLOADS = ("train",)
    SLACK_KEYS = {"train": ("placement",)}
    SIZES = {"toy/7b": 7_000_000_000, "toy/30b": 30_000_000_000}

    @staticmethod
    def describe(model, *, revision=None, trust_remote_code=False):
        return Shape(model, Toy.SIZES.get(model, 1_000_000_000), None if model in Toy.SIZES else "toy knows no such model")

    @staticmethod
    def refusal(topology):
        return "not a model the toy backend described" if not isinstance(topology, Shape) else topology.why_not

    @staticmethod
    def summary(topology):
        return {"model": topology.model, "model_type": "toy", "summary": f"{topology.params / 1e9:.0f}B toy params"}

    @staticmethod
    def probe(gpu):
        return Status()

    @staticmethod
    def policy_notes(topology, constraints):
        return []

    @staticmethod
    def candidates(topology, workload, constraints, status):
        unknown = set(constraints.fixed) - {"precision", "placement"}
        if unknown:
            raise ValueError(f"unknown setup fields fixed by the caller: {sorted(unknown)}")
        return [{"precision": p, "placement": w, **constraints.fixed}
                for p in ("bf16", "q4") for w in ("device", "stream")]

    @staticmethod
    def estimate(topology, setup, workload):
        weights = int(topology.params * {"bf16": 2, "q4": 0.5}[setup["precision"]])
        if setup["placement"] == "device":
            lines = [("frozen weights", "device", weights, "derived", "")]
        else:
            lines = [("frozen weights, two layers staged", "device", weights // 8, "derived", ""),
                     ("frozen weights, host home", "host", weights, "derived", ""),
                     ("frozen weights streamed per micro-batch", "link", 2 * weights, "derived", "")]
        lines.append(("activations", "device", workload.tokens_per_microbatch * 256 * 1024, "heuristic",
                      "256 KiB per token"))
        return lines, ("toy workspaces",), ()

    @staticmethod
    def speed_rank(setup):
        return (setup["placement"] == "stream", setup["precision"] == "q4")

    @staticmethod
    def label(setup):
        return f"{setup['precision']} weights, {setup['placement']}"

    @staticmethod
    def explain(sel, feasible, infeasible, budget, status, constraints, workload):
        return [f"toy chose {Toy.label(sel.setup)}"]

    @staticmethod
    def residency(setup):
        return {"device": "device", "stream": "host"}.get(setup.get("placement"))

    @staticmethod
    def plan_warnings(setup, gpu):
        if setup["placement"] == "stream" and gpu.pcie_width_current.value < gpu.pcie_width_max.value:
            return ["toy weights stream over a narrowed link"]
        return []

    @staticmethod
    def relaxed_candidates(topology, workload, constraints, status):
        if constraints.fixed.get("placement") != "device":
            return []
        return [("allow streaming", {"precision": p, "placement": "stream"}) for p in ("bf16", "q4")]


@dataclass(frozen=True)
class Other:
    model: str


class OtherBackend(Toy):
    """A second backend that describes models its own way and plans only those."""

    NAME = "other"

    @staticmethod
    def describe(model, *, revision=None, trust_remote_code=False):
        return Other(model)

    @staticmethod
    def refusal(topology):
        return "the other backend plans nothing yet"


class Missing(Toy):
    NAME = "missing"

    @staticmethod
    def describe(model, *, revision=None, trust_remote_code=False):
        raise NoModelProvider("the missing backend's package is not installed")


def toy_plan(model="toy/7b", free_gib=24.0, constraints=Constraints(), observations=(), backends=(Toy,), **kw):
    return plan(Toy.describe(model), hw(free_gib, **kw), Workload(seq_len=512), constraints, backends=backends,
                observations=observations)


def test_a_second_backend_plans_with_its_own_setup_fields():
    p = toy_plan()
    assert p.status == "feasible" and p.selected.backend == "toy"
    assert p.selected.setup == {"precision": "bf16", "placement": "device"}
    assert p.model == {"model": "toy/7b", "model_type": "toy", "summary": "7B toy params"}
    assert "toy chose bf16 weights, device" in p.reasons
    assert all(set(c.setup) == {"precision", "placement"} for c in (p.selected, *p.alternatives))
    assert p.to_json() == toy_plan().to_json()
    assert ExecutionPlan.from_dict(json.loads(p.to_json())).to_json() == p.to_json()
    assert "Selected  backend toy: bf16 weights, device" in p.render()


def test_a_tight_card_takes_the_backends_next_choice():
    p = toy_plan(free_gib=12.0)
    assert p.selected.setup == {"precision": "q4", "placement": "device"}
    assert any(not c.feasible and c.setup == {"precision": "bf16", "placement": "device"} for c in p.alternatives)


def test_the_backends_refusal_is_the_plans_answer():
    p = toy_plan("toy/unknown")
    assert p.status == "refused" and p.refusal["reasons"] == ["toy knows no such model"]
    assert p.model["model"] == "toy/unknown"


def test_with_two_backends_the_one_that_admits_plans_and_a_refusal_names_each():
    p = toy_plan(backends=(OtherBackend, Toy))
    assert p.status == "feasible" and p.selected.backend == "toy" and p.model["model_type"] == "toy"
    p = toy_plan("toy/unknown", backends=(Toy, OtherBackend))
    assert p.status == "refused"
    assert p.refusal["reasons"] == ["toy: toy knows no such model", "other: the other backend plans nothing yet"]


def test_describe_model_returns_the_first_description_a_backend_can_plan_for():
    assert describe_model("toy/7b", backends=(Missing, OtherBackend, Toy)) == Shape("toy/7b", 7_000_000_000)
    assert describe_model("toy/7b", backends=(OtherBackend,)) == Other("toy/7b")       # refused, but described
    with pytest.raises(NoModelProvider, match="not installed"):
        describe_model("toy/7b", backends=(Missing,))


def test_a_relaxation_the_backend_offers_is_priced_into_the_refusal():
    p = toy_plan("toy/30b", free_gib=12.0, constraints=Constraints(fixed={"placement": "device"}))
    assert p.status == "refused"
    assert any(s.startswith("allow streaming: ") and s.endswith(" GiB host fits") for s in p.refusal["suggestions"])


def test_the_backends_warnings_reach_the_plan():
    p = toy_plan("toy/30b", free_gib=12.0, width=8)
    assert p.selected.setup["placement"] == "stream"
    assert "toy weights stream over a narrowed link" in p.warnings
    assert p.selected.bounds["link_bytes_per_step"] == 2 * 15_000_000_000


def _receipt(placement):
    return {"run_id": f"toy-{placement}", "status": "OK", "model": {"model": "toy/7b"},
            "workload": {"kind": "train", "tokens_per_microbatch": 512},
            "setup": {"precision": "bf16", "placement": placement},
            "plan": {"selected": {"backend": "toy"}},
            "hardware": {"gpu": {"name": "Test GPU", "driver": "575.64.05"}},
            "measured": {"cuda_context_bytes": 300 << 20, "device_peak_bytes": 10 * GiB,
                         "device_reserved_peak_bytes": 11 * GiB, "host_anon_after_load_bytes": 2 * GiB,
                         "host_baseline_bytes": 1 * GiB}}


def test_receipts_are_read_through_the_backend_that_ran_them():
    p = toy_plan(observations=(_receipt("device"),))
    lines = {ln.name: ln for ln in p.selected.lines}
    assert lines["CUDA context + library workspaces"].bytes == 300 << 20
    base = lines["process baseline (torch, CUDA, libraries, model objects)"]
    assert base.bytes == 2 * GiB and "anonymous RSS after load" in base.detail      # weights were on the device
    reserve = lines["allocator reserve (cached, unallocated blocks)"]
    assert reserve.basis == "measured" and "this GPU, setup and model" in reserve.detail
    streamed = toy_plan(observations=(_receipt("stream"),))
    base = {ln.name: ln for ln in streamed.selected.lines}["process baseline (torch, CUDA, libraries, model objects)"]
    assert base.bytes == 1 * GiB                                                    # streamed: pinned homes inflate RSS


def test_when_every_backend_refuses_each_reason_reaches_the_plan():
    from loggetta.model import Refused

    t = describe_model("toy/unknown", backends=(Toy, OtherBackend))
    assert isinstance(t, Refused) and t.model == "toy/unknown"          # attributes read through to Toy's description
    assert t.reasons == {"toy": "toy knows no such model", "other": "the other backend plans nothing yet"}
    p = plan(t, hw(), Workload(seq_len=512), backends=(Toy, OtherBackend))
    assert p.refusal["reasons"] == ["toy: toy knows no such model", "other: the other backend plans nothing yet"]
    assert p.model["model_type"] == "toy"
