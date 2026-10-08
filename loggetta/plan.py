"""The ExecutionPlan: the planner's product. An inspectable, serializable, deterministic artifact, separate from
running anything; ``execution.execute`` hands it to the backend it selected.

A plan is computed from (model topology, hardware inventory, workload, constraints, available backends, earlier
receipts) and nothing else. It records the choice AND the alternatives with the reason each lost, the memory
estimate line by line with the basis of every line, what the estimate leaves out, and -- when nothing fits -- a
refusal that says why and what would change the answer. Refusal is a plan status, not an exception.

Schema identifiers are generic (``execution-plan/1``) so the project can be renamed without migrating any
serialized plan or receipt.
"""
from __future__ import annotations

import json
import math

from .schedule import describe as describe_lr
from dataclasses import asdict, dataclass, field

PLAN_SCHEMA = "execution-plan/1"
GiB = 1 << 30


@dataclass(frozen=True)
class Workload:
    """What to run: QLoRA training (``seq_len`` x ``micro_batch``) or paged serving (``context_len`` x ``concurrency``)."""

    kind: str = "train"                 # "train" | "serve"
    seq_len: int = 512
    micro_batch: int = 1
    grad_accum: int = 1
    steps: int = 20
    #: "adamw" (torch) or "adamw_8bit" (bitsandbytes): its state is part of the footprint
    optimizer: str = "adamw"
    # serving shape: tokens per sequence (prompt + output) and sequences decoded together
    context_len: int | None = None
    concurrency: int | None = None
    phase: str | None = None            # "prefill" | "decode" | None (both)
    data: dict | None = None            # optional TrainingData fields; old plans keep the Alpaca demonstration
    learning_rate: float = 2e-4
    #: passes over the dataset; when set, the planner derives ``steps`` from the data profile and records both
    epochs: float | None = None
    #: "constant" or "cosine" (warmup, then cosine decay); see :mod:`.schedule`
    lr_schedule: str = "constant"
    #: linear warmup steps; None = the schedule's default (3% of the steps under cosine, none under constant)
    warmup_steps: int | None = None

    def __post_init__(self):
        for name in ("seq_len", "micro_batch", "grad_accum", "steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.seq_len < 2:
            raise ValueError("seq_len must be at least 2 for causal language-model loss")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if self.data is not None:
            from .data import TrainingData

            if self.kind != "train":
                raise ValueError("training data does not apply to a serving plan")
            TrainingData.from_dict(self.data)
        if self.epochs is not None:
            if isinstance(self.epochs, bool) or not isinstance(self.epochs, (int, float)) \
                    or not math.isfinite(self.epochs) or self.epochs <= 0:
                raise ValueError("epochs must be finite and positive")
            if self.data is None:
                raise ValueError("epochs need a dataset (--dataset): the planner counts its tokens")
        from .schedule import SCHEDULES

        if self.lr_schedule not in SCHEDULES:
            raise ValueError(f"lr_schedule must be one of {', '.join(SCHEDULES)}")
        if self.warmup_steps is not None and (isinstance(self.warmup_steps, bool)
                                              or not isinstance(self.warmup_steps, int) or self.warmup_steps < 0):
            raise ValueError("warmup_steps must be a non-negative integer")

    @property
    def tokens_per_microbatch(self) -> int:
        return self.seq_len * self.micro_batch


@dataclass(frozen=True)
class Constraints:
    """What the caller insists on. Every field is optional: unset means "let the planner choose"."""

    device: int = 0
    #: device memory this plan may use; ``None`` = what the driver reports free now
    vram_budget: int | None = None
    #: host memory this plan may use; ``None`` = what the container can still allocate now
    ram_budget: int | None = None
    #: device memory kept unused for fragmentation and spikes; ``None`` = the planner's policy
    headroom: int | None = None
    #: where frozen experts may live, e.g. ("device",) forbids offload; ``None`` = anywhere a backend supports
    expert_residency: tuple | None = None
    #: backend setup fields fixed by the caller (expert mode), e.g. {"expert_kernel": "reference"}
    fixed: dict = field(default_factory=dict)
    #: "speed" | "min_vram" | "min_ram"
    objective: str = "speed"
    #: refuse a candidate whose PROVABLE lower bound on step time exceeds this (seconds); never a prediction
    target_s_per_step: float | None = None
    #: explicit permission for a backend executor still awaiting its registered GPU proof
    allow_development_executor: bool = False
    #: named prospective dense memory hypothesis; None keeps the shipped policy
    dense_reserve_policy: str | None = None
    #: explicit allocator declaration for a registered memory policy; its runner verifies before CUDA initialization
    allocator_profile: str | None = None


@dataclass(frozen=True)
class MemoryLine:
    name: str
    where: str          # "device" | "host"
    bytes: int
    basis: str          # "derived" | "heuristic" | "measured" | "inferred" | "policy"
    detail: str = ""


@dataclass(frozen=True)
class Candidate:
    backend: str
    setup: dict
    lines: tuple
    device_bytes: int
    host_bytes: int
    feasible: bool
    #: why it cannot run under the constraints (empty when feasible)
    rejected: tuple = ()
    unmodelled: tuple = ()
    #: where it sits in the objective's ordering, and the evidence behind that ordering
    rank: tuple = ()
    rank_note: str = ""
    #: lower bounds that follow from the estimate and a bandwidth (each with its basis); empty when none apply
    bounds: dict = field(default_factory=dict)
    #: the backend's one-line description of this setup (the planner never reads setup fields itself)
    label_text: str = ""

    def label(self) -> str:
        if self.label_text:
            return self.label_text
        s = self.setup
        return (f"experts on {s.get('expert_residency')}, {s.get('expert_kernel')} kernel"
                + (", NF4 attention" if s.get("attn_4bit") else "")
                + (", pageable" if s.get("expert_residency") == "host" and not s.get("pin", True) else "")
                + (f", keep {s['keep_moe_layers']} MoE layers" if s.get("keep_moe_layers") else ""))


@dataclass(frozen=True)
class ExecutionPlan:
    status: str                          # "feasible" | "refused"
    model: dict
    hardware: dict
    workload: Workload
    constraints: Constraints
    budget: dict
    selected: Candidate | None
    alternatives: tuple
    reasons: tuple
    warnings: tuple = ()
    refusal: dict | None = None
    performance: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    schema: str = PLAN_SCHEMA
    #: the training data's profile (``data.encode_dataset``), when the dataset was read before planning
    data_profile: dict | None = None

    def to_dict(self) -> dict:
        out = asdict(self)
        if out["constraints"]["allow_development_executor"] is False:
            del out["constraints"]["allow_development_executor"]
        for key in ("dense_reserve_policy", "allocator_profile"):
            if out["constraints"][key] is None:
                del out["constraints"][key]
        # Optional v1 additions: old plans still round-trip byte-for-byte, including their omitted defaults.
        if out["workload"]["data"] is None:
            del out["workload"]["data"]
        if out["workload"]["learning_rate"] == 2e-4:
            del out["workload"]["learning_rate"]
        if out["workload"]["epochs"] is None:
            del out["workload"]["epochs"]
        if out["workload"]["lr_schedule"] == "constant":
            del out["workload"]["lr_schedule"]
        if out["workload"]["warmup_steps"] is None:
            del out["workload"]["warmup_steps"]
        if out["data_profile"] is None:
            del out["data_profile"]
        return out

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=1, sort_keys=True, default=str)

    @classmethod
    def from_dict(cls, d: dict) -> "ExecutionPlan":
        cand = lambda c: None if c is None else Candidate(**{**c, "lines": tuple(MemoryLine(**x) for x in c["lines"]),  # noqa: E731
                                                            **{k: tuple(c[k]) for k in ("rejected", "unmodelled", "rank")}})
        d = dict(d)
        d["workload"] = Workload(**d["workload"])
        d["constraints"] = Constraints(**{**d["constraints"], "expert_residency": (
            tuple(d["constraints"]["expert_residency"]) if d["constraints"]["expert_residency"] else None)})
        d["selected"] = cand(d["selected"])
        d["alternatives"] = tuple(cand(c) for c in d["alternatives"])
        for k in ("reasons", "warnings"):
            d[k] = tuple(d[k])
        return cls(**d)

    # ------------------------------------------------------------------------------------------------- rendering
    def render(self, verbose: bool = False) -> str:
        gb = lambda n: f"{n / GiB:6.2f} GiB"  # noqa: E731
        m, w = self.model, self.workload
        shape = (f"{w.context_len} tokens per sequence x {w.concurrency} sequences" if w.kind == "serve" else
                 f"seq {w.seq_len} x micro-batch {w.micro_batch} (= {w.tokens_per_microbatch} tokens/forward), "
                 f"grad-accum {w.grad_accum}, {w.steps} steps")
        out = [f"Model     {m.get('model')}  ({m.get('model_type')}; {m.get('summary', '')})",
               f"Workload  {w.kind}: {shape}",
               f"Budget    device {gb(self.budget['device'])} [{self.budget['device_source']}]   "
               f"host {gb(self.budget['host'])} [{self.budget['host_source']}]   "
               f"headroom {gb(self.budget['headroom'])} [policy]"
               + (f"   host headroom {gb(self.budget['host_headroom'])} [policy]" if self.budget.get("host_headroom") else "")]
        if w.kind == "train":
            d = self.data_profile
            if d is not None:
                ln, src = d["lengths"], d["source"]
                ident = (f"; sha256 {src['sha256'][:12]}" if src.get("sha256") else
                         f"; revision {src['requested_revision']}" if src.get("requested_revision") else
                         "; unpinned Hub revision" if src.get("kind") == "hub" else "")
                out += [f"Data      {w.data['source']} ({d['format']}, split {d['split']}{ident}): {d['rows']:,} examples, "
                        f"{d['tokens']:,} tokens, validated and tokenized before planning",
                        f"          tokens per example: p50 {ln['p50']:,}, p90 {ln['p90']:,}, p99 {ln['p99']:,}, max {ln['max']:,}; "
                        + (f"loss on assistant tokens ({d['loss_tokens'] / d['tokens']:.0%})" if d.get("loss_mode") == "assistant"
                           else "full-sequence loss")
                        + ("; isolated packing" if d.get("packing_mode") == "isolated" else "; concatenated packing")
                        + f"; {describe_lr(w)}"]
            elif w.data is not None:
                loss = {"all": "full-sequence loss", "assistant": "loss on assistant tokens",
                        "auto": "loss on assistant tokens for chat/alpaca, else full-sequence"}[w.data.get("loss", "all")]
                out += [f"Data      {w.data['source']} ({w.data.get('format', 'auto')}, split {w.data.get('split', 'train')}); "
                        f"{loss}; {describe_lr(w)}"]
            else:
                out += [f"Data      demonstration: tatsu-lab/alpaca, full-sequence loss, repeated as needed; {describe_lr(w)}"]
        if self.status == "refused":
            out += ["", "NOT FEASIBLE under the requested constraints.", ""]
            out += [f"  {r}" for r in self.refusal.get("reasons", ())]
            if self.refusal.get("closest"):
                out += ["", f"Closest candidate: {self.refusal['closest']}"]
                out += [f"  {ln}" for ln in self.refusal.get("closest_lines", ())]
            if self.refusal.get("suggestions"):
                out += ["", "What would change the answer:"] + [f"  - {s}" for s in self.refusal["suggestions"]]
        else:
            c = self.selected
            out += ["", f"Selected  backend {c.backend}: {c.label()}"]
            out += [f"  {k}: {v}" for k, v in sorted(c.setup.items())]
            out += ["", "Estimated memory (each line says how it is known)"]
            out += _lines(c.lines)
            out += [f"  {'device total':44s} {gb(c.device_bytes)}   of {gb(self.budget['device'])} budget",
                    f"  {'host total':44s} {gb(c.host_bytes)}   of {gb(self.budget['host'])} budget"]
            if c.unmodelled:
                out += ["", "Not modelled (each can make the real peak higher)"] + [f"  - {u}" for u in c.unmodelled]
        out += ["", "Why"] + [f"  - {r}" for r in self.reasons]
        if self.warnings:
            out += ["", "Warnings"] + [f"  - {x}" for x in self.warnings]
        if self.performance:
            out += ["", f"Performance  {self.performance.get('statement')}"]
        if self.selected is not None and "s_per_step_lower_bound" in self.selected.bounds:
            b = self.selected.bounds
            out.append(f"  transfer bound  >= {b['s_per_step_lower_bound']:.2f} s/step: {b['link_bytes_per_step'] / 1e9:.2f} GB "
                       f"host-to-device per step over <= {b['link_gbps']:.2f} GB/s [{b['link_gbps_basis']}: {b['link_gbps_source']}]")
        if self.alternatives:
            out += ["", "Alternatives considered"]
            for a in self.alternatives:
                tag = "fits " if a.feasible else "no   "
                why = "; ".join(a.rejected) if a.rejected else a.rank_note
                out.append(f"  [{tag}] {a.label():58s} device {gb(a.device_bytes)}  host {gb(a.host_bytes)}  {why}")
                if verbose:
                    out += ["      " + x for x in _lines(a.lines)]
        return "\n".join(out)


def _lines(lines) -> list:
    out = []
    for ln in lines:
        out.append(f"  {ln.where:6s} {ln.name:37s} {ln.bytes / GiB:6.2f} GiB  [{ln.basis}]"
                   + (f"  {ln.detail}" if ln.detail else ""))
    return out
