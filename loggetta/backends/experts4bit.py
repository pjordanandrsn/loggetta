"""The experts4bit-qlora backend: fused-MoE models in 4-bit, for QLoRA training and the paged server.

This module is the planner's side of the boundary: how it asks experts4bit-qlora (and, through it, grouped-nf4-gemm)
what a setup costs, and how it hands a plan back to run. Everything it knows about memory, structure or mechanism it
ASKS the package that owns it:

* topology and admission -- ``experts4bit_qlora.arch.topology.describe_moe``;
* setup validity and the itemized footprint -- ``experts4bit_qlora.recipe`` (``setup_refusals``,
  ``estimate_qlora_footprint``) and ``serve_recipe`` (``estimate_serve_footprint``, ``min_hot_rows``), which price
  the same modules ``prepare_qlora_training`` and ``serve_paged`` build;
* which kernels run on a device -- grouped-nf4-gemm's ``nf4_route.route_for`` / ``MIN_CAPABILITY``, and
  experts4bit-qlora's ``fused_append_unsupported`` for decode graphs.

What is decided HERE is policy: which setups are worth considering, and in what order a speed objective tries them.
Running a plan is :func:`executor`: a training plan's setup is built by ``prepare_qlora_training`` itself.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

NAME = "experts4bit"
WORKLOADS = ("train", "serve")
#: setup fields that separate allocator-slack regimes, per workload (measured: training offload vs resident; serving
#: solver tiers 15% vs all-VRAM 1-2% on the A2000)
SLACK_KEYS = {"train": ("expert_residency", "expert_kernel"),
              "serve": ("placement", "graphs", "prefill_graph", "exp_int4", "attn_int4")}
#: what a receipt that predates a key field ran with (the server's defaults: the int4 levers off)
SLACK_DEFAULTS = {"serve": {"exp_int4": False, "attn_int4": False}}
#: setup fields the planner sizes to the budget (fill_knobs) or the mechanism derives from those sizes (``hot_rows``,
#: resolved from the tier split after the fill): they move a few large one-time allocations, not the workload's
#: transient ones, so a receipt that differs only in them has the candidate's allocation pattern (lanes SV4 and SV6,
#: Qwen3-30B 8 x 8192 under the solver on an RTX 4090: 4.1% and 3.8% slack at VRAM tiers 10.9 and 12.6 GiB)
BUDGET_FIELDS = {"serve": ("vram_gb", "dram_gb", "hot_rows")}
#: which probed kernels each workload uses: the planner reports only those as unusable
KERNELS_FOR = {"train": ("grouped_nf4", "reference"), "serve": ("paged_fp8", "paged_graphs", "cpu_tier")}
GiB = 1 << 30

#: The order a speed objective tries setups in, and the evidence for each step of it. Not a performance model:
#: no time is predicted, only which of two otherwise-valid setups the measurements say is faster.
SPEED_EVIDENCE = {
    "expert_kernel": "grouped_nf4 before reference: e4b.train.h2h.unsloth.olmoe.5090.2026-09-19 (1.39 vs 14.88 s/step), "
                     "gnf4.kernel.e2e-training-real-prose (4.50x on a 4090)",
    "expert_residency": "device before host: host residency adds a per-layer host-to-device copy in every forward and "
                        "every recompute (structural; the register has no same-card resident/offload ratio)",
    "attn_4bit": "bf16 attention before NF4: NF4 attention measured 1.013x step cost for -1.34 GB "
                 "(experts4bit_qlora.lora.quantize_attention_projections_4bit, audit VRAMCENSUS)",
    "pin": "pinned host homes before pageable: a pageable copy is staged through a bounce buffer (CUDA), so it is "
           "slower per byte; pinned requests round up to a power of two (grouped-nf4-gemm#71), pageable ones do not",
}


@dataclass
class BackendStatus:
    available: bool
    reason: str = ""
    versions: dict = field(default_factory=dict)
    #: kernel name -> (usable on the planned device, why)
    kernels: dict = field(default_factory=dict)


def _version(dist):
    try:
        from importlib.metadata import version

        return version(dist)
    except Exception:  # noqa: BLE001 - absence is the answer
        return None


def describe(model, *, revision=None, trust_remote_code=False):
    """The model as experts4bit-qlora sees it (``describe_moe``: config + a meta-device module tree, no weights). A
    config its loader refuses comes back described, with ``loader_refusal`` set: :func:`refusal` turns that into words."""
    from ..model import NoModelProvider

    try:
        from experts4bit_qlora.arch.topology import describe_moe
    except ImportError as e:
        raise NoModelProvider("no model-family provider installed: experts4bit-qlora with arch.topology is "
                              f"required to describe a model ({e})") from e
    return describe_moe(model, revision=revision, trust_remote_code=trust_remote_code)


def refusal(topology) -> str | None:
    """Why this backend can plan nothing for ``topology``, or None. A description from another backend is not ours."""
    if not hasattr(topology, "expert_stacks"):
        return "not a model experts4bit-qlora described"
    if topology.loader_refusal:
        return f"the model-family layer cannot load this model: {topology.loader_refusal}"
    return None


def summary(topology) -> dict:
    """The plan's ``model`` section: identity and the topology facts a reader needs."""
    t = topology
    return {"model": t.model, "model_type": t.model_type, "revision": t.revision,
            "convention": t.convention, "summary": t.summary(), "n_layers": t.n_layers,
            "moe_layers": len(t.expert_stacks), "n_experts": t.n_experts, "top_k": t.top_k,
            "expert_params": t.expert_numel, "dense_params": t.dense_numel,
            "loader_refusal": t.loader_refusal, "provenance": t.provenance}


def residency(setup: dict) -> str | None:
    """Where a setup's frozen weights live: "device", "host" (streamed to the GPU), or None if the setup does not say
    (a serving setup)."""
    return setup.get("expert_residency")


def plan_warnings(setup: dict, gpu) -> list:
    """Warnings about running ``setup`` on ``gpu`` right now."""
    if setup.get("expert_residency") == "host" and gpu.pcie_width_current.value and gpu.pcie_width_max.value and \
            gpu.pcie_width_current.value < gpu.pcie_width_max.value:
        return [f"host-resident experts stream over PCIe, and the driver reports the link at "
                f"x{gpu.pcie_width_current.value} of x{gpu.pcie_width_max.value} right now"]
    return []


def relaxed_candidates(topology, workload, constraints, status) -> list:
    """Setups outside the caller's constraints whose fit would change a refusal, each with the words for that change:
    ``[(words, setup)]``, in the order to try them. Here: host-backed experts, when the caller forbade them."""
    if not constraints.expert_residency or "host" in constraints.expert_residency:
        return []
    from dataclasses import replace

    relaxed = replace(constraints, expert_residency=None)
    return [("allow host-backed experts", s) for s in candidates(topology, workload, relaxed, status)
            if s.get("expert_residency") == "host"]


def probe(gpu) -> BackendStatus:
    """Is this backend importable, and which expert kernels can it use on ``gpu`` (a hardware.GPU, or None)?"""
    try:
        import experts4bit_qlora  # noqa: F401
        from experts4bit_qlora.recipe import estimate_qlora_footprint  # noqa: F401
    except ImportError as e:
        return BackendStatus(False, f"experts4bit-qlora with recipe.estimate_qlora_footprint is not importable: {e}")
    import torch

    versions = {"experts4bit-qlora": _version("experts4bit-qlora"), "grouped-nf4-gemm": _version("grouped-nf4-gemm"),
                "torch": torch.__version__, "bitsandbytes": _version("bitsandbytes"),
                "transformers": _version("transformers")}
    kernels = {}
    cap = gpu.compute_capability.value if gpu is not None else None
    if cap is None:
        kernels["reference"] = (False, "no CUDA device: bitsandbytes' 4-bit training path needs one")
    else:
        kernels["reference"] = (True, "bitsandbytes per-expert loop (always available on CUDA)")
    try:
        import nf4_qlora  # noqa: F401
        from nf4_route import route_for
    except ImportError as e:
        kernels["grouped_nf4"] = (False, f"grouped-nf4-gemm with nf4_route.route_for not importable ({e})")
    else:
        route, why = route_for(cap, has_grouped_mm=hasattr(torch, "_grouped_mm"))
        kernels["grouped_nf4"] = (route is not None, f"training route {route!r}: {why}" if route else why)
    try:
        import fp8_kv  # noqa: F401
        import fp8_paged_attn  # noqa: F401
        from experts4bit_qlora.serve_recipe import estimate_serve_footprint  # noqa: F401
        from nf4_route import MIN_CAPABILITY
    except ImportError as e:
        kernels["paged_fp8"] = (False, f"the paged server's estimate or grouped-nf4-gemm's paged FP8 kernels are not "
                                       f"importable ({e})")
    else:
        if cap is None or tuple(cap) < tuple(MIN_CAPABILITY):
            kernels["paged_fp8"] = (False, f"grouped-nf4-gemm's floor is sm_{MIN_CAPABILITY[0]}{MIN_CAPABILITY[1]}")
        else:
            try:
                from experts4bit_qlora.engines.fp8_paged_kv import fused_append_unsupported
            except ImportError:
                kernels["paged_graphs"] = (False, "this experts4bit-qlora cannot say whether its decode graphs run "
                                                  "here (no fp8_paged_kv.fused_append_unsupported): eager decode only")
            else:
                why = fused_append_unsupported(tuple(cap))
                kernels["paged_graphs"] = (why is None, why or "bucketed decode graphs (fused FP8 KV append)")
            try:
                import cpu_grouped
                cpu_ok = bool(cpu_grouped.cpu_kernels_available())
            except ImportError:
                cpu_ok = False
            kernels["cpu_tier"] = (cpu_ok, "grouped-nf4-gemm's native CPU kernels (the solver's DRAM tier computes on "
                                           "the CPU)" if cpu_ok else "grouped-nf4-gemm's native CPU kernels are not "
                                           "built here: the solver's tiered placement cannot engage")
            kernels["paged_fp8"] = (True, "paged FP8 KV pool + attention, " + (
                "fp8 compute" if tuple(cap) >= (8, 9) else "f32 compute (fp8 compute needs sm_89+, "
                                                            "fp8_paged_attn.fp8_compute_unsupported)"))
    return BackendStatus(True, "", versions, kernels)


def candidates(topology, workload, constraints, status: BackendStatus) -> list:
    """Setups worth estimating, as dicts of ``QLoRASetup`` fields. Fields in ``constraints.fixed`` are not varied."""
    if workload.kind == "serve":
        return _serve_candidates(workload, constraints, status)
    from experts4bit_qlora.recipe import QLoRASetup

    fixed = dict(constraints.fixed)
    unknown = set(fixed) - set(QLoRASetup.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown setup fields fixed by the caller: {sorted(unknown)}")
    attn = topology.attention
    wrappable = attn is not None and attn.count > 0
    base = QLoRASetup().to_dict()
    if not wrappable and "train_attention" not in fixed:
        base["train_attention"] = False            # policy_notes() says why
    axes = {
        "expert_residency": list(constraints.expert_residency or ("device", "host")),
        "expert_kernel": [k for k in ("grouped_nf4", "reference") if status.kernels.get(k, (False,))[0]],
        "attn_4bit": [False, True] if wrappable and not attn.any_bias else [False],
        "pin": [True, False],
    }
    for k, v in fixed.items():
        axes[k] = [v]
    keys = sorted(axes)
    out = []
    for combo in itertools.product(*(axes[k] for k in keys)):
        setup = {**base, **fixed, **dict(zip(keys, combo))}
        if setup["expert_residency"] == "device" and not setup["pin"] and "pin" not in fixed:
            continue                               # pinning only means something for host-resident experts
        out.append(setup)
    return out


SERVE_EVIDENCE = {
    "graphs": "decode graphs before eager decode: serve_paged's own default for all-VRAM placement on CUDA "
              "(_graphs_env); the graph pools are listed as not modelled",
}


def label(setup: dict) -> str:
    if "max_seqs" in setup:
        tiers = (f", tiers VRAM {setup['vram_gb']:.2f} / DRAM {setup['dram_gb']:.2f} GiB"
                 if setup.get("placement") == "solver" else "")
        return (f"serve {setup['placement']}, fp8 paged KV, {setup['max_seqs']} seqs x {setup['max_tokens_per_seq']} "
                f"tokens{', decode graphs' if setup.get('graphs') else ''}{tiers}"
                f"{', prefill graph ' + str(setup['prefill_graph']) if setup.get('prefill_graph') not in (None, '0') else ''}"
                f"{', int4 experts' if setup.get('exp_int4') else ''}{', int4 attention' if setup.get('attn_int4') else ''}")
    s = setup
    return (f"experts on {s.get('expert_residency')}, {s.get('expert_kernel')} kernel"
            + (", NF4 attention" if s.get("attn_4bit") else "")
            + (", pageable" if s.get("expert_residency") == "host" and not s.get("pin", True) else "")
            + (f", keep {s['keep_moe_layers']} MoE layers" if s.get("keep_moe_layers") else ""))


def run_tag(setup: dict) -> str:
    """The setup's part of a receipt's run id, e.g. ``device-grouped_nf4-attn4``."""
    return f"{setup['expert_residency']}-{setup['expert_kernel']}{'-attn4' if setup['attn_4bit'] else ''}"


def executor(kind: str):
    """What runs a feasible plan of this workload kind through experts4bit-qlora: ``run(plan, *, seed, log)``, or
    None when the kind is planned only (serve: the plan's Why carries the server's environment)."""
    if kind == "train":
        from .experts4bit_train import run

        return run
    return None


def explain(sel, feasible, infeasible, budget, status, constraints, workload) -> list:
    """Why this candidate, in words: the backend knows its own setup fields; the planner does not read them."""
    s, out = sel.setup, []
    if workload.kind == "serve":
        if s.get("placement") == "solver":
            by = {ln.name: ln.bytes for ln in sel.lines}
            tier = lambda key: next((b for n, b in by.items() if n.startswith(key)), 0) / GiB  # noqa: E731
            allv = [c for c in feasible + infeasible if c.setup.get("placement") == "all-vram"]
            need = min((c.device_bytes for c in allv), default=None)
            out.append(f"experts split across tiers (placement solver): {tier('expert stacks, VRAM'):.2f} GiB in VRAM, "
                       f"{tier('expert stacks, DRAM'):.2f} GiB in DRAM (computed on the CPU), "
                       f"{tier('expert rows on NVMe'):.2f} GiB on NVMe (streamed through the cold tier)"
                       + (f"; all-VRAM needs {need / GiB:.2f} GiB + headroom, over the {budget['device'] / GiB:.2f} GiB "
                          "budget" if need is not None and need + budget["headroom"] > budget["device"] else ""))
            out.append("tier budgets: VRAM filled to the device budget, then DRAM to the host budget less its headroom "
                       "(E4B_PAGED_VRAM_GB / E4B_PAGED_DRAM_GB); routing is assumed uniform, as serve_paged runs the "
                       "solver without a profile")
            if "hot_rows" not in constraints.fixed:
                out.append(f"cold tier: hot_rows {s.get('hot_rows')}, the fewest the split can serve with (a cold layer's "
                           "routed experts per step); its pinned landing, cold view and setup tier scale with it, so "
                           "the server's default of 64 would cost host memory the DRAM tier can use")
            out.append(f"kernels: {status.kernels.get('cpu_tier', (None, ''))[1]}")
        else:
            out.append(f"all experts resident (placement all-vram): {sel.device_bytes / GiB:.2f} GiB estimated + "
                       f"{budget['headroom'] / GiB:.2f} GiB headroom fits the {budget['device'] / GiB:.2f} GiB budget")
        kv = next((ln for ln in sel.lines if ln.name == "FP8 paged KV pool"), None)
        if kv:
            out.append(f"KV pool {kv.bytes / GiB:.2f} GiB for {s['max_seqs']} sequences x {s['max_tokens_per_seq']} "
                       "tokens; it scales linearly in both")
        out.append(f"kernels: {status.kernels.get('paged_fp8', (None, ''))[1]}; grouped NF4 / int4 decode routes as "
                   "serve_paged resolves them (reported at /health)")
        if constraints.objective == "speed" and any(c.setup.get("graphs") for c in feasible + infeasible):
            out += [f"ordering, {k}: {v}" for k, v in SERVE_EVIDENCE.items()]
        from experts4bit_qlora.serve_recipe import DEFAULT_BUCKETS
        if s.get("graphs") and "buckets" not in constraints.fixed and tuple(s.get("buckets", ())) != tuple(DEFAULT_BUCKETS):
            out.append(f"decode-graph buckets {list(s['buckets'])}: none above {s['max_seqs']} sequences, since a step "
                       "never carries more rows than sequences; each bucket costs a graph, and the largest sizes the "
                       "scratch slots (per-slot state on a hybrid model)")
        if s.get("prefill_graph") == "0" and "prefill_graph" not in constraints.fixed:
            out.append("first-chunk prefill graph off (the server's default is auto): its private pool is not priced "
                       "(SC2b measured +3.3 GiB at Qwen3-30B), so the plan's memory would not bound the process; fix "
                       "prefill_graph=auto to let the server engage it when that much is free")
        by = {ln.name: ln for ln in sel.lines}
        if s.get("exp_int4"):
            rd = by.get("int4 repack: one layer's experts in fp32 (load)")
            out.append("experts on the int4-b32 grid (exp_int4, round-to-nearest): repacked at load from the source "
                       "checkpoint, which must be on local disk and is read whole"
                       + (f"; the repack holds {rd.bytes / GiB:.2f} GiB of host memory at its peak, one layer in fp32 "
                          "at a time, and the process keeps most of it after load" if rd else ""))
        if s.get("attn_int4"):
            # the kept bf16 copy is exactly bf16 attention's bytes, so the grid and the workspaces are the extra
            extra = sum(by[n].bytes for n in ("attention projections on the int4-b32 grid", "int4 attention workspaces")
                        if n in by)
            out.append(f"attention on the int4-b32 grid (attn_int4): {extra / GiB:+.2f} GiB against bf16 attention once a "
                       "prompt is served, because each projection keeps a bf16 copy for calls over 16 rows; a "
                       "decode-speed lever, not a memory one")
        if not (s.get("exp_int4") or s.get("attn_int4")) and not {"exp_int4", "attn_int4"} & set(constraints.fixed):
            out.append("int4 levers off: exp_int4 and attn_int4 change the served weights (round-to-nearest int4), so "
                       "the planner prices them only when fixed")
        out.append("not planned yet: a measured routing profile for the solver, decode speed")
        from experts4bit_qlora.serve_recipe import ServeSetup

        env = " ".join(f"{k}={v}" for k, v in sorted(_serve_setup(ServeSetup, s).to_env().items()))
        out.append(f"to serve it: {env} python -m experts4bit_qlora.serve_paged (with the model's arena and calibration)")
        return out
    resident = [c for c in feasible + infeasible if c.setup.get("expert_residency") == "device"
                and not any("cannot" in r or "refus" in r for r in c.rejected)]
    if s["expert_residency"] == "device":
        out.append(f"experts resident on the device: {sel.device_bytes / GiB:.2f} GiB estimated + "
                   f"{budget['headroom'] / GiB:.2f} GiB headroom fits the {budget['device'] / GiB:.2f} GiB budget")
    else:
        need = min((c.device_bytes for c in resident), default=None)
        out.append("experts host-backed (pinned, streamed one layer at a time): "
                   + (f"the cheapest resident setup needs {need / GiB:.2f} GiB + headroom, over the "
                      f"{budget['device'] / GiB:.2f} GiB budget" if need is not None and constraints.objective == "speed"
                      else f"chosen by objective {constraints.objective}"))
    out.append(f"expert kernel {s['expert_kernel']}: {describe_kernel(s, status)}")
    if constraints.objective == "speed":
        out += [f"ordering, {axis}: {ev}" for axis, ev in SPEED_EVIDENCE.items()]
    if s["attn_4bit"]:
        out.append("attention stored in NF4 because no bf16-attention setup fit")
    if constraints.fixed:
        out.append(f"fixed by the caller: {constraints.fixed}")
    out.append("activation policy: every decoder layer checkpointed (recomputed in backward); "
               + (f"MoE activations kept in {s['keep_moe_layers']} layers" if s.get("keep_moe_layers")
                  else "no MoE activations kept (keep_moe_layers is a dial the planner does not choose yet)"))
    return out


def policy_notes(topology, constraints) -> list:
    """Choices this backend's candidate policy made on the caller's behalf, in words."""
    attn = topology.attention
    if (attn is None or attn.count == 0) and "train_attention" not in constraints.fixed:
        why = topology.provenance.get("attention", "no q_proj/k_proj/o_proj linears found")
        return [f"attention LoRA off: this model's attention cannot take the adapter ({why}); experts still train"]
    return []


def _serve_candidates(workload, constraints, status):
    from experts4bit_qlora.serve_recipe import ServeSetup

    fixed = dict(constraints.fixed)
    unknown = set(fixed) - set(ServeSetup.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown serve setup fields fixed by the caller: {sorted(unknown)}")
    if not status.kernels.get("paged_fp8", (False,))[0]:
        return []
    base = {**ServeSetup().to_dict(), "max_seqs": workload.concurrency or 1,
            "max_tokens_per_seq": workload.context_len or 4096}
    try:                                    # the server's own rule (experts4bit-qlora#1234) where it has one
        from experts4bit_qlora.serve_recipe import usable_buckets as _usable
    except ImportError:
        _usable = usable_buckets
    base["buckets"] = _usable(base["max_seqs"], base["buckets"])
    if "prefill_graph" in base:
        base["prefill_graph"] = "0"                # its pool is not priced: a plan bounds memory by what it priced
    residency = constraints.expert_residency or ("device", "host")
    out = []
    if "device" in residency and fixed.get("placement", "all-vram") == "all-vram":
        can_graph = status.kernels.get("paged_graphs", (False,))[0]
        graphs = [fixed["graphs"]] if "graphs" in fixed else ([True, False] if can_graph else [False])
        out += [{**base, **fixed, "placement": "all-vram", "graphs": g} for g in graphs if can_graph or not g]
    solver_ok = "vram_gb" in ServeSetup.__dataclass_fields__ and status.kernels.get("cpu_tier", (False,))[0]
    if solver_ok and fixed.get("placement", "solver") == "solver" and ("host" in residency or "placement" in fixed):
        # the tiers' budgets are filled by the planner (fill_knobs) unless the caller fixed them
        out.append({**base, "graphs": False, "vram_gb": 0.0, "dram_gb": 0.0, "hot_rows": "auto", **fixed,
                    "placement": "solver"})
    return out


def usable_buckets(max_seqs: int, buckets) -> tuple:
    """The decode-graph buckets a server of ``max_seqs`` sequences can use. A decode step never carries more rows than
    sequences, and the runner pads a step to the next bucket, so a bucket above ``max_seqs`` never runs. It still
    costs what every bucket costs: a captured graph, and the scratch slots the largest bucket sizes (a hybrid model's
    linear-attention state is per slot: ~62 MiB each on Qwen3.6-35B). The planner keeps the buckets below
    ``max_seqs`` and ends at ``max_seqs`` itself, capped at the largest bucket (a wider step runs in chunks of it)."""
    keep = {int(b) for b in buckets if int(b) < max_seqs}
    return tuple(sorted(keep | {min(int(max_seqs), max(int(b) for b in buckets))}))


def _serve_setup(cls, setup: dict):
    """A ServeSetup from a plan's or a receipt's setup dict: fields it does not carry (a receipt's torch threads) are
    dropped, and values a receipt recorded from the environment as text are read back as numbers."""
    kinds = {"vram_gb": float, "dram_gb": float, "hot_rows": int, "max_seqs": int, "max_tokens_per_seq": int,
             "chunk_tokens": int}
    known = cls.__dataclass_fields__
    vals = {k: (kinds[k](v) if k in kinds else v) for k, v in setup.items() if k in known}
    if "buckets" in vals:
        vals["buckets"] = tuple(int(b) for b in (vals["buckets"].split(",") if isinstance(vals["buckets"], str)
                                                 else vals["buckets"]))
    for k in ("graphs", "exp_int4", "attn_int4"):
        if isinstance(vals.get(k), str):
            vals[k] = vals[k] not in ("0", "false", "False")
    return cls(**vals)


def resolve(topology, setup: dict, workload=None) -> dict:
    """Concrete values for the fields a candidate leaves to the mechanism: ``hot_rows="auto"`` becomes the fewest cold
    rows the tier split can serve with (experts4bit-qlora's ``min_hot_rows``, at least 1), or the server's default
    when the installed package cannot say."""
    if setup.get("hot_rows") != "auto":
        return setup
    try:
        from experts4bit_qlora.serve_recipe import ServeSetup, min_hot_rows
    except ImportError:
        return {**setup, "hot_rows": 64}
    probe = _serve_setup(ServeSetup, {**setup, "hot_rows": 1})
    return {**setup, "hot_rows": max(1, min_hot_rows(topology, probe))}


def fill_knobs(topology, setup: dict, workload) -> list:
    """Setup fields the planner should raise to the largest value its budget allows, in order: ``(field, side, upper)``.
    Under the solver, the VRAM tier fills the device budget and then the DRAM tier fills the host budget; neither can
    usefully exceed the whole expert slab."""
    if setup.get("placement") != "solver":
        return []
    slab = next((ln[2] for ln in estimate(topology, {**setup, "placement": "all-vram", "graphs": False}, workload)[0]
                 if ln[0].startswith("frozen expert stacks")), 0)
    hi = slab / GiB
    return [(k, side, hi) for k, side in (("vram_gb", "device"), ("dram_gb", "host"))]


def estimate(topology, setup: dict, workload):
    """``(lines, unmodelled, refusals)`` for one setup; ``lines`` are ``(name, where, bytes, basis, detail)`` tuples."""
    if workload.kind == "serve":
        from experts4bit_qlora.serve_recipe import ServeSetup, estimate_serve_footprint

        fp = estimate_serve_footprint(topology, _serve_setup(ServeSetup, resolve(topology, setup, workload)))
        return [(i.name, i.where, i.bytes, i.basis, i.detail) for i in fp.items], fp.unmodelled, fp.refusals
    from experts4bit_qlora.recipe import QLoRASetup, estimate_qlora_footprint

    fp = estimate_qlora_footprint(topology, QLoRASetup(**setup), tokens_per_microbatch=workload.tokens_per_microbatch,
                                  optimizer=workload.optimizer)
    lines = [(i.name, i.where, i.bytes, i.basis, i.detail) for i in fp.items]
    return lines, fp.unmodelled, fp.refusals


def speed_rank(setup: dict) -> tuple:
    if "max_seqs" in setup:                     # serve: all experts in VRAM first; graphs replay the decode step
        return (0 if setup.get("placement") == "all-vram" else 1, 0 if setup.get("graphs") else 1)
    return ((0 if setup["expert_residency"] == "device" else 1),
            (0 if setup["expert_kernel"] == "grouped_nf4" else 1),
            (1 if setup["attn_4bit"] else 0),
            (0 if setup.get("pin", True) else 1),
            -int(setup.get("keep_moe_layers") or 0))


def describe_kernel(setup: dict, status: BackendStatus) -> str:
    k = setup["expert_kernel"]
    ok, why = status.kernels.get(k, (False, "unknown"))
    if k == "grouped_nf4":
        return f"grouped-nf4-gemm {status.versions.get('grouped-nf4-gemm')} ({why})"
    return f"bitsandbytes {status.versions.get('bitsandbytes')} per-expert dequant loop"
