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


def isolation_refusal(topology) -> str | None:
    """Why resetting positions cannot isolate packed examples in this model, or None. Positions isolate attention;
    a layer that mixes tokens through a recurrent state (state-space, convolution, linear attention) carries one
    example into the next regardless."""
    attn = topology.attention
    if attn is None:
        return ("this model's attention could not be described, so isolating packed examples cannot be checked "
                f"({topology.provenance.get('attention', 'no description')})")
    mixing = topology.n_layers - attn.layers
    if mixing > 0:
        return (f"{mixing} of {topology.n_layers} decoder layers mix tokens through a state (state-space, convolution "
                "or linear attention): resetting positions does not isolate packed examples there")
    return None


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


def single_stream_levers(topology) -> str | None:
    """What a default ``serve_paged`` decodes one sequence with on this model's family, as the INSTALLED
    experts4bit-qlora resolves it (its family-scoped B=1 fused stack, lane P115: ``serve_paged.resolve_fusion_modes``
    and the reads it names), never a table kept here. None when experts4bit-qlora's serving module is not importable."""
    try:
        from experts4bit_qlora import serve_paged as sp
    except ImportError:
        return None
    family = getattr(topology, "model_type", None)
    resolve = getattr(sp, "resolve_fusion_modes", None)
    if resolve is None or not hasattr(sp, "FUSION_KNOBS") or not hasattr(sp, "FUSION_UNSET"):
        return ("the B=1 fused stack (fused q/k/v and three glue folds) is off unless set: this experts4bit-qlora has "
                "no family-scoped fusion default")
    modes, _sources = resolve({k: sp.FUSION_UNSET for k in sp.FUSION_KNOBS}, family)
    on = sorted(k for k, v in modes.items() if v != "0")
    if on:
        why = getattr(sp, "FUSION_DEFAULT_FAMILIES", {}).get(family, "")
        return (f"the B=1 fused stack (fused q/k/v and three glue folds) is on by default for {family}"
                + (f" ({why})" if why else "") + "; " + ", ".join(f"{k}=0" for k in on) + " turns it off")
    why = getattr(sp, "FUSION_UNLICENSED", {}).get(family, "no registered read")
    return (f"the B=1 fused stack (fused q/k/v and three glue folds) is off by default for {family}: {why}; setting "
            "its knobs to auto applies it where the structure matches, without that read")


def performance(topology, setup: dict, workload, gpu, observations, status) -> dict | None:
    """A single-stream decode figure measured on exactly this setup, or None: then nothing is shown, and the planner's
    own statement (no performance prediction) stands.

    Only a measurement of the same thing counts: a serve receipt of this model at one sequence, on this GPU name and
    driver, with this setup and the same experts4bit-qlora and grouped-nf4-gemm versions (defaults that move decode
    speed change between versions), that recorded its decode step (``measured.decode_step_ms_b1``, from the server's
    step trace). Absolute decode times do not travel between hosts, so nothing is extrapolated from another GPU."""
    if workload.kind != "serve" or setup.get("max_seqs") != 1:
        return None
    vers = getattr(status, "versions", None) or {}
    want = {d: vers.get(d) for d in ("experts4bit-qlora", "grouped-nf4-gemm")}
    gname, gdrv = (gpu.name, gpu.driver.value) if gpu is not None else (None, None)
    match = None
    for obs in observations:
        m = obs.get("measured") or {}
        g = (obs.get("hardware") or {}).get("gpu") or {}
        got = ((obs.get("provenance") or {}).get("versions") or {})
        if (obs.get("status") == "OK" and (obs.get("workload") or {}).get("kind") == "serve"
                and (obs.get("workload") or {}).get("concurrency") == 1
                and (obs.get("model") or {}).get("model") == topology.model and obs.get("setup") == setup
                and g.get("name") == gname and g.get("driver") == gdrv
                and all(got.get(d) == v for d, v in want.items())
                and isinstance(m.get("decode_step_ms_b1"), (int, float)) and m["decode_step_ms_b1"] > 0):
            match = obs
            break
    if match is None:
        return None                 # nothing is shown: no figure is interpolated or borrowed
    m = match["measured"]
    ms = float(m["decode_step_ms_b1"])
    # a decode step grows with the KV position it attends over: say where this one was measured
    w = match.get("workload") or {}
    pt, nt = w.get("prompt_tokens"), w.get("new_tokens")
    where = (f" after a {pt}-token prompt over {nt} new tokens" if isinstance(pt, int) and isinstance(nt, int)
             else "")
    return {"statement": (f"measured on this setup: one sequence decodes at {1000 / ms:.0f} tokens/s ({ms:.2f} ms per "
                          f"decode step, median of {m.get('decode_steps_b1', '?')} steps{where}; receipt "
                          f"{match.get('run_id')}, {gname}, experts4bit-qlora {want['experts4bit-qlora']}). Not an "
                          "estimate for another GPU, driver, setup, version or context length"),
            "estimate": {"basis": "measured-same-setup", "decode_step_ms_b1": ms,
                         "tokens_per_s_b1": round(1000 / ms, 1), "decode_device_ms_b1": m.get("decode_device_ms_b1"),
                         "prompt_tokens": pt, "new_tokens": nt, "receipt": match.get("run_id")}}


def explain(sel, feasible, infeasible, budget, status, constraints, workload, topology=None) -> list:
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
                       "(SV1 measured +0.56 GiB at Qwen3-30B with NF4 experts), so the plan's memory would not bound the process; fix "
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
        if s.get("max_seqs") == 1 and topology is not None:
            levers = single_stream_levers(topology)
            if levers:
                out.append(f"single stream: {levers}")
        out.append("not planned yet: a measured routing profile for the solver; decode speed beyond a receipt of "
                   "this same setup (see performance)")
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


def estimate_env() -> dict | None:
    """The environment switches experts4bit-qlora's estimate reads, with their values in this process, as
    experts4bit-qlora reports them (``recipe.estimate_env``), or None where the installed release has no such accessor.
    A plan records it; ``execute`` compares it with the running process's, since a run that sees another setting builds
    a different model than the one priced (``E4B_CHUNKED_LM_LOSS`` changes the loss branch)."""
    try:
        from experts4bit_qlora.recipe import estimate_env as accessor
    except ImportError:
        return None
    return dict(accessor())


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
    unmodelled = fp.unmodelled
    if not fp.refusals and setup.get("expert_kernel") == "grouped_nf4":
        if not e4b_prices_gnf4_backward():
            line = _gnf4_backward_line(topology, setup, workload.tokens_per_microbatch, lines)
            if line is not None:
                lines.append(line)
        unmodelled = tuple(unmodelled) + GNF4_UNMODELLED
    return lines, unmodelled, fp.refusals


# --- grouped_nf4 training: the MoE layer backward's working set (#43) -------------------------------------------------
# experts4bit-qlora's activation item is ``boundaries + max(logits, 2 x one layer)``, whose layer term counts T x top_k
# routed rows and does not depend on the kernel. With grouped-nf4-gemm the training peak moves into an MoE layer's
# backward (#44, OLMoE on the A2000), where the kernel's padded LoRA delta and its fused workspaces are live.
# The routing that sizes the padded delta is data-dependent, so it is priced as a BOUND in shape, never a constant.
#
# grouped-nf4-gemm exposes no public sizing or route helper, so its rules are mirrored here from kernel/nf4_qlora.py.
# tests/test_gnf4_training_terms.py pins each one against the installed package: a kernel change fails that test
# instead of drifting silently away from this estimate.

#: ``_PAD_BYTES_LIMIT``: the ``auto`` route takes the per-expert loop (nothing padded) when the padded block,
#: ``G x widest x (K + N)`` at the activations' itemsize, would exceed this
GNF4_PAD_BYTES_LIMIT = 2 * 2 ** 30
#: ``_PAD_BUCKETS_AUTO_MIN_ROWS``: a call with at least this many routed rows (T x top_k) pads by buckets (0.42.0+)
GNF4_PAD_BUCKETS_MIN_ROWS = 16384
#: ``_PAD_BUCKET_RATIO``: within a bucket the widest group has at most this many times the narrowest's rows, so the
#: buckets hold at most this many times the routed rows
GNF4_PAD_BUCKET_RATIO = 2
#: ``T x top_k x first_out`` bf16 buffers live at the grouped kernel's backward peak (#44: fused_experts_train_
#: forward, fused_grouped_lora, gemm_4bit_grouped, _scaled and nf4_qlora's forward, 32 MiB each on OLMoE at T=1024)
GNF4_FUSED_WORKSPACES = 5
GNF4_UNMODELLED = (
    "grouped_nf4 LoRA-delta overrides read at run time and not by this estimate: NF4_QLORA_LORA_PATH, "
    "NF4_QLORA_PAD_BYTES_LIMIT, NF4_QLORA_PAD_WASTE_LIMIT, NF4_QLORA_PAD_BUCKETS(_MIN_ROWS), "
    "NF4_QLORA_PAD_BUCKETS_LADDER, NF4_QLORA_SINGLE_LADDER, NF4_QLORA_COMPACT_DELTA",
)


def e4b_prices_gnf4_backward() -> bool:
    """Whether the installed experts4bit-qlora prices the grouped_nf4 MoE backward itself (experts4bit-qlora#1526: a
    branch of its ``activations`` item). Then this module adds no line of its own; for an older release it still does.
    Feature-detected, not version-compared."""
    try:
        from experts4bit_qlora import recipe
    except ImportError:
        return False
    return hasattr(recipe, "grouped_nf4_padded_rows_bound")


def _ladder_up(n: int) -> int:
    """grouped-nf4-gemm's ``_ladder_up``: the smallest rung >= ``n``, every integer up to 4 and then four rungs per
    octave (``{4, 5, 6, 7} x 2**k``), so a rung is under 1.25 x ``n``."""
    n = int(n)
    if n <= 4:
        return max(n, 0)
    step = 1 << (n.bit_length() - 3)
    return -(-n // step) * step


def gnf4_padded_rows_bound(*, n_experts: int, top_k: int, tokens: int, hidden: int, first_out: int,
                           intermediate: int, adapter_dtype: str) -> tuple:
    """``(rows, how)``: an upper bound on the padded LoRA delta's rows in one MoE layer pass of ``tokens`` tokens.

    The single padded block is ``G x widest`` rows (``_lora_delta_padded``): ``G`` routed-to experts, at most
    ``min(E, T x top_k)``, each padded to the hottest one's rows, at most ``T`` (an expert sees a token once). Then:

    * the ``auto`` route pads only while ``G x widest x (K + N) x 2`` stays under ``GNF4_PAD_BYTES_LIMIT``; the smaller
      ``K + N`` of the two projections (gate_up: H + first_out; down: I + H) gives the widest cap any padded call has;
    * fp32 adapters take the single-block ladder (``NF4_QLORA_SINGLE_LADDER=auto``, grouped-nf4-gemm 0.44.0): ``G`` and
      ``widest`` each rounded up to a rung, at most 1.5625 x the single block. Bounding it on older releases, which
      have no ladder, only overstates;
    * a call with at least ``GNF4_PAD_BUCKETS_MIN_ROWS`` routed rows pads by buckets instead, at most
      ``GNF4_PAD_BUCKET_RATIO x T x top_k`` rows.
    """
    routed = tokens * top_k
    if routed >= GNF4_PAD_BUCKETS_MIN_ROWS:
        return (GNF4_PAD_BUCKET_RATIO * routed,
                f"bucketed (T x top_k = {routed} >= {GNF4_PAD_BUCKETS_MIN_ROWS}): "
                f"at most {GNF4_PAD_BUCKET_RATIO} x T x top_k rows")
    g, w = min(n_experts, routed), tokens
    cap = GNF4_PAD_BYTES_LIMIT // (min(hidden + first_out, intermediate + hidden) * 2)
    single = min(g * w, cap)
    how = f"single block: min(E, T x top_k) x T = {g} x {w}" + (f", capped at {cap} by the pad route" if cap < g * w
                                                              else "")
    if adapter_dtype == "fp32":
        rows = min(_ladder_up(g) * _ladder_up(w), -(-single * 25 // 16))
        return rows, how + f", on the fp32 single-block ladder: {rows}"
    return single, how


def _gnf4_backward_line(topology, setup: dict, tokens: int, lines: list):
    """The grouped kernel's MoE-backward working set above experts4bit-qlora's activation item, or None if that item
    already covers it. The backward branch is ``boundaries + padded LoRA delta + fused workspaces``; the activation
    item is ``boundaries + max(logits, 2 x layer)``, so the line is what the branch exceeds it by. The expert LoRA
    gradients live there are already priced (the 'adapter gradients' line)."""
    act = next((ln[2] for ln in lines if ln[0] == "activations"), None)
    k = topology.top_k or 0
    if act is None or not k or not topology.expert_stacks:
        return None
    ab = 4 if setup.get("adapter_dtype") == "fp32" else 2
    T, H = int(tokens), topology.hidden_size
    boundaries = topology.n_layers * T * H * 2
    best = None
    for st in topology.expert_stacks:
        first_out = _first_out(st)
        ws = GNF4_FUSED_WORKSPACES * T * k * first_out * 2
        rows, how = 0, "experts not trained: no LoRA delta"
        if setup.get("train_experts", True):
            rows, how = gnf4_padded_rows_bound(n_experts=st.n_experts, top_k=k, tokens=T, hidden=st.hidden,
                                               first_out=first_out, intermediate=st.intermediate,
                                               adapter_dtype=setup.get("adapter_dtype", "bf16"))
        padded = rows * (st.hidden + first_out + st.intermediate) * ab
        if best is None or padded + ws > best[0] + best[1]:
            best = (padded, ws, rows, how, st, first_out)
    padded, ws, rows, how, st, first_out = best
    extra = boundaries + padded + ws - act
    if extra <= 0:
        return None
    return ("grouped_nf4 MoE backward (above the activation item)", "device", extra, "heuristic",
            f"boundaries {boundaries / 2**20:.1f} MiB + padded LoRA delta {padded / 2**20:.1f} MiB "
            f"({rows} rows x (H + first_out + I = {st.hidden} + {first_out} + {st.intermediate}) x {ab} B; {how}) "
            f"+ fused workspaces {ws / 2**20:.1f} MiB ({GNF4_FUSED_WORKSPACES} x T x top_k x first_out bf16) "
            f"- the activation item {act / 2**20:.1f} MiB; a bound in shape (#43, #44)")


def _first_out(stack) -> int:
    """Per-expert output width of the first projection (2I gated, I not), as experts4bit-qlora's recipe reads it."""
    n = 1
    for d in stack.first_shape:
        n *= d
    return n // (stack.n_experts * stack.hidden)


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
