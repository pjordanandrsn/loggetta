"""The experts4bit-qlora backend: QLoRA training of fused-MoE models in 4-bit, optionally with host-resident experts.

Everything this module knows about memory, structure or mechanism it ASKS the package that owns it:

* topology and admission -- ``experts4bit_qlora.arch.topology.describe_moe``;
* setup validity and the itemized footprint -- ``experts4bit_qlora.recipe`` (``setup_refusals``,
  ``estimate_qlora_footprint``), which prices the same modules ``prepare_qlora_training`` builds;
* the grouped kernel's training route on a device -- ``grouped-nf4-gemm``'s ``nf4_route.route_for``.

What is decided HERE is policy: which setups are worth considering, and in what order a speed objective tries them.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

NAME = "experts4bit"
WORKLOADS = ("train", "serve")
#: which probed kernels each workload uses: the planner reports only those as unusable
KERNELS_FOR = {"train": ("grouped_nf4", "reference"), "serve": ("paged_fp8", "paged_graphs")}
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
        return (f"serve {setup['placement']}, fp8 paged KV, {setup['max_seqs']} seqs x {setup['max_tokens_per_seq']} "
                f"tokens{', decode graphs' if setup.get('graphs') else ''}"
                f"{', prefill graph ' + str(setup['prefill_graph']) if setup.get('prefill_graph') not in (None, '0') else ''}")
    s = setup
    return (f"experts on {s.get('expert_residency')}, {s.get('expert_kernel')} kernel"
            + (", NF4 attention" if s.get("attn_4bit") else "")
            + (", pageable" if s.get("expert_residency") == "host" and not s.get("pin", True) else "")
            + (f", keep {s['keep_moe_layers']} MoE layers" if s.get("keep_moe_layers") else ""))


def explain(sel, feasible, infeasible, budget, status, constraints, workload) -> list:
    """Why this candidate, in words: the backend knows its own setup fields; the planner does not read them."""
    s, out = sel.setup, []
    if workload.kind == "serve":
        out.append(f"all experts resident (placement all-vram): {sel.device_bytes / GiB:.2f} GiB estimated + "
                   f"{budget['headroom'] / GiB:.2f} GiB headroom fits the {budget['device'] / GiB:.2f} GiB budget")
        kv = next((ln for ln in sel.lines if ln.name == "FP8 paged KV pool"), None)
        if kv:
            out.append(f"KV pool {kv.bytes / GiB:.2f} GiB for {s['max_seqs']} sequences x {s['max_tokens_per_seq']} "
                       "tokens; it scales linearly in both")
        out.append(f"kernels: {status.kernels.get('paged_fp8', (None, ''))[1]}; grouped NF4 / int4 decode routes as "
                   "serve_paged resolves them (reported at /health)")
        if constraints.objective == "speed":
            out += [f"ordering, {k}: {v}" for k, v in SERVE_EVIDENCE.items()]
        if s.get("prefill_graph") == "0" and "prefill_graph" not in constraints.fixed:
            out.append("first-chunk prefill graph off (the server's default is auto): its private pool is not priced "
                       "(SC2b measured +3.3 GiB at Qwen3-30B), so the plan's memory would not bound the process; fix "
                       "prefill_graph=auto to let the server engage it when that much is free")
        out.append("not planned yet: the solver's VRAM/DRAM/NVMe tiers, int4 expert stores, decode speed")
        from experts4bit_qlora.serve_recipe import ServeSetup

        env = " ".join(f"{k}={v}" for k, v in sorted(ServeSetup(**s).to_env().items()))
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
    if constraints.expert_residency is not None and "device" not in constraints.expert_residency:
        base["placement"] = "solver"               # the tiered placement, which the estimate refuses in words
    if "prefill_graph" in base:
        base["prefill_graph"] = "0"                # its pool is not priced: a plan bounds memory by what it priced
    can_graph = status.kernels.get("paged_graphs", (False,))[0]
    graphs = [fixed["graphs"]] if "graphs" in fixed else ([True, False] if can_graph else [False])
    return [{**base, **fixed, "graphs": g} for g in graphs if can_graph or not g]


def estimate(topology, setup: dict, workload):
    """``(lines, unmodelled, refusals)`` for one setup; ``lines`` are ``(name, where, bytes, basis, detail)`` tuples."""
    if workload.kind == "serve":
        from experts4bit_qlora.serve_recipe import ServeSetup, estimate_serve_footprint

        fp = estimate_serve_footprint(topology, ServeSetup(**setup))
        return [(i.name, i.where, i.bytes, i.basis, i.detail) for i in fp.items], fp.unmodelled, fp.refusals
    from experts4bit_qlora.recipe import QLoRASetup, estimate_qlora_footprint

    fp = estimate_qlora_footprint(topology, QLoRASetup(**setup), tokens_per_microbatch=workload.tokens_per_microbatch,
                                  optimizer=workload.optimizer)
    lines = [(i.name, i.where, i.bytes, i.basis, i.detail) for i in fp.items]
    return lines, fp.unmodelled, fp.refusals


def speed_rank(setup: dict) -> tuple:
    if "max_seqs" in setup:                     # serve: graphs replay the decode step instead of launching it
        return (0 if setup.get("graphs") else 1,)
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
