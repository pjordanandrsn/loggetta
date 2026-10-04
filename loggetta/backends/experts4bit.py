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
WORKLOADS = ("train",)

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
    return BackendStatus(True, "", versions, kernels)


def candidates(topology, workload, constraints, status: BackendStatus) -> list:
    """Setups worth estimating, as dicts of ``QLoRASetup`` fields. Fields in ``constraints.fixed`` are not varied."""
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


def policy_notes(topology, constraints) -> list:
    """Choices this backend's candidate policy made on the caller's behalf, in words."""
    attn = topology.attention
    if (attn is None or attn.count == 0) and "train_attention" not in constraints.fixed:
        why = topology.provenance.get("attention", "no q_proj/k_proj/o_proj linears found")
        return [f"attention LoRA off: this model's attention cannot take the adapter ({why}); experts still train"]
    return []


def estimate(topology, setup: dict, workload):
    """``(lines, unmodelled, refusals)`` for one setup; ``lines`` are ``(name, where, bytes, basis, detail)`` tuples."""
    from experts4bit_qlora.recipe import QLoRASetup, estimate_qlora_footprint

    fp = estimate_qlora_footprint(topology, QLoRASetup(**setup), tokens_per_microbatch=workload.tokens_per_microbatch,
                                  optimizer=workload.optimizer)
    lines = [(i.name, i.where, i.bytes, i.basis, i.detail) for i in fp.items]
    return lines, fp.unmodelled, fp.refusals


def speed_rank(setup: dict) -> tuple:
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
