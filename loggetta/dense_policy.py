"""Explicit registered dense memory hypotheses. No observation import or implicit default."""
from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path

from .plan import MemoryLine

POLICY_FILE = Path(__file__).with_name("dense_dq10_policy.json")


def policy():
    raw = POLICY_FILE.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def runtime_versions():
    try:
        return {name: metadata.version(name) for name in policy()[0]["runtime"]}
    except metadata.PackageNotFoundError as error:
        raise ValueError("DQ10 runtime package is missing") from error


def selection(topology, gpu, setup, workload, constraints):
    """Return the frozen hypothesis only for the registered subject/card/runtime/recipe."""
    payload, digest = policy()
    if constraints.dense_reserve_policy != payload["name"]:
        raise ValueError("unknown dense reserve policy; only explicit dq10 is registered")
    if constraints.allocator_profile != payload["allocator_profile"]:
        raise ValueError("DQ10 requires an explicit default allocator profile")
    if workload.kind != "train" or workload.seq_len not in payload["sequence_rungs"]:
        raise ValueError("DQ10 applies only to its registered training sequence rungs")
    if (workload.micro_batch != 1 or workload.grad_accum != 1 or workload.optimizer != "adamw"
            or workload.steps != 2 or workload.learning_rate != 2e-4 or workload.lr_schedule != "constant"
            or workload.warmup_steps not in (None, 0) or workload.epochs is not None):
        raise ValueError("DQ10 training workload differs from its registered recipe")
    if setup.get("placement") not in payload["hypotheses"] or any(
            setup.get(k) != v for k, v in payload["recipe"].items()):
        raise ValueError("DQ10 dense setup differs from its registered recipe")
    hardware = payload["hardware"]
    if (gpu is None or gpu.name != hardware["name"] or gpu.driver.value != hardware["driver"]
            or not isinstance(gpu.memory_total.value, int)
            or not hardware["memory_min"] <= gpu.memory_total.value <= hardware["memory_max"]):
        raise ValueError("DQ10 GPU/card capacity/driver differs from its registered scope")
    if runtime_versions() != payload["runtime"]:
        raise ValueError("DQ10 runtime differs from its registered pins")
    config = Path(topology.model) / "config.json"
    if not config.is_file() or hashlib.sha256(config.read_bytes()).hexdigest() not in payload["config_sha256"]:
        raise ValueError("DQ10 requires an unchanged registered local subject config")
    return payload["hypotheses"][setup["placement"]], digest


def lines(topology, gpu, setup, workload, constraints, allocator_bytes):
    hypothesis, digest = selection(topology, gpu, setup, workload, constraints)
    n, d = hypothesis["reserve_numerator"], hypothesis["reserve_denominator"]
    charge = (allocator_bytes * n + d - 1) // d
    detail = f"DQ10 prospective hypothesis; policy SHA256 {digest}; no calibration or capacity licence"
    return (MemoryLine("allocator reserve (DQ10 training peak slack)", "device", charge, "inferred",
                       detail + f"; ceil(allocator estimate * {n}/{d})"),
            MemoryLine("CUDA context + library workspaces (DQ10)", "device", hypothesis["context_bytes"],
                       "inferred", detail + "; maximum fitting-row sampled driver peak minus reserved peak"))


def validate_plan(plan):
    """Recheck a serialized policy plan and fresh hardware before tokenizer/weights/optimizer loading."""
    from .backends import dense
    from .hardware import probe

    tagged = tuple(line for line in plan.selected.lines if "DQ10" in line.name)
    if plan.constraints.dense_reserve_policy is None:
        if tagged:
            raise ValueError("DQ10 pricing lines require explicit policy selection")
        return None
    if plan.selected.backend != dense.NAME:
        raise ValueError("DQ10 policy requires the dense backend")
    topology = dense.describe(plan.model["model"], revision=plan.model.get("revision"))
    gpu = probe().gpu(plan.constraints.device)
    allocator = sum(line.bytes for line in plan.selected.lines if line.where == "device"
                    and not line.name.startswith(("CUDA context", "allocator reserve")))
    expected = lines(topology, gpu, plan.selected.setup, plan.workload, plan.constraints, allocator)
    if tagged != expected or plan.selected.device_bytes != sum(
            line.bytes for line in plan.selected.lines if line.where == "device"):
        raise ValueError("DQ10 plan pricing or frozen policy changed; plan again")
    return {"name": "dq10", "sha256": policy()[1], "basis": "inferred", "licensed": False,
            "allocator_profile": plan.constraints.allocator_profile}
