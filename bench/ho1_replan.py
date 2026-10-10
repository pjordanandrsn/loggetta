"""HO1 plans and grading, per bench/HO1-PREREG.md (registered in loggetta #59).

    python bench/ho1_replan.py --e4b E4B_CHECKOUT --rows evidence/2026-10-10-heldout-cards/rows.json \
        --out evidence/2026-10-10-heldout-cards [--a2000-evidence evidence]

One plan per distinct setup (card, model, tokens, kernel, placement, adapter dtype; attn_4bit, rank and alpha fixed
as recorded), with no receipts on file, and again with this repository's committed A2000 training receipts on file
as the reported variant. Card memory comes from each run's committed ``forensics.txt`` (its nvidia-smi line); the
rest of the card and host comes from the arm receipt. Configs come from the Hugging Face Hub, without weights.
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import os
import re
import subprocess
import tarfile

GiB, MiB = 1 << 30, 1 << 20


def setup_key(r):
    s = r["setup"]
    return (r["gpu"], r["model"], r["seq"], r["micro_batch"], s["expert_kernel"], s["expert_residency"],
            s["adapter_dtype"], s["attn_4bit"], r.get("r"), r.get("alpha"), r.get("accum"),
            "adamw_8bit" if str(r.get("optimizer") or "").startswith("adamw_8bit") else "adamw")


def card_total(archive, arm_receipt, gpu):
    f = os.path.join(archive, os.path.dirname(arm_receipt), "forensics.txt")
    for ln in open(f):
        m = re.match(r"\s*" + re.escape(gpu) + r",\s*(\d+) MiB", ln)
        if m:
            return int(m.group(1)) * MiB
    raise ValueError(f"no nvidia-smi memory line for {gpu} in {f}")


def hardware(archive, row):
    from loggetta.hardware import GPU, Fact, HardwareProfile, Host

    arm = json.load(open(os.path.join(archive, row["arm_receipt"])))
    env, h = arm["env"], arm["env"].get("host") or {}
    f = lambda v: Fact(v, "stated")  # noqa: E731
    i = lambda k: int(h[k]) if str(h.get(k, "")).isdigit() else None  # noqa: E731
    total = card_total(archive, row["arm_receipt"], env["gpu"])
    gpu = GPU(index=0, vendor="nvidia", name=env["gpu"], uuid=None, compute_capability=f(tuple(env["cap"])),
              memory_total=f(total), memory_free=f(total), driver=f(h.get("driver_version")),
              pcie_gen_max=f(i("pcie.link.gen.max")), pcie_width_max=f(i("pcie.link.width.max")),
              pcie_gen_current=f(i("pcie.link.gen.current")), pcie_width_current=f(i("pcie.link.width.current")))
    ram = int(float(h["host_mem_gib"]) * GiB) if h.get("host_mem_gib") else None
    host = Host(cpu_model=f(h.get("cpu")), cpus=f(h.get("cpu_threads")), memory_total=f(ram),
                memory_available=f(ram), memory_limit=f(ram))
    return HardwareProfile(gpus=(gpu,), host=host, platform="Linux"), total


def plan_one(topology, hw, k, observations):
    from loggetta import Constraints, Workload, plan

    _, _, seq, mb, kernel, residency, dtype, attn4, r, alpha, accum, opt = k[:12]
    wl = Workload(kind="train", seq_len=seq, micro_batch=mb, grad_accum=accum or 1, optimizer=opt)
    fixed = {"expert_kernel": kernel, "expert_residency": residency, "adapter_dtype": dtype, "attn_4bit": attn4}
    if r:
        fixed["r"] = r
    if alpha:
        fixed["alpha"] = alpha
    p = plan(topology, hw, wl, Constraints(fixed=fixed), observations=observations)
    c = p.selected or (p.alternatives[0] if p.alternatives else None)
    if c is None:
        return {"status": p.status, "plan_total_bytes": None}
    line = lambda prefix: next((ln for ln in c.lines if ln.name.startswith(prefix)), None)  # noqa: E731
    res, ctx = line("allocator reserve"), line("CUDA context")
    return {"status": p.status, "refused": p.selected is None, "plan_total_bytes": c.device_bytes,
            "allocator_estimate_bytes": c.device_bytes - (res.bytes if res else 0) - (ctx.bytes if ctx else 0),
            "reserve": f"{res.basis} {res.bytes / GiB:.3f}" if res else "none",
            "context": f"{ctx.basis} {ctx.bytes / GiB:.3f}" if ctx else "none"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e4b", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--a2000-evidence", help="this repository's evidence/ (the A2000-on-file variant)")
    a = ap.parse_args()
    reg = json.load(open(a.rows))
    archive = os.path.join(a.out, ".e4b-archive")
    os.makedirs(archive, exist_ok=True)
    blob = subprocess.run(["git", "-C", a.e4b, "archive", reg["e4b_commit"], *reg["dirs"]], check=True,
                          capture_output=True).stdout
    tarfile.open(fileobj=io.BytesIO(blob)).extractall(archive, filter="data")

    from loggetta import describe_model
    from loggetta.execution import RECEIPT_SCHEMA

    def _receipts(d):
        """``load_observations`` for one directory, skipping JSON that is not an object (some evidence is a list)."""
        out = []
        for name in os.listdir(d):
            if name.endswith(".json"):
                try:
                    r = json.load(open(os.path.join(d, name)))
                except (OSError, ValueError):
                    continue
                if isinstance(r, dict) and r.get("schema") == RECEIPT_SCHEMA:
                    out.append(r)
        return sorted(out, key=lambda r: ((r.get("provenance") or {}).get("started_at") or "", r.get("run_id") or ""),
                      reverse=True)

    a2000 = []
    if a.a2000_evidence:
        seen = set()
        for d, _, files in sorted(os.walk(a.a2000_evidence)):
            if not any(f.endswith(".json") for f in files):
                continue
            for o in _receipts(d):
                card = ((o.get("hardware") or {}).get("gpu") or {}).get("name", "")
                kind = (o.get("workload") or {}).get("kind")
                if "A2000" in card and kind == "train" and o.get("run_id") not in seen:
                    seen.add(o.get("run_id"))
                    a2000.append(o)
        print(f"{len(a2000)} A2000 training receipts on file for the variant")

    groups = collections.defaultdict(list)
    for r in reg["rows"]:
        groups[(*setup_key(r), r["primary"])].append(r)  # primary arms are graded apart from excluded ones
    topologies, out = {}, []
    for k, rs in sorted(groups.items(), key=lambda kv: [str(x) for x in kv[0]]):
        model = k[1]
        if model not in topologies:
            topologies[model] = describe_model(model, revision=rs[0].get("revision"))
        hw, total = hardware(archive, rs[0])
        none = plan_one(topologies[model], hw, k, ())
        var = plan_one(topologies[model], hw, k, a2000) if a.a2000_evidence else None
        drv = [r["driver_peak_bytes"] for r in rs if r["driver_peak_bytes"]]
        alloc = [r["allocated_peak_bytes"] for r in rs]
        pt, est = none.get("plan_total_bytes"), none.get("allocator_estimate_bytes")
        out.append({
            "setup": dict(zip(("gpu", "model", "seq", "micro_batch", "expert_kernel", "expert_residency", "adapter_dtype",
                                "attn_4bit", "r", "alpha", "grad_accum", "optimizer"), k)),
            "card_total_bytes": total, "primary": k[-1],
            "primary_arms": sum(r["primary"] for r in rs), "arms": len(rs),
            "e4b_versions": sorted({str(r["e4b_version"]) for r in rs}),
            "driver_peak_max_bytes": max(drv) if drv else None, "allocated_peak_max_bytes": max(alloc),
            "plan_none": none, "plan_a2000_on_file": var,
            "under": bool(pt and drv and max(drv) > pt),
            "estimate_short": bool(est and max(alloc) > est),
            "false_refusal": bool(none.get("refused") or none.get("plan_total_bytes") is None),
            "arm_rows": [{"arm_receipt": r["arm_receipt"], "primary": r["primary"],
                          "driver_peak_bytes": r["driver_peak_bytes"], "allocated_peak_bytes": r["allocated_peak_bytes"],
                          "driver_over_plan": (r["driver_peak_bytes"] / pt) if (pt and r["driver_peak_bytes"]) else None,
                          "allocated_over_estimate": (r["allocated_peak_bytes"] / est) if est else None} for r in rs],
        })
    json.dump({"registration": "bench/HO1-PREREG.md", "setups": out}, open(os.path.join(a.out, "ho1.json"), "w"),
              indent=1, sort_keys=True)
    print(f"{len(out)} setups written")


if __name__ == "__main__":
    main()
