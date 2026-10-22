"""IN-SAMPLE re-plans: today's training plan for rows the planner was built on, each row's own setup fixed, no receipts on file.

    python bench/insample_replan.py receipts --out evidence/<dir>/lg-insample-now.json
    python bench/insample_replan.py dense --reads 'E4B/bench/dq7/receipts/dq7-5090-1/read-*.json' --configs E4B/bench/dq7/configs \
        --out evidence/<dir>/dq7-insample-now.json [--skip ARM]

``receipts`` reads every training ExecutionReceipt under ``evidence/`` (plan attached or imported, like FP1's) and plans it again
through its own backend. ``dense`` reads experts4bit-qlora's DQ7-style ``read-<model>-<placement>-<seq>[-<variant>].json`` files,
describes each model from the lane's committed config, and plans it on the dense backend. Each row records today's plan total,
its allocator estimate where the backend itemizes one, and the receipt's own measured peaks. Every row is SEEN in the planner's
split, so the output is in-sample by construction: it shows where today's plan misses and is never a band.
"""
from __future__ import annotations

import argparse
import glob
import importlib
import json
import os

GiB, MiB = 1 << 30, 1 << 20


def _hw(gpu: dict):
    from loggetta.hardware import GPU, Fact, HardwareProfile, Host

    r = lambda v: Fact(v, "stated")  # noqa: E731
    mt = gpu["memory_total"]
    mt = int(str(mt).split()[0]) * MiB if isinstance(mt, str) else int(mt)       # imported receipts record "32607 MiB"
    name = str(gpu.get("name"))
    cc = gpu.get("compute_capability") or ((12, 0) if "5090" in name else (8, 9) if "4090" in name else (8, 6))
    g = GPU(index=0, vendor="nvidia", name=name, uuid=None, compute_capability=r(tuple(cc)), memory_total=r(mt), memory_free=r(mt),
            driver=r("stated"), pcie_gen_max=r(4), pcie_width_max=r(16), pcie_gen_current=r(4), pcie_width_current=r(16))
    host = Host(cpu_model=r("stated"), cpus=r(16), memory_total=r(256 * GiB), memory_available=r(240 * GiB), memory_limit=r(256 * GiB))
    return HardwareProfile(gpus=(g,), host=host, platform="stated")


def _workload(wd: dict):
    from loggetta import Workload

    return Workload(kind="train", seq_len=wd["seq_len"], micro_batch=wd["micro_batch"], grad_accum=wd.get("grad_accum", 1),
                    steps=wd.get("steps", 20), optimizer=wd.get("optimizer", "adamw"))


def _plan(topology, backend, receipt):
    from loggetta import plan
    from loggetta.plan import Constraints

    return plan(topology, _hw(receipt["hardware"]["gpu"]), _workload(receipt["workload"]), backends=(backend,),
                constraints=Constraints(fixed=dict(receipt["setup"]), allow_development_executor=True))


def receipts(out):
    rows, topos = [], {}
    for f in sorted(glob.glob("evidence/**/*.json", recursive=True)):
        if "planner-split" in f or "insample" in f:
            continue
        try:
            d = json.load(open(f))
        except (OSError, ValueError):
            continue
        if not (isinstance(d, dict) and {"measured", "setup", "workload", "hardware", "model"} <= set(d)):
            continue
        if (d["workload"] or {}).get("kind", "train") != "train":
            continue
        sel = ((d.get("plan") or {}).get("selected")) or {}
        be = importlib.import_module(f"loggetta.backends.{sel.get('backend') or d.get('backend') or 'experts4bit'}")
        mid, rev = d["model"].get("model"), d["model"].get("revision")
        key = (be.__name__, mid, rev)
        if key not in topos:
            topos[key] = be.describe(mid, revision=rev) if rev else be.describe(mid)
        p = _plan(topos[key], be, d)
        m = d["measured"] or {}
        rows.append({"file": f, "model": mid, "gpu": d["hardware"]["gpu"].get("name"), "setup": d["setup"], "status": p.status,
                     "plan_total": p.selected.device_bytes if p.status == "feasible" else None,
                     "allocated_peak": m.get("device_peak_bytes"), "driver_peak": m.get("driver_process_peak_bytes"),
                     "plan_total_then": sel.get("device_bytes")})
    json.dump(rows, open(out, "w"), indent=1, default=str)
    return rows


def dense_reads(pattern, configs, out, skip=()):
    import transformers as tr
    from loggetta.backends import dense

    rows, topos = [], {}
    for f in sorted(glob.glob(pattern)):
        arm = os.path.basename(f)[len("read-"):-len(".json")]
        if arm in skip:
            continue
        d = json.load(open(f))
        model = arm.split("-")[0]
        if model not in topos:
            c = tr.AutoConfig.for_model(**json.load(open(os.path.join(configs, f"{model}.json"))))
            c._name_or_path = model
            topos[model] = dense.describe(c)
        lines, _, refused = dense.estimate(topos[model], dict(d["setup"]), _workload(d["workload"]))
        p = _plan(topos[model], dense, d)
        m = d["measured"]
        rows.append({"arm": arm, "status": p.status, "alloc_est": None if refused else sum(b for _, w, b, _, _ in lines if w == "device"),
                     "plan_total": p.selected.device_bytes if p.status == "feasible" else None, "A": m["device_peak_bytes"],
                     "R": m["device_reserved_peak_bytes"], "D": m["driver_process_peak_bytes"],
                     "then_alloc": d["comparison"]["device_allocator"]["estimated"], "then_total": d["comparison"]["device_driver"]["estimated"]})
    json.dump(rows, open(out, "w"), indent=1)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("receipts")
    r.add_argument("--out", required=True)
    d = sub.add_parser("dense")
    d.add_argument("--reads", required=True)
    d.add_argument("--configs", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--skip", action="append", default=[], help="an arm to leave out, e.g. a held-out row")
    a = ap.parse_args()
    rows = receipts(a.out) if a.cmd == "receipts" else dense_reads(a.reads, a.configs, a.out, set(a.skip))
    print(f"{len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()
