"""Before/after for one measured serve receipt: plan its exact workload and setup without, then with, the receipt's
directory among the observations, and set both beside what it measured.

    python bench/replan_serve.py RECEIPT --hardware PROFILE.json --before DIR [--before DIR ...] --after DIR \
        --out evidence/<name>.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

GiB = 1 << 30
OVERHEAD = ("allocator reserve (cached, unallocated blocks)", "CUDA context + library workspaces")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("receipt")
    ap.add_argument("--hardware", required=True)
    ap.add_argument("--before", action="append", default=[])
    ap.add_argument("--after", required=True, help="the directory that holds the receipt")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from loggetta import Constraints, Workload, describe_model, plan
    from loggetta.hardware import HardwareProfile
    from loggetta.runtime import load_observations

    r = json.load(open(a.receipt))
    m, wl = r["measured"], r["workload"]
    hw = HardwareProfile.from_dict(json.load(open(a.hardware)), origin=a.hardware)
    topo = describe_model(r["model"]["model"])
    w = Workload(kind="serve", context_len=wl["context_len"], concurrency=wl["concurrency"])
    old = [o for d in a.before for o in load_observations(d)]
    fixed = {"graphs": r["setup"]["graphs"]}
    if r["setup"].get("placement") == "solver":           # the run's own tier budgets, as the server read them
        fixed.update(placement="solver", vram_gb=float(r["setup"]["vram_gb"]), dram_gb=float(r["setup"]["dram_gb"]),
                     hot_rows=int(r["setup"]["hot_rows"]))
    c = Constraints(fixed=fixed, vram_budget=hw.gpu(0).memory_total.value)
    rows = []
    for label, obs in (("before", old), ("after", old + load_observations(a.after))):
        p = plan(topo, hw, w, c, observations=obs)
        cand = p.selected or min(p.alternatives, key=lambda x: x.device_bytes)
        lines = {ln.name: ln for ln in cand.lines}
        alloc = sum(ln.bytes for n, ln in lines.items() if ln.where == "device" and n not in OVERHEAD)
        row = {"observations": label, "status": p.status, "allocator_estimate_gib": round(alloc / GiB, 3),
               "plan_host_total_gib": round(cand.host_bytes / GiB, 3),
               "nvme_gib": round(sum(ln.bytes for ln in cand.lines if ln.where == "nvme") / GiB, 3),
               "reserve_gib": round(lines[OVERHEAD[0]].bytes / GiB, 3), "reserve_basis": lines[OVERHEAD[0]].basis,
               "reserve_detail": lines[OVERHEAD[0]].detail, "context_gib": round(lines[OVERHEAD[1]].bytes / GiB, 3),
               "context_detail": lines[OVERHEAD[1]].detail, "plan_device_total_gib": round(cand.device_bytes / GiB, 3)}
        rows.append(row)
        print(f"{label:6s} alloc est {row['allocator_estimate_gib']:.3f} | reserve {row['reserve_gib']:.2f} "
              f"[{row['reserve_basis']}] context {row['context_gib']:.2f} | plan total {row['plan_device_total_gib']:.3f}"
              f" | host {row['plan_host_total_gib']:.3f} nvme {row['nvme_gib']:.3f}")
    meas = {"allocator_peak_gib": round(m["device_peak_bytes"] / GiB, 3),
            "host_required_peak_gib": round(m["host_required_peak_bytes"] / GiB, 3) if m.get("host_required_peak_bytes")
            else None,
            "reserved_peak_gib": round(m["device_reserved_peak_bytes"] / GiB, 3),
            "driver_peak_gib": round(m["driver_process_peak_bytes"] / GiB, 3) if m.get("driver_process_peak_bytes") else None}
    print("measured", meas)
    json.dump({"schema": "replan-comparison/1", "receipt": r["run_id"], "model": r["model"]["model"],
               "gpu": r["hardware"]["gpu"]["name"], "workload": wl, "setup": r["setup"], "rows": rows, "measured": meas},
              open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
