"""Before/after: plan P109's serve setups (Qwen3-30B-A3B, all-VRAM, 16 x 4096, graphs on and off) on the RTX 5090
with the training receipts only, then with P109's serve receipts added, and set each beside what P109 measured.

    python bench/replan_with_p109.py --a2000 evidence/2026-10-04-rtx-a2000 --fp1 evidence/2026-10-04-fp1-rtx5090 \
        --p109 evidence/2026-10-04-p109-rtx5090-serve --out evidence/p109-replan.json

The allocator estimate (the backend's lines) is independent of P109. The reserve slack in the "after" plans is
P109's own, so agreement on that line is by construction; it is shown to say what the 20% default cost.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

GiB = 1 << 30
OVERHEAD = ("allocator reserve (cached, unallocated blocks)", "CUDA context + library workspaces")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a2000", required=True)
    ap.add_argument("--fp1", required=True)
    ap.add_argument("--p109", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from replan_with_fp1 import rtx5090_profile

    from loggetta import Constraints, Workload, describe_model, plan
    from loggetta.execution import load_observations

    old = load_observations(a.a2000) + load_observations(a.fp1)
    p109 = load_observations(a.p109)
    hw = rtx5090_profile(a.fp1)
    topo = describe_model("Qwen/Qwen3-30B-A3B")
    w = Workload(kind="serve", context_len=4096, concurrency=16)
    rows = []
    for graphs in (False, True):
        measured = [o for o in p109 if o["setup"]["graphs"] == graphs]
        for label, obs in (("before", old), ("after", old + p109)):
            c = Constraints(fixed={"graphs": graphs}, vram_budget=hw.gpu(0).memory_total.value)
            p = plan(topo, hw, w, c, observations=obs)
            cand = p.selected or min(p.alternatives, key=lambda x: x.device_bytes)
            lines = {ln.name: ln for ln in cand.lines}
            alloc_est = sum(ln.bytes for n, ln in lines.items() if ln.where == "device" and n not in OVERHEAD)
            used = max(o["measured"]["device_reserved_peak_bytes"] + o["measured"]["cuda_context_bytes"] for o in measured)
            row = {"graphs": graphs, "observations": label, "status": p.status,
                   "allocator_estimate_gib": round(alloc_est / GiB, 3),
                   "measured_allocator_peak_gib": [round(o["measured"]["device_peak_bytes"] / GiB, 3) for o in measured],
                   "reserve_gib": round(lines[OVERHEAD[0]].bytes / GiB, 3), "reserve_basis": lines[OVERHEAD[0]].basis,
                   "reserve_detail": lines[OVERHEAD[0]].detail,
                   "context_gib": round(lines[OVERHEAD[1]].bytes / GiB, 3), "context_basis": lines[OVERHEAD[1]].basis,
                   "plan_device_total_gib": round(cand.device_bytes / GiB, 3),
                   "measured_device_used_gib": round(used / GiB, 3),
                   "measured_arms": [o["run_id"] for o in measured]}
            rows.append(row)
            print(f"graphs={graphs!s:5s} {label:6s} alloc est {row['allocator_estimate_gib']:.3f} vs measured "
                  f"{row['measured_allocator_peak_gib']} GiB | reserve {row['reserve_gib']:.2f} [{row['reserve_basis']}] "
                  f"context {row['context_gib']:.2f} [{row['context_basis']}] | plan total {row['plan_device_total_gib']:.2f}"
                  f" vs used (largest arm) {row['measured_device_used_gib']:.2f} GiB")
    json.dump({"schema": "replan-comparison/1", "model": "Qwen/Qwen3-30B-A3B", "workload": "serve 4096 x 16, all-vram",
               "rows": rows}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
