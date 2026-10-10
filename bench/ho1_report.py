"""Render evidence/2026-10-10-heldout-cards/RESULTS-table.md from ho1.json (every number is read, none typed)."""
from __future__ import annotations

import json
import os
import sys

GiB = 1 << 30
here = sys.argv[1] if len(sys.argv) > 1 else "evidence/2026-10-10-heldout-cards"
setups = json.load(open(os.path.join(here, "ho1.json")))["setups"]
g = lambda b: "—" if b is None else f"{b / GiB:.2f}"  # noqa: E731
lines = ["| card | model | seq × mb | kernel | adapter | arms | driver peak (sampled) | plan | driver / plan | "
         "allocated peak | estimate | allocated / estimate | plan status | verdict |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
P = sorted((s for s in setups if s["primary"]), key=lambda s: (s["setup"]["gpu"], s["setup"]["model"], s["setup"]["seq"],
                                                                s["setup"]["expert_kernel"], s["setup"]["adapter_dtype"]))
for s in P:
    S, n = s["setup"], s["plan_none"]
    pt, est = n["plan_total_bytes"], n["allocator_estimate_bytes"]
    drv, al = s["driver_peak_max_bytes"], s["allocated_peak_max_bytes"]
    arms_under = sum(1 for a in s["arm_rows"] if a["driver_peak_bytes"] and a["driver_peak_bytes"] > pt)
    verdict = f"**UNDER** ({arms_under} of {s['arms']} arms)" if s["under"] else "not shown under"
    lines.append(f"| {S['gpu'].replace('NVIDIA ', '').replace('GeForce ', '')} | {S['model'].split('/')[-1]} | "
                 f"{S['seq']} × {S['micro_batch']} | {S['expert_kernel']} | {S['adapter_dtype']} | {s['arms']} | {g(drv)} | "
                 f"{g(pt)} | {drv / pt:.3f} | {g(al)} | {g(est)} | {al / est:.3f}{' **short**' if s['estimate_short'] else ''} | "
                 f"{'refused' if n.get('refused') else 'feasible'} | {verdict} |")
c5 = [s for s in P if "5090" in s["setup"]["gpu"]]
summary = {
    "5090 setups": len(c5), "UNDER": sum(s["under"] for s in c5),
    "estimate short (exact)": sum(s["estimate_short"] for s in c5),
    "false refusals": sum(s["false_refusal"] for s in c5),
    "A2000-on-file variant differs": sum(1 for s in P if (s["plan_a2000_on_file"] or {}).get("plan_total_bytes")
                                         != s["plan_none"]["plan_total_bytes"]),
}
out = "\n".join(lines) + "\n\nGiB. Driver peak, allocated peak: the maximum over the setup's primary arms.\n\n" + \
    "\n".join(f"- {k}: {v}" for k, v in summary.items()) + "\n"
open(os.path.join(here, "RESULTS-table.md"), "w").write(out)
print(out)
