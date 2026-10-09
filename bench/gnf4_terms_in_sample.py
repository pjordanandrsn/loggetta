"""The grouped_nf4 MoE backward line (#43) against every committed MoE training receipt with an allocated peak.

    PYTHONPATH=. python bench/gnf4_terms_in_sample.py [--json OUT]

IN-SAMPLE: these are the receipts the term was attributed on (#44 re-ran three of them on the A2000) or the audit that
found the gap (#29), so agreement here is not evidence for another card, shape or model. Per receipt, it prices the
receipt's own setup and workload with the experts4bit backend's estimate. "Before" is the estimate without the new line
(nothing else in the estimate changed); "after" is with it. Configs come from the Hugging Face Hub (no weights).
"""
from __future__ import annotations

import argparse
import glob
import json
import os

MiB = 1 << 20
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "evidence")
LINE = "grouped_nf4 MoE backward (above the activation item)"


def receipts(root):
    for f in sorted(glob.glob(os.path.join(root, "**", "*.json"), recursive=True)):
        try:
            r = json.load(open(f))
        except (OSError, ValueError):
            continue
        if not isinstance(r, dict) or r.get("schema") != "execution-receipt/1":
            continue
        setup, alloc = r.get("setup") or {}, (r.get("measured") or {}).get("device_peak_bytes")
        if "expert_kernel" not in setup or not alloc or (r.get("workload") or {}).get("kind", "train") != "train":
            continue
        if not isinstance(r.get("model"), dict):     # a direct arm's receipt names no model (its planned run does)
            continue
        yield os.path.relpath(f, root), r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", default=ROOT)
    ap.add_argument("--json")
    a = ap.parse_args()
    from loggetta import Workload, describe_model
    from loggetta.backends import experts4bit

    topologies, rows = {}, []
    for rel, r in receipts(a.evidence):
        name, rev = r["model"]["model"], r["model"].get("revision")
        if (name, rev) not in topologies:
            topologies[name, rev] = describe_model(name, revision=rev)
        wl = r.get("workload") or r["plan"]["workload"]
        wl = Workload(**{k: v for k, v in wl.items() if k in Workload.__dataclass_fields__})
        lines, _, refusals = experts4bit.estimate(topologies[name, rev], dict(r["setup"]), wl)
        if refusals:
            continue
        after = sum(ln[2] for ln in lines if ln[1] == "device")
        new = next((ln for ln in lines if ln[0] == LINE), None)
        before = after - (new[2] if new else 0)
        alloc = r["measured"]["device_peak_bytes"]
        rows.append({"receipt": rel, "card": r["hardware"]["gpu"]["name"], "model": name,
                     "kernel": r["setup"]["expert_kernel"], "residency": r["setup"]["expert_residency"],
                     "tokens": wl.tokens_per_microbatch, "allocated_peak_bytes": alloc,
                     "estimate_before_bytes": before, "estimate_after_bytes": after,
                     "new_line_bytes": new[2] if new else 0, "new_line_detail": new[4] if new else ""})
    print("| receipt | card | kernel, residency | allocated peak | est. before | before − alloc | new line | "
          "est. after | after − alloc |")
    print("|---|---|---|---|---|---|---|---|---|")
    for x in rows:
        al, b, af = x["allocated_peak_bytes"], x["estimate_before_bytes"], x["estimate_after_bytes"]
        print(f"| {x['receipt'].removesuffix('.json')} | {x['card'].replace('NVIDIA ', '')} | {x['kernel']}, "
              f"{x['residency']} | {al / MiB:.1f} | {b / MiB:.1f} | {(b - al) / MiB:+.1f} | {x['new_line_bytes'] / MiB:.1f} | "
              f"{af / MiB:.1f} | {(af - al) / MiB:+.1f} |")
    print("\nMiB. A negative difference is an estimate under the allocated peak.")
    for x in rows:
        if x["new_line_detail"]:
            print(f"\n- {x['receipt'].removesuffix('.json')}: {x['new_line_detail']}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rows, f, indent=1, sort_keys=True)


if __name__ == "__main__":
    main()
