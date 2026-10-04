"""Plan the paged server (all-VRAM placement) over a context x concurrency grid: what fits, what the KV pool costs,
and what a refusal suggests. Planning only; nothing is loaded or run.

    python bench/serve_plan_sweep.py --a2000 evidence/2026-10-04-rtx-a2000/hardware-profile.json \
        --obs evidence/2026-10-04-rtx-a2000 --fp1 evidence/2026-10-04-fp1-rtx5090 --out evidence/serve-plan-sweep.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

GiB = 1 << 30
CONTEXTS = (2048, 8192, 32768)
CONCURRENCY = (1, 4, 16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a2000", required=True)
    ap.add_argument("--obs", required=True, help="the A2000 receipts (CUDA context, host baseline)")
    ap.add_argument("--fp1", required=True, help="the imported FP1 observations (the RTX 5090's context)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from replan_with_fp1 import rtx5090_profile

    from loggetta import Constraints, Workload, describe_model, plan
    from loggetta.hardware import HardwareProfile
    from loggetta.runtime import load_observations

    obs = load_observations(a.obs) + load_observations(a.fp1)
    cases = [("allenai/OLMoE-1B-7B-0924", "RTX A2000 (seat)", HardwareProfile.from_dict(json.load(open(a.a2000)),
                                                                                        origin=a.a2000)),
             ("Qwen/Qwen3-30B-A3B", "RTX 5090 (FP1's box)", rtx5090_profile(a.fp1))]
    rows = []
    for model, hw_name, hw in cases:
        topo = describe_model(model)
        c = Constraints(vram_budget=hw.gpu(0).memory_total.value)
        for ctx in CONTEXTS:
            for seqs in CONCURRENCY:
                p = plan(topo, hw, Workload(kind="serve", context_len=ctx, concurrency=seqs), c, observations=obs)
                cand = p.selected or min(p.alternatives, key=lambda x: x.device_bytes)
                lines = {ln.name: ln for ln in cand.lines}
                kv = lines.get("FP8 paged KV pool")
                row = {"model": model, "hardware": hw_name, "context": ctx, "concurrency": seqs, "status": p.status,
                       "device_gib": round(cand.device_bytes / GiB, 2), "budget_gib": round(p.budget["device"] / GiB, 2),
                       "kv_gib": round(kv.bytes / GiB, 2) if kv else None,
                       "weights_gib": round(sum(ln.bytes for n, ln in lines.items()
                                                if n.startswith(("frozen expert", "dense weights"))) / GiB, 2),
                       "graphs": cand.setup.get("graphs"),
                       "suggestions": (p.refusal or {}).get("suggestions", [])}
                rows.append(row)
                print(f"{model:26s} {hw_name:20s} ctx {ctx:6d} x {seqs:2d}  {p.status:9s} device {row['device_gib']:6.2f}"
                      f" / {row['budget_gib']:5.2f} GiB  kv {row['kv_gib']}  {' | '.join(row['suggestions'])}")
    json.dump({"schema": "serve-plan-sweep/1", "placement": "all-vram", "rows": rows}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
