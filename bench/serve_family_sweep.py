"""Serve plans for every family in the training sweep, on four cards and two workloads: which placement each gets
(all-VRAM, or the solver's VRAM/DRAM/NVMe tiers), how the experts split, and why a refusal refuses. Planning only.

    python bench/serve_family_sweep.py --a2000 evidence/2026-10-04-rtx-a2000/hardware-profile.json \
        --fp1 evidence/2026-10-04-fp1-rtx5090 --obs DIR [--obs DIR ...] --out evidence/serve-family-sweep.json

The A2000 profile is the seat's (probed); the RTX 5090 is rebuilt from FP1's receipt; the RTX 4090 and RTX 3090 are
stated cards ("user" facts: total memory, capability, PCIe, a 64 GiB host), so their plans are what-ifs.
"""
import argparse
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

GiB = 1 << 30
MODELS = ("allenai/OLMoE-1B-7B-0924", "Qwen/Qwen3-30B-A3B", "Qwen/Qwen3.6-35B-A3B", "ibm-granite/granite-3.1-3b-a800m-instruct",
          "ibm-granite/granite-4.0-h-tiny", "LiquidAI/LFM2-8B-A1B", "mistralai/Mixtral-8x7B-Instruct-v0.1",
          "baidu/ERNIE-4.5-21B-A3B-PT", "deepseek-ai/DeepSeek-V2-Lite", "openai/gpt-oss-20b")
WORKLOADS = ((4096, 1), (8192, 8))


def stated_card(name, total_gib, cap, host_gib=64):
    from loggetta.hardware import GPU, Fact, HardwareProfile, Host

    u = lambda v: Fact(v, "user", "stated card for a what-if plan")  # noqa: E731
    gpu = GPU(index=0, vendor="nvidia", name=name, uuid=None, compute_capability=u(cap),
              memory_total=u(int(total_gib * GiB)), memory_free=u(int(total_gib * GiB)), driver=u("stated"),
              pcie_gen_max=u(4), pcie_width_max=u(16), pcie_gen_current=u(4), pcie_width_current=u(16))
    host = Host(cpu_model=u("stated"), cpus=u(16), memory_total=u(host_gib * GiB), memory_available=u(host_gib * GiB),
                memory_limit=Fact(None, "unknown"))
    return HardwareProfile(gpus=(gpu,), host=host, platform="Linux x86_64", notes=("stated card",))


def tier(c, prefix):
    return round(sum(ln.bytes for ln in c.lines if ln.name.startswith(prefix)) / GiB, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a2000", required=True)
    ap.add_argument("--fp1", required=True)
    ap.add_argument("--obs", action="append", default=[])
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--cards", nargs="*", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from replan_with_fp1 import rtx5090_profile

    from loggetta import Constraints, Workload, describe_model, plan
    from loggetta.hardware import HardwareProfile
    from loggetta.runtime import load_observations

    obs = [o for d in a.obs + [a.fp1] for o in load_observations(d)]
    cards = {"RTX A2000 12GB (seat, probed)": HardwareProfile.from_dict(json.load(open(a.a2000)), origin=a.a2000),
             "RTX 5090 32GB (FP1's box)": rtx5090_profile(a.fp1),
             "RTX 4090 24GB (stated)": stated_card("NVIDIA GeForce RTX 4090", 24, (8, 9)),
             "RTX 3090 24GB (stated)": stated_card("NVIDIA GeForce RTX 3090", 24, (8, 6))}
    if a.cards:
        cards = {k: v for k, v in cards.items() if any(c in k for c in a.cards)}
    rows = []
    for model in a.models:
        t0 = time.time()
        try:
            topo = describe_model(model)
        except Exception as e:  # noqa: BLE001 - a family the describer cannot read is a row, not a crash
            rows.append({"model": model, "status": "undescribable", "error": f"{type(e).__name__}: {e}"[:400]})
            print(model, "undescribable:", e, flush=True)
            continue
        print(f"{model}: described in {time.time() - t0:.0f}s ({topo.summary()})", flush=True)
        for card, hw in cards.items():
            for ctx, seqs in WORKLOADS:
                try:
                    p = plan(topo, hw, Workload(kind="serve", context_len=ctx, concurrency=seqs),
                             Constraints(vram_budget=hw.gpu(0).memory_total.value), observations=obs)
                except Exception as e:  # noqa: BLE001
                    rows.append({"model": model, "card": card, "context": ctx, "concurrency": seqs, "status": "error",
                                 "error": traceback.format_exc()[-800:]})
                    print(" ", card, ctx, seqs, "ERROR", e, flush=True)
                    continue
                c = p.selected or (min(p.alternatives, key=lambda x: x.device_bytes) if p.alternatives else None)
                row = {"model": model, "model_type": topo.model_type, "card": card, "context": ctx, "concurrency": seqs,
                       "status": p.status, "placement": c.setup.get("placement") if c else None,
                       "graphs": c.setup.get("graphs") if c else None,
                       "vram_tier_gib": tier(c, "expert stacks, VRAM") or tier(c, "frozen expert stacks") if c else None,
                       "dram_tier_gib": tier(c, "expert stacks, DRAM") if c else None,
                       "nvme_gib": tier(c, "expert rows on NVMe") if c else None,
                       "kv_gib": tier(c, "FP8 paged KV pool") if c else None,
                       "device_gib": round(c.device_bytes / GiB, 2) if c else None,
                       "host_gib": round(c.host_bytes / GiB, 2) if c else None,
                       "budget_gib": round(p.budget["device"] / GiB, 2), "host_budget_gib": round(p.budget["host"] / GiB, 2),
                       "refusal": (p.refusal or {}).get("reasons"), "suggestions": (p.refusal or {}).get("suggestions"),
                       "kernel_notes": [r for r in p.reasons if "not usable here" in r]}
                rows.append(row)
                print(f"  {card:30s} {ctx:5d}x{seqs:<2d} {p.status:9s} {row['placement']!s:9s} V {row['vram_tier_gib']} "
                      f"D {row['dram_tier_gib']} N {row['nvme_gib']} | dev {row['device_gib']}/{row['budget_gib']} "
                      f"host {row['host_gib']}/{row['host_budget_gib']}", flush=True)
    json.dump({"schema": "serve-family-sweep/1", "workloads": WORKLOADS, "rows": rows}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
