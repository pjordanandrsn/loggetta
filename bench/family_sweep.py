"""Architecture pressure test: plan many MoE families with ZERO per-family code in the planner.

For each model: describe it (config.json + a meta-device tree), plan a QLoRA workload against a saved hardware
profile, and record what happened -- planned, refused (and why), or failed (an exception: a defect somewhere).
The point is the column "per-family code added": it must stay empty.

    python bench/family_sweep.py --hardware ../runs/seat-hw.json --out bench/family_sweep.json
"""
import argparse
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

MODELS = [
    "allenai/OLMoE-1B-7B-0924",                        # olmoe
    "Qwen/Qwen3-30B-A3B",                              # qwen3_moe
    "Qwen/Qwen3.6-35B-A3B",                            # qwen3_5_moe (hybrid linear attention)
    "ibm-granite/granite-3.1-3b-a800m-instruct",       # granitemoe (pre-fused legacy spelling)
    "ibm-granite/granite-4.0-h-tiny",                  # granitemoehybrid (Mamba + attention, shared MLP)
    "LiquidAI/LFM2-8B-A1B",                            # lfm2_moe (short convolutions)
    "mistralai/Mixtral-8x7B-Instruct-v0.1",            # mixtral (block_sparse_moe on disk)
    "baidu/ERNIE-4.5-21B-A3B-PT",                      # ernie4_5_moe (qwen2_moe convention, admitted by convention)
    "deepseek-ai/DeepSeek-V2-Lite",                    # deepseek_v2 (MLA attention, shared experts, dense first layer)
    "openai/gpt-oss-20b",                              # gpt_oss (biased, clamped experts: expert LoRA must refuse)
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hardware", required=True)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "family_sweep.json"))
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--observations", help="receipt directory measured on the profiled machine")
    a = ap.parse_args()
    from loggetta import Workload, describe_model, plan
    from loggetta.hardware import HardwareProfile
    from loggetta.execution import load_observations

    obs = load_observations(a.observations)
    hw = HardwareProfile.from_dict(json.load(open(a.hardware)), origin=os.path.basename(a.hardware))
    rows, texts = [], []
    for m in a.models:
        t0 = time.time()
        row = {"model": m}
        try:
            topo = describe_model(m)
            row.update(model_type=topo.model_type, convention=topo.convention, summary=topo.summary(),
                       moe_layers=len(topo.expert_stacks), n_layers=topo.n_layers,
                       attention_layers=topo.attention.layers if topo.attention else None,
                       expert_bias=list(topo.expert_bias_tensors), loader_refusal=topo.loader_refusal)
            p = plan(topo, hw, Workload(seq_len=a.seq), observations=obs)
            row.update(status=p.status, seconds=round(time.time() - t0, 2))
            if p.status == "feasible":
                row.update(selected=p.selected.setup, device_gib=round(p.selected.device_bytes / 2**30, 2),
                           host_gib=round(p.selected.host_bytes / 2**30, 2))
            else:
                row.update(refusal=p.refusal["reasons"], suggestions=p.refusal.get("suggestions"))
            texts.append(f"\n{'=' * 110}\n{m}\n{p.render()}")
        except Exception as e:  # noqa: BLE001 - every failure is a finding, recorded
            row.update(status="ERROR", error=f"{type(e).__name__}: {e}"[:600], trace=traceback.format_exc()[-1500:])
        rows.append(row)
        print(json.dumps({k: row.get(k) for k in ("model", "model_type", "status", "device_gib", "host_gib", "error")}),
              flush=True)
    json.dump({"schema": "family-sweep/1", "hardware": a.hardware, "seq": a.seq, "per_family_code_added": [],
               "rows": rows}, open(a.out, "w"), indent=1, default=str)
    open(a.out.replace(".json", ".txt"), "w").write("\n".join(texts))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
