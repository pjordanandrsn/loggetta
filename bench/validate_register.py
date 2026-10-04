"""Price recipes experts4bit-qlora has ALREADY measured, and compare with the recorded allocator peaks.

No GPU and no weights: each row is ``describe_moe`` on the model's config plus ``estimate_qlora_footprint`` on the
recipe the register states. The measured value is torch ``max_memory_allocated`` from the cited claim, and the
estimate is compared WITHOUT the planner's CUDA-context line, because the allocator never sees the context. A
model whose config is not cached locally is fetched (config.json only).

    python bench/validate_register.py [--out bench/validate_register.json]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
GB = 1e9      # the register quotes decimal GB

# (claim id, model, setup fields, tokens per micro-batch, optimizer, measured peak GB, recipe note)
FIELD = dict(r=16, alpha=16, adapter_dtype="fp32", attn_4bit=True, expert_kernel="grouped_nf4", dgrad=True)
# T = the LARGEST micro-batch's padded token count, because the peak follows the largest forward. TC3 receipts record
# every micro-batch's padded length (``microbatch_padded_len``): max 555 tokens per row. tp4's receipts are not in the
# repository; its field recipe uses the same alpaca truncation, so T=2x555 is ASSUMED for those rows and labelled so.
ROWS = [
    ("e4b.train.h2h.unsloth.olmoe.5090.2026-09-19.arm.e4b.fused_attn4", "allenai/OLMoE-1B-7B-0924",
     dict(FIELD), 1110, "adamw_8bit", 6.419, "tp4 field recipe, RTX 5090, mb 2 (T assumed)"),
    ("e4b.train.h2h.unsloth.olmoe.5090.2026-09-19.arm.e4b.reference_attn4", "allenai/OLMoE-1B-7B-0924",
     dict(FIELD, expert_kernel="reference"), 1110, "adamw_8bit", 5.866, "tp4 field recipe, reference loop (T assumed)"),
    ("e4b.train.h2h.unsloth.granite.5090.2026-09-19.arm.e4b.fused_attn4", "ibm-granite/granite-3.1-3b-a800m-instruct",
     dict(FIELD), 1110, "adamw_8bit", 3.782, "tp4 field recipe, RTX 5090, mb 2 (T assumed)"),
    ("e4b.train.h2h.unsloth.qwen3.5090.2026-09-19.arm.e4b.fused_attn4", "Qwen/Qwen3-30B-A3B",
     dict(FIELD), 1110, "adamw_8bit", 24.581, "tp4 field recipe, RTX 5090, resident, mb 2 (T assumed)"),
    ("e4b.train.frontier.qwen3.a2000-12gb.2026-10-02", "Qwen/Qwen3-30B-A3B",
     dict(FIELD, expert_residency="host"), 555, "adamw_8bit", 10.463,
     "TC3, RTX A2000 12 GB, offload, mb 1 (T from the receipt: max padded row 555)"),
    ("e4b.train.frontier.qwen3.4090-24gb.2026-10-02", "Qwen/Qwen3-30B-A3B",
     dict(FIELD, expert_residency="host"), 1110, "adamw_8bit", 11.881,
     "TC3, RTX 4090 24 GB, offload, mb 2 (T from the receipt tc3-4090-8: 2 x 555)"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "validate_register.json"))
    a = ap.parse_args()
    from experts4bit_qlora import QLoRASetup, describe_moe, estimate_qlora_footprint

    topos, out = {}, []
    print(f"{'claim':72s} {'est GB':>7s} {'meas GB':>8s} {'resid':>7s}  derived/heuristic GB")
    for cid, model, setup, tokens, opt, measured, note in ROWS:
        topo = topos.get(model) or topos.setdefault(model, describe_moe(model))
        fp = estimate_qlora_footprint(topo, QLoRASetup(**setup), tokens_per_microbatch=tokens, optimizer=opt)
        est = fp.device_bytes / GB
        d, h = fp.total("device", "derived") / GB, fp.total("device", "heuristic") / GB
        row = {"claim": cid, "model": model, "setup": setup, "tokens_per_microbatch": tokens, "optimizer": opt,
               "note": note, "estimated_gb": round(est, 3), "measured_gb": measured,
               "residual_gb": round(measured - est, 3), "ratio": round(measured / est, 4),
               "derived_gb": round(d, 3), "heuristic_gb": round(h, 3),
               "items": [{"name": i.name, "gb": round(i.bytes / GB, 3), "basis": i.basis} for i in fp.items
                         if i.where == "device"]}
        out.append(row)
        print(f"{cid:72s} {est:7.3f} {measured:8.3f} {measured - est:+7.3f}  {d:.3f}/{h:.3f}")
    with open(a.out, "w") as f:
        json.dump({"schema": "estimate-validation/1", "rows": out}, f, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
