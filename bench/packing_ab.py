"""Concatenated vs isolated packing, A/B on one GPU: step time and allocator peak, same model, data, loss and seq.

The question (owner's decision, 2026-10-07: isolated packing is the default for chat and alpaca, its cost priced, the
speed cost measured before release): what does isolation cost on a real QLoRA step? Isolation hands SDPA a
materialized [rows, 1, seq, seq] mask, which rules out the flash kernel; the plan prices the mask's memory
(micro-batch x seq^2 x 3 B) and states the speed cost without modelling it.

One process loads the model once through ``prepare_qlora_training`` (the executor's own path) and runs the measured
loop (``train_loop``) for each arm, in the order concat, isolated, concat, isolated, so drift shows up as A != A'.
The data is a deterministic synthetic alpaca set (no download): response-only loss in both arms, so only the packing
differs. Writes one JSON file with every arm's measurements, the plan's priced mask, and the environment.

    python bench/packing_ab.py --model allenai/OLMoE-1B-7B-0924 --seq 2048 --steps 12 --out evidence/<dir>
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time


def synthetic_alpaca(path, rows=400, seed=0):
    rng = random.Random(seed)
    words = "seal valve pump filter gasket hose bolt flange bearing motor sensor relay".split()
    with open(path, "w") as f:
        for i in range(rows):
            n_in, n_out = rng.choice([8, 30, 120, 400]), rng.choice([10, 40, 160, 500])
            f.write(json.dumps({"instruction": f"Note {i}: " + " ".join(rng.choices(words, k=n_in)),
                                "input": "", "output": " ".join(rng.choices(words, k=n_out))}) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--kernel", default="grouped_nf4", choices=("grouped_nf4", "reference"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch
    from experts4bit_qlora.recipe import QLoRASetup, prepare_qlora_training

    from loggetta.backends.experts4bit_train import train_loop
    from loggetta.data import TrainingData, encode_dataset, load_tokenizer, prepare_data
    from loggetta.measure import DriverMemorySampler, provenance
    from loggetta.plan import Workload
    from loggetta.planner import data_memory_lines

    prov = provenance()
    os.makedirs(a.out, exist_ok=True)
    data_path = os.path.join(a.out, "synthetic-alpaca.jsonl")
    synthetic_alpaca(data_path)
    tok = load_tokenizer(a.model)
    arms = {}
    for packing in ("concat", "isolated"):
        spec = TrainingData(data_path, format="alpaca", packing=packing)
        with encode_dataset(tok, spec, seq_len=a.seq) as enc:
            arms[packing] = {"spec": spec, "profile": enc.profile}
    torch.cuda.set_device(0)
    torch.zeros(1, device="cuda")
    results = []
    with DriverMemorySampler() as smi:
        t0 = time.time()
        prep = prepare_qlora_training(a.model, QLoRASetup(expert_kernel=a.kernel), device="cuda")
        load_s = time.time() - t0
        for packing in ("concat", "isolated", "concat", "isolated"):
            spec, profile = arms[packing]["spec"], arms[packing]["profile"]
            w = Workload(seq_len=a.seq, steps=a.steps, data=spec.to_dict())
            meas = {}
            with prepare_data(tok, a.steps, a.seq, spec, expect=profile) as prepared:
                out = train_loop(prep.model, prep.trainable, a.model, w, smi, meas, warmup=a.warmup,
                                 prepared_data=prepared)
                info = prepared.info
            priced = sum(ln.bytes for ln in data_memory_lines(w, profile))
            row = {"packing": packing, "s_per_step_median": meas["s_per_step_median"],
                   "step_seconds": meas["step_seconds"], "device_peak_bytes": meas["device_peak_bytes"],
                   "device_reserved_peak_bytes": meas["device_reserved_peak_bytes"],
                   "loss_tokens": info.get("loss_tokens"), "priced_mask_bytes": priced,
                   "losses": out["correctness"]["losses"], "status": out["status"]}
            results.append(row)
            print(json.dumps({k: row[k] for k in ("packing", "s_per_step_median", "device_peak_bytes",
                                                  "priced_mask_bytes", "loss_tokens")}), flush=True)
    record = {"schema": "packing-ab/1", "model": a.model, "seq": a.seq, "steps": a.steps, "warmup": a.warmup,
              "kernel": a.kernel, "load_seconds": load_s, "engaged": prep.report,
              "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
              "profiles": {k: v["profile"] for k, v in arms.items()}, "arms": results, "provenance": prov}
    with open(os.path.join(a.out, "packing-ab.json"), "w") as f:
        json.dump(record, f, indent=1, sort_keys=True, default=str)


if __name__ == "__main__":
    main()
