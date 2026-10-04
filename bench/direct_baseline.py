"""The pre-integration path, for comparison: the same setup built by hand from experts4bit-qlora's public calls.

No planner, no ``recipe.prepare_qlora_training``: ``load_moe_4bit_streaming`` -> gradient checkpointing ->
``add_attention_lora`` -> ``enable_fast_train`` -> parameter selection, as the package's documented fast path does
it. The measured loop is the one the planner's executor uses (``train_loop``), so any difference between this
arm and a planned run is the orchestration layer's, not the loop's.

    python bench/direct_baseline.py --receipt receipts/<planned run>.json --out receipts/
reads the planned run's model, workload and setup and repeats them by hand.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--receipt", required=True, help="a planned run's receipt to repeat by hand")
    ap.add_argument("--out", default="receipts")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from loggetta.measure import changed_since, provenance

    prov = {**provenance(), "taken": "at process start, before the measured code was imported"}
    planned = json.load(open(a.receipt))
    s, wd = planned["setup"], planned["workload"]
    model_id, revision = planned["model"]["model"], planned["model"].get("revision")

    import torch

    from experts4bit_qlora import enable_fast_train, load_moe_4bit_streaming
    from experts4bit_qlora.lora import ExpertsLoRA, LoRALinear, add_attention_lora

    from loggetta.backends.experts4bit_train import train_loop
    from loggetta.measure import DriverMemorySampler, proc_status
    from loggetta.plan import Workload

    assert not s["attn_4bit"] and s["adapter_dtype"] == "bf16" and not s["keep_moe_layers"], \
        "this baseline repeats the default setup only"
    w = Workload(**{k: wd[k] for k in Workload.__dataclass_fields__})
    torch.manual_seed(a.seed)
    torch.zeros(1, device="cuda")
    meas = {"host_baseline_bytes": proc_status().get("VmRSS")}
    with DriverMemorySampler() as smi:
        t0 = time.time()
        host = s["expert_residency"] == "host"
        model, _ = load_moe_4bit_streaming(model_id, "cuda", torch.bfloat16, s["r"], s["alpha"], offload=host,
                                           pin=s["pin"], quant_type=s["quant_type"], revision=revision,
                                           blocksize=s["blocksize"])
        if not host:
            model.to("cuda")
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        n_attn = add_attention_lora(model, s["r"], s["alpha"], torch.bfloat16) if s["train_attention"] else 0
        n_fused = enable_fast_train(model, dgrad=s["dgrad"]) if s["expert_kernel"] == "grouped_nf4" else 0
        if s["expert_kernel"] == "grouped_nf4":
            assert n_fused > 0
        trainable = [p for m in model.modules() if isinstance(m, (ExpertsLoRA, LoRALinear))
                     for n, p in m.named_parameters(recurse=False) if "lora" in n]
        keep = {id(p) for p in trainable}
        for p in model.parameters():
            p.requires_grad_(id(p) in keep)
        meas["load_seconds"] = time.time() - t0
        out = train_loop(model, trainable, model_id, w, smi, meas, revision=revision, seed=a.seed)
    out.update(arm="direct (hand-composed public API, no planner)", setup=s, workload=wd,
               engaged={"attention_lora_projections": n_attn, "fused_expert_modules": n_fused,
                        "trainable_numel": sum(p.numel() for p in trainable)},
               provenance={**prov, "changed_during_run": changed_since(prov)}, planned_receipt=os.path.basename(a.receipt), schema="execution-receipt/1",
               run_id="direct-" + planned["run_id"])
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, out["run_id"] + ".json")
    json.dump(out, open(path, "w"), indent=1, sort_keys=True, default=str)
    print(json.dumps({"status": out["status"], "s_per_step_median": meas["s_per_step_median"],
                      "device_peak_gib": meas["device_peak_bytes"] / 2 ** 30, "losses": out["correctness"]["losses"]},
                     indent=1))
    print("receipt:", path)


if __name__ == "__main__":
    main()
