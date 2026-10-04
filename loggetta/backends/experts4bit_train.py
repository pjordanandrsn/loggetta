"""Execute a feasible experts4bit training plan, measuring what the plan estimated.

The model is built by ``experts4bit_qlora.recipe.prepare_qlora_training`` from exactly the setup the plan selected,
so the run executes what was priced. The loop is deliberately plain (AdamW, gradient accumulation, clip 1.0) on
fixed-shape packed blocks: fixed shapes make peak memory and step time comparable across runs and with the plan.
"""
from __future__ import annotations

import hashlib
import math
import statistics
import time

GiB = 1 << 30
PROMPT = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}"


def packed_blocks(tokenizer, n_blocks: int, seq_len: int, dataset="tatsu-lab/alpaca", seed: int = 0):
    """``n_blocks`` rows of ``seq_len`` token ids: alpaca examples in dataset order, joined by EOS, cut into blocks."""
    from datasets import load_dataset

    ds = load_dataset(dataset, split="train")
    ids, need, i = [], n_blocks * seq_len, 0
    eos = tokenizer.eos_token_id
    while len(ids) < need:
        ex = ds[i % len(ds)]
        ids += tokenizer(PROMPT.format(**ex), add_special_tokens=False)["input_ids"] + [eos]
        i += 1
    return [ids[k * seq_len:(k + 1) * seq_len] for k in range(n_blocks)], {"dataset": dataset, "examples_used": i}


def _expert_digest(model, layers=(0, -1)) -> dict:
    """sha256 of the frozen packed expert bytes (+ absmax) of a few stacks, wherever they live (device or host home)."""
    import torch
    from experts4bit_qlora import ExpertsNbit, offload_handles

    bases = [m for m in model.modules() if isinstance(m, ExpertsNbit)]
    handles = list(offload_handles(model))                      # one per layer under host residency, else empty
    out = {}
    for li in layers:
        base = bases[li]
        tensors = {n: getattr(base, n) for n in ("gate_up_proj", "down_proj", "gate_up_absmax", "down_absmax")}
        if tensors["gate_up_proj"].numel() == 0 and handles:   # offloaded: the bytes are in the host home
            tensors = dict(handles[li].home)
        sha = hashlib.sha256()
        for n in sorted(tensors):
            if tensors[n] is not None:
                sha.update(tensors[n].detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
        out[f"stack[{li}]"] = sha.hexdigest()
    return out


def link_h2d(n_bytes: int = 256 << 20, reps: int = 5) -> dict:
    """Measured pinned host-to-device copy bandwidth (GB/s, median of ``reps``), before anything else uses the GPU."""
    import statistics as st

    import torch

    src = torch.empty(n_bytes, dtype=torch.uint8).pin_memory()
    dst = torch.empty(n_bytes, dtype=torch.uint8, device="cuda")
    dst.copy_(src, non_blocking=True)
    torch.cuda.synchronize()
    rates = []
    for _ in range(reps):
        t = time.perf_counter()
        dst.copy_(src, non_blocking=True)
        torch.cuda.synchronize()
        rates.append(n_bytes / (time.perf_counter() - t) / 1e9)
    del src, dst
    torch.cuda.empty_cache()
    return {"link_h2d_gbps": st.median(rates), "link_h2d_probe": f"{n_bytes >> 20} MiB pinned x{reps}, median"}


def run(plan, *, seed: int = 0, warmup: int = 2, log=print) -> dict:
    """Build the plan's setup with ``prepare_qlora_training`` and train it through :func:`train_loop`."""
    import torch

    from experts4bit_qlora.recipe import QLoRASetup, prepare_qlora_training

    from ..measure import DriverMemorySampler, proc_status

    torch.manual_seed(seed)
    torch.cuda.set_device(plan.constraints.device)
    torch.zeros(1, device="cuda")                       # CUDA context up before the baseline reading
    meas = {"host_baseline_bytes": proc_status().get("VmRSS"), **link_h2d()}
    with DriverMemorySampler() as smi:
        t0 = time.time()
        prep = prepare_qlora_training(plan.model["model"], QLoRASetup(**plan.selected.setup), device="cuda",
                                      revision=plan.model.get("revision"))
        meas["load_seconds"] = time.time() - t0
        meas["load_device_peak_bytes"] = torch.cuda.max_memory_allocated()
        meas["host_anon_after_load_bytes"] = proc_status().get("RssAnon")
        out = train_loop(prep.model, prep.trainable, plan.model["model"], plan.workload, smi, meas,
                         revision=plan.model.get("revision"), seed=seed, warmup=warmup, log=log)
    out["engaged"] = prep.report
    return out


def train_loop(model, trainable, model_id, w, smi, meas, *, revision=None, seed=0, warmup=2, log=print) -> dict:
    """The measured loop, shared by every arm that must be comparable: fixed-shape packed blocks, AdamW 2e-4,
    gradient accumulation, clip 1.0. Fills ``meas`` and returns the correctness record."""
    import torch
    from transformers import AutoTokenizer

    from ..measure import proc_status

    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    blocks, data_info = packed_blocks(tok, w.steps * w.grad_accum * w.micro_batch, w.seq_len, seed=seed)
    digest_before = _expert_digest(model)
    b_norm = lambda: sum(float(p.detach().float().norm()) for n, p in model.named_parameters()  # noqa: E731
                         if p.requires_grad and "lora_B" in n)
    b_norm_before = b_norm()
    torch.manual_seed(seed)
    if w.optimizer == "adamw_8bit":
        import bitsandbytes as bnb

        opt = bnb.optim.AdamW8bit(trainable, lr=2e-4)
    else:
        opt = torch.optim.AdamW(trainable, lr=2e-4)
    model.train()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    smi_floor = smi.peak
    losses, step_s, k = [], [], 0
    for step in range(w.steps):
        torch.cuda.synchronize()
        ts = time.time()
        opt.zero_grad(set_to_none=True)
        acc = 0.0
        for _ in range(w.grad_accum):
            ids = torch.tensor(blocks[k:k + w.micro_batch], device="cuda")
            k += w.micro_batch
            loss = model(input_ids=ids, labels=ids).loss
            (loss / w.grad_accum).backward()
            acc += float(loss.detach()) / w.grad_accum
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        torch.cuda.synchronize()
        step_s.append(time.time() - ts)
        losses.append(acc)
        if step == 0 or (step + 1) % 5 == 0 or step + 1 == w.steps:
            log(f"  step {step + 1}/{w.steps}  loss {acc:.4f}  {step_s[-1]:.2f}s  "
                f"peak {torch.cuda.max_memory_allocated() / GiB:.2f} GiB", flush=True)
    meas.update(device_peak_bytes=torch.cuda.max_memory_allocated(),
                device_reserved_peak_bytes=torch.cuda.max_memory_reserved())
    time.sleep(2 * smi.interval)                          # let the sampler see the end state
    meas["driver_process_peak_bytes"] = smi.peak or None
    meas["driver_samples"] = smi.samples
    meas["driver_peak_before_training_bytes"] = smi_floor or None
    if smi.peak:
        meas["cuda_context_bytes"] = smi.peak - meas["device_reserved_peak_bytes"]
    meas["host_peak_bytes"] = proc_status().get("VmHWM")
    meas["host_anon_peak_bytes"] = smi.anon_peak or None
    meas["host_file_peak_bytes"] = smi.file_peak or None
    timed = step_s[warmup:] or step_s
    meas.update(step_seconds=step_s, s_per_step_median=statistics.median(timed),
                tokens_per_s=w.tokens_per_microbatch * w.grad_accum / statistics.median(timed),
                timed_steps=f"{warmup + 1}..{w.steps}")
    digest_after = _expert_digest(model)
    b_norm_after = b_norm()
    third = max(1, len(losses) // 3)
    correctness = {
        "losses": losses,
        "all_finite": all(math.isfinite(x) for x in losses),
        "loss_first_third_mean": statistics.mean(losses[:third]),
        "loss_last_third_mean": statistics.mean(losses[-third:]),
        "loss_decreased": statistics.mean(losses[-third:]) < statistics.mean(losses[:third]),
        "frozen_expert_bytes_unchanged": digest_before == digest_after,
        "frozen_expert_digest": digest_after,
        "adapter_B_norm_before": b_norm_before,
        "adapter_B_norm_after": b_norm_after,
        "adapters_moved": b_norm_after > b_norm_before,
    }
    ok = correctness["all_finite"] and correctness["frozen_expert_bytes_unchanged"] and correctness["adapters_moved"]
    return {"measured": meas, "correctness": correctness, "data": data_info, "status": "OK" if ok else "ALARM"}
