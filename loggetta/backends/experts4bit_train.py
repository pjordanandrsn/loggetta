"""Execute a feasible experts4bit training plan, measuring what the plan estimated.

The model is built by ``experts4bit_qlora.recipe.prepare_qlora_training`` from exactly the setup the plan selected,
so the run executes what was priced. Everything model-shaped (loading, the 4-bit expert stores, adapters, the fused
engines, offload, kernels) is experts4bit-qlora's and comes from that one call. What is here is the measurement
around it: the data, a deliberately plain loop (AdamW, gradient accumulation, clip 1.0) on fixed-shape packed blocks,
timing, memory sampling, and the integrity checks a receipt records. Fixed shapes make peak memory and step time
comparable across runs and with the plan.

One check reads experts4bit-qlora's storage directly: ``_expert_digest`` hashes the frozen expert bytes by the
attribute and buffer names of ``ExpertsNbit`` and of the offload handles' host homes.
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


#: The buffers experts4bit-qlora's ``compress_expert_absmax_`` stores in place of one projection's fp32 absmax
#: (``<which>_absmax_q`` / ``_s`` / ``_off`` / ``_code``, every release since 0.49.0). It is the default for resident
#: ``grouped_nf4`` training, and it leaves a guard under the old ``<which>_absmax`` name that raises on any use.
_COMPRESSED_ABSMAX = ("_absmax_q", "_absmax_s", "_absmax_off", "_absmax_code")


def _stored_expert_tensors(base) -> dict:
    """What one ``ExpertsNbit`` stack STORES for its frozen weights: the packed codes and, per projection, the absmax as
    it is held -- the fp32 buffer, or the double-quantized payload when experts4bit-qlora compressed it. Nothing is
    decompressed: the digest is of the stored bytes. Without compression the names, and so the digest, are as before."""
    out = {n: getattr(base, n) for n in ("gate_up_proj", "down_proj")}
    for which in ("gate_up", "down"):
        if which + _COMPRESSED_ABSMAX[0] in base._buffers:
            out.update({which + s: base._buffers[which + s] for s in _COMPRESSED_ABSMAX})
        else:
            out[which + "_absmax"] = getattr(base, which + "_absmax")
    return out


def _expert_digest(model, layers=(0, -1)) -> dict:
    """sha256 of the frozen expert bytes as stored (packed codes + absmax, fp32 or double-quantized) of a few stacks,
    wherever they live (device or host home)."""
    import torch
    from experts4bit_qlora import ExpertsNbit, offload_handles

    bases = [m for m in model.modules() if isinstance(m, ExpertsNbit)]
    handles = list(offload_handles(model))                      # one per layer under host residency, else empty
    out = {}
    for li in layers:
        base = bases[li]
        tensors = _stored_expert_tensors(base)
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


def run(plan, *, seed: int = 0, warmup: int = 2, log=print, adapter_dir: str | None = None) -> dict:
    """Prepare user data, build the selected setup, train, then export reusable native adapters.

    Data validation/tokenization precedes model loading. Adapter export happens after training measurement, so
    save-time allocations do not change the meaning of historical training peaks and step timings.
    """
    import torch
    from transformers import AutoTokenizer
    from experts4bit_qlora.recipe import QLoRASetup, prepare_qlora_training

    from ..data import TrainingData, prepare_data
    from ..measure import DriverMemorySampler, proc_status
    from .experts4bit_adapters import save_adapter, validate_target

    target = validate_target(adapter_dir) if adapter_dir is not None else None
    spec = TrainingData.from_dict(plan.workload.data)
    if plan.workload.data is None:
        log("No dataset supplied: using the documented Alpaca demonstration, repeated as needed.", flush=True)
    tok = AutoTokenizer.from_pretrained(plan.model["model"], revision=plan.model.get("revision"))
    t_data = time.perf_counter()
    n_blocks = plan.workload.steps * plan.workload.grad_accum * plan.workload.micro_batch
    epochs = getattr(plan.workload, "epochs", None)
    # a plan made from a data profile re-reads the data and refuses to go on if it is not what was planned
    with prepare_data(tok, n_blocks, plan.workload.seq_len, spec, seed=seed,
                      cache_dir=str(target.parent) if target is not None else None,
                      expect=getattr(plan, "data_profile", None),
                      max_passes=math.ceil(epochs) if epochs and not spec.repeat else None) as prepared:
        data_seconds = time.perf_counter() - t_data
        log(f"Data ready: {prepared.info['tokens']} tokens from {prepared.info['examples_used']} rows; "
            f"{prepared.info['passes']} pass(es); {prepared.info['loss']}.", flush=True)
        torch.manual_seed(seed)
        torch.cuda.set_device(plan.constraints.device)
        torch.zeros(1, device="cuda")
        meas = {"host_baseline_bytes": proc_status().get("VmRSS"), **link_h2d(),
                "data_prepare_seconds": data_seconds, "data_token_cache_bytes": prepared.info["token_cache_bytes"]}
        with DriverMemorySampler() as smi:
            t0 = time.time()
            prep = prepare_qlora_training(plan.model["model"], QLoRASetup(**plan.selected.setup), device="cuda",
                                          revision=plan.model.get("revision"))
            meas["load_seconds"] = time.time() - t0
            meas["load_device_peak_bytes"] = torch.cuda.max_memory_allocated()
            meas["host_anon_after_load_bytes"] = proc_status().get("RssAnon")
            out = train_loop(prep.model, prep.trainable, plan.model["model"], plan.workload, smi, meas,
                             revision=plan.model.get("revision"), seed=seed, warmup=warmup, log=log,
                             prepared_data=prepared)
        out["engaged"] = prep.report
        if target is not None:
            out["artifacts"] = {}
            if out["status"] != "OK":
                out["artifact_error"] = "integrity checks failed; no reusable adapter was exported"
            else:
                save_start = time.perf_counter()
                try:
                    out["artifacts"]["adapter"] = save_adapter(prep.model, tok, target, plan,
                                                                data=out["data"], report=prep.report, seed=seed)
                except (OSError, ValueError, RuntimeError) as exc:
                    out["status"] = "SAVE_FAILED"
                    out["artifact_error"] = f"{type(exc).__name__}: {exc}"
                out["measured"]["adapter_save_seconds"] = time.perf_counter() - save_start
        return out


def train_loop(model, trainable, model_id, w, smi, meas, *, revision=None, seed=0, warmup=2, log=print,
               prepared_data=None, device="cuda", frozen_digest=None, frozen_kind="expert") -> dict:
    """The measured loop, shared by every arm that must be comparable: fixed-shape packed blocks, configurable AdamW learning rate,
    gradient accumulation, clip 1.0. Fills ``meas`` and returns the correctness record."""
    import torch
    from transformers import AutoTokenizer

    from ..measure import proc_status

    if prepared_data is None:
        # Legacy direct benchmark arms retain the original demo data path. User execution always supplies
        # prevalidated data from run(), including for the demo, so malformed data fails before loading weights.
        if w.data is not None:
            raise ValueError("custom data must be prepared before the model loads")
        tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
        blocks, data_info = packed_blocks(tok, w.steps * w.grad_accum * w.micro_batch, w.seq_len, seed=seed)
    else:
        blocks, data_info = prepared_data.blocks, prepared_data.info
    is_cuda = torch.device(device).type == "cuda"
    sync = torch.cuda.synchronize if is_cuda else lambda: None
    digest = frozen_digest or _expert_digest
    digest_before = digest(model)
    b_norm = lambda: sum(float(p.detach().float().norm()) for n, p in model.named_parameters()  # noqa: E731
                         if p.requires_grad and "lora_B" in n)
    b_norm_before = b_norm()
    torch.manual_seed(seed)
    if w.optimizer == "adamw_8bit":
        import bitsandbytes as bnb

        opt = bnb.optim.AdamW8bit(trainable, lr=w.learning_rate)
    else:
        opt = torch.optim.AdamW(trainable, lr=w.learning_rate)
    from ..schedule import describe as describe_lr, learning_rates

    lrs = learning_rates(w)
    meas.update(lr_schedule=describe_lr(w), lr_first=lrs[0], lr_peak=max(lrs), lr_last=lrs[-1])
    model.train()
    sync()
    if is_cuda:
        torch.cuda.reset_peak_memory_stats()
    smi_floor = smi.peak
    # a loss mask (1 = trained) beside the blocks; None trains every token, exactly as before masks existed
    masks = getattr(prepared_data, "mask", None) if prepared_data is not None else None
    # isolated packing: positions restart at 0 for each example; transformers then masks attention per example, but
    # only when no KV cache exists, so the call says use_cache=False rather than trusting the model's config
    positions = getattr(prepared_data, "positions", None) if prepared_data is not None else None
    losses, step_s, k, skipped = [], [], 0, 0
    for step in range(w.steps):
        sync()
        ts = time.time()
        opt.zero_grad(set_to_none=True)
        acc = 0.0
        counts = total = None
        if masks is not None:
            # trained targets per micro-batch (a row's first token is never a target: labels shift by one). The step's
            # gradient is the mean over all of them, so each micro-batch's mean loss is weighted by its share
            counts = [int(masks[k + i * w.micro_batch:k + (i + 1) * w.micro_batch, 1:].sum())
                      for i in range(w.grad_accum)]
            total = sum(counts)
            if total == 0:                                   # nothing in this step trains: no update
                k += w.grad_accum * w.micro_batch
                skipped += 1
                continue
        for i in range(w.grad_accum):
            ids = torch.tensor(blocks[k:k + w.micro_batch], dtype=torch.long, device=device)
            if masks is not None and counts[i] == 0:
                k += w.micro_batch
                continue
            labels = ids if masks is None else ids.masked_fill(
                torch.tensor(masks[k:k + w.micro_batch], device=device) == 0, -100)
            extra = {} if positions is None else {
                "position_ids": torch.tensor(positions[k:k + w.micro_batch], dtype=torch.long, device=device),
                "use_cache": False}
            k += w.micro_batch
            loss = model(input_ids=ids, labels=labels, **extra).loss
            if not torch.isfinite(loss):
                raise ValueError(f"non-finite loss at step {step + 1}; training stopped before an optimizer update")
            if masks is None:
                (loss / w.grad_accum).backward()
                acc += float(loss.detach()) / w.grad_accum
            else:
                (loss * (counts[i] / total)).backward()
                acc += float(loss.detach()) * counts[i] / total
        torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
        for group in opt.param_groups:
            group["lr"] = lrs[step]
        opt.step()
        sync()
        step_s.append(time.time() - ts)
        losses.append(acc)
        if step == 0 or (step + 1) % 5 == 0 or step + 1 == w.steps:
            log(f"  step {step + 1}/{w.steps}  loss {acc:.4f}  {step_s[-1]:.2f}s  "
                f"peak {(torch.cuda.max_memory_allocated() / GiB if is_cuda else 0):.2f} GiB", flush=True)
    if not losses:
        raise ValueError("no optimizer step had a token to train on under the loss mask")
    meas.update(device_peak_bytes=torch.cuda.max_memory_allocated() if is_cuda else None,
                device_reserved_peak_bytes=torch.cuda.max_memory_reserved() if is_cuda else None)
    time.sleep(2 * smi.interval)                          # let the sampler see the end state
    meas["driver_process_peak_bytes"] = smi.peak or None
    meas["driver_samples"] = smi.samples
    meas["driver_peak_before_training_bytes"] = smi_floor or None
    if smi.peak and is_cuda:
        meas["cuda_context_bytes"] = smi.peak - meas["device_reserved_peak_bytes"]
    meas["host_peak_bytes"] = proc_status().get("VmHWM")
    meas["host_anon_peak_bytes"] = smi.anon_peak or None
    meas["host_shmem_peak_bytes"] = smi.shmem_peak or None
    meas["host_file_peak_bytes"] = smi.file_peak or None
    meas["host_required_peak_bytes"] = smi.required_peak or None
    timed = step_s[warmup:] or step_s
    meas.update(step_seconds=step_s, s_per_step_median=statistics.median(timed),
                tokens_per_s=w.tokens_per_microbatch * w.grad_accum / statistics.median(timed),
                timed_steps=f"{warmup + 1 if len(step_s) > warmup else 1}..{w.steps}")
    digest_after = digest(model)
    b_norm_after = b_norm()
    third = max(1, len(losses) // 3)
    correctness = {
        "losses": losses,
        "all_finite": all(math.isfinite(x) for x in losses),
        "loss_first_third_mean": statistics.mean(losses[:third]),
        "loss_last_third_mean": statistics.mean(losses[-third:]),
        "loss_decreased": statistics.mean(losses[-third:]) < statistics.mean(losses[:third]),
        f"frozen_{frozen_kind}_bytes_unchanged": bool(digest_before) and digest_before == digest_after,
        f"frozen_{frozen_kind}_digest": digest_after,
        f"frozen_{frozen_kind}_checked_stacks": list(digest_before),
        "adapter_B_norm_before": b_norm_before,
        "adapter_B_norm_after": b_norm_after,
        "adapters_moved": b_norm_after > b_norm_before,
        "steps_without_trained_tokens": skipped,
    }
    ok = correctness["all_finite"] and correctness[f"frozen_{frozen_kind}_bytes_unchanged"] and correctness["adapters_moved"]
    return {"measured": meas, "correctness": correctness, "data": data_info, "status": "OK" if ok else "ALARM"}
