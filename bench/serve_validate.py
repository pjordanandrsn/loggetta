"""Check a serve plan against the paged server it describes: plan, build ``serve_paged``'s engine with exactly the
plan's environment (``ServeSetup.to_env``), decode, and write an ``execution-receipt/1`` with the measured peaks
beside the estimate.

    python bench/serve_validate.py allenai/OLMoE-1B-7B-0924 --arena ARENA --calib CALIB --context 4096 \
        --concurrency 4 --hardware evidence/.../hardware-profile.json --observations evidence/... --out runs/receipts

``--bake`` writes the NF4 arena first when ``ARENA`` does not exist (grouped-nf4-gemm's ``bake_nf4``, on the GPU).
Under the all-VRAM placement the calibration blob only feeds the solver whose answer is then overridden, so any
valid blob serves; the receipt records which.
"""
import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from loggetta.measure import provenance  # noqa: E402

PROV = provenance()                                  # at process start, before anything below is imported
GiB = 1 << 30


def _sha256(path, limit=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read() if limit is None else f.read(limit))
    return h.hexdigest()


def bake(model, arena, log=print):
    from huggingface_hub import snapshot_download
    from nvme_bake_nf4 import bake_nf4

    snap = snapshot_download(model, local_files_only=True)
    t0 = time.time()
    bake_nf4(snap, arena, log=log)
    return {"snapshot": snap, "seconds": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--arena", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--bake", action="store_true")
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--prompt-tokens", type=int, default=1024, help="per request; > chunk exercises chunked prefill")
    ap.add_argument("--new-tokens", type=int, default=64)
    ap.add_argument("--fix", action="append", default=[], help="FIELD=VALUE for the serve setup (expert mode)")
    ap.add_argument("--hardware", help="saved hardware profile (default: probe this machine)")
    ap.add_argument("--observations", action="append", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-id", default=None)
    a = ap.parse_args()

    from loggetta import Constraints, Workload, describe_model, plan
    from loggetta.cli import _parse_fixed
    from loggetta.hardware import HardwareProfile, probe
    from loggetta.runtime import load_observations

    baked = None
    if not os.path.exists(a.arena):
        if not a.bake:
            sys.exit(f"{a.arena} does not exist (pass --bake to write it)")
        baked = bake(a.model, a.arena)
    hw = HardwareProfile.from_dict(json.load(open(a.hardware)), origin=a.hardware) if a.hardware else probe()
    obs = [o for d in a.observations for o in load_observations(d)]
    w = Workload(kind="serve", context_len=a.context, concurrency=a.concurrency)
    p = plan(describe_model(a.model), hw, w, Constraints(fixed=_parse_fixed(a.fix)), observations=obs)
    print(p.render())
    run_id = a.run_id or f"serve-{a.model.split('/')[-1]}-c{a.context}x{a.concurrency}-{time.strftime('%Y%m%dT%H%M%S')}"
    receipt = {"schema": "execution-receipt/1", "run_id": run_id, "plan": json.loads(p.to_json()),
               "model": {"model": a.model, "revision": p.model.get("revision")},
               "workload": {"kind": "serve", "context_len": a.context, "concurrency": a.concurrency,
                            "prompt_tokens": a.prompt_tokens, "new_tokens": a.new_tokens},
               "provenance": {**PROV, "arena": a.arena, "arena_head_sha256": _sha256(a.arena, 1 << 20),
                              "calib": a.calib, "calib_sha256": _sha256(a.calib), "baked": baked}}
    os.makedirs(a.out, exist_ok=True)
    out_path = os.path.join(a.out, run_id + ".json")
    if p.status != "feasible":
        receipt.update(status="REFUSED", refusal=p.refusal)
        json.dump(receipt, open(out_path, "w"), indent=1)
        print("refused; receipt", out_path)
        return

    import torch

    from experts4bit_qlora.serve_recipe import ServeSetup
    from loggetta.measure import DriverMemorySampler, changed_since, proc_status

    setup = ServeSetup(**p.selected.setup)
    env = {**setup.to_env(), "E4B_PAGED_MODEL": a.model, "E4B_PAGED_ARENA": a.arena, "E4B_PAGED_CALIB": a.calib}
    os.environ.update(env)
    torch.cuda.set_device(0)
    torch.zeros(1, device="cuda")
    meas = {"host_baseline_bytes": proc_status().get("VmRSS")}
    from experts4bit_qlora.serve_paged import PagedServeConfig, build_engine

    cfg = PagedServeConfig.from_env()
    with DriverMemorySampler() as smi:
        t0 = time.time()
        parts = build_engine(cfg)
        torch.cuda.synchronize()
        meas.update(load_seconds=round(time.time() - t0, 1), load_device_peak_bytes=torch.cuda.max_memory_allocated(),
                    load_reserved_peak_bytes=torch.cuda.max_memory_reserved(),
                    host_anon_after_load_bytes=proc_status().get("RssAnon"))
        g = torch.Generator().manual_seed(0)
        vocab = int(getattr(parts.tokenizer, "vocab_size", 0) or 32000)
        n_prompt = min(a.prompt_tokens, a.context - a.new_tokens - 1)
        for _ in range(a.concurrency):
            ids = torch.randint(10, vocab - 10, (n_prompt,), generator=g).tolist()
            parts.scheduler.add_request(ids, max_new_tokens=a.new_tokens)
        t1 = time.time()
        steps = parts.scheduler.run_until_idle()
        torch.cuda.synchronize()
        dt = time.time() - t1
    alloc, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
    meas.update(device_peak_bytes=alloc, device_reserved_peak_bytes=reserved, driver_process_peak_bytes=smi.peak or None,
                driver_samples=smi.samples, cuda_context_bytes=(smi.peak - reserved) if smi.peak else None,
                host_required_peak_bytes=smi.required_peak, scheduler_steps=steps, generate_seconds=round(dt, 2),
                tokens_per_s=round(a.concurrency * a.new_tokens / dt, 1) if dt else None)
    gpu = hw.gpu(0)
    est = {ln.name: ln.bytes for ln in p.selected.lines if ln.where == "device"}
    backend_alloc = sum(b for n, b in est.items() if n not in ("allocator reserve (cached, unallocated blocks)",
                                                               "CUDA context + library workspaces"))
    receipt.update(
        status="OK", setup=p.selected.setup, env=env, info={k: v for k, v in (parts.info or {}).items()
                                                             if isinstance(v, (int, float, str, bool, dict, list))},
        hardware={"gpu": {"name": gpu.name, "driver": gpu.driver.value, "memory_total": gpu.memory_total.value}},
        measured=meas,
        estimate={"device_lines": est, "allocator_estimate_bytes": backend_alloc, "device_total_bytes": p.selected.device_bytes},
        comparison={"allocator_peak_minus_estimate_gib": round((alloc - backend_alloc) / GiB, 3),
                    "driver_peak_minus_plan_total_gib": round(((smi.peak or 0) - p.selected.device_bytes) / GiB, 3)
                    if smi.peak else None},
    )
    receipt["provenance"]["changed_during_run"] = changed_since(PROV)
    json.dump(receipt, open(out_path, "w"), indent=1, default=str)
    print(json.dumps({"measured": meas, "comparison": receipt["comparison"]}, indent=1, default=str))
    print("receipt", out_path)


if __name__ == "__main__":
    main()
