"""Check a serve plan against the paged server it describes: plan, build ``serve_paged``'s engine with exactly the
plan's environment (``ServeSetup.to_env``), decode, and write an ``execution-receipt/1`` with the measured peaks
beside the estimate. At ``--concurrency 1`` it also traces the engine's steps (``E4B_PAGED_STEP_TRACE``) and records
the single-stream decode step (``measured.decode_step_ms_b1``, :func:`loggetta.measure.decode_step_b1`): the figure a
plan of this same setup on this GPU, driver and experts4bit-qlora version may then quote, labelled measured.

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

from loggetta import shareable  # noqa: E402
from loggetta.measure import provenance  # noqa: E402

PROV = provenance()                                  # at process start, before anything below is imported
GiB = 1 << 30


def _sha256(path, limit=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read() if limit is None else f.read(limit))
    return h.hexdigest()


def bake(model, arena, log=print, **kw):
    from huggingface_hub import snapshot_download
    from nvme_bake_nf4 import bake_nf4

    snap = snapshot_download(model, local_files_only=True, allow_patterns=["*.json", "*.safetensors"])
    t0 = time.time()
    bake_nf4(snap, arena, log=log, **kw)
    return {"snapshot": snap, "seconds": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--arena", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--bake", action="store_true")
    ap.add_argument("--bake-kw", action="append", default=[], help="KEY=VALUE passed to bake_nf4, a comma making a "
                    "tuple (GraniteMoe: fused_marker=.block_sparse_moe. and "
                    "fused_proj=input_linear.weight,output_linear.weight)")
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--prompt-tokens", type=int, default=1024, help="per request; > chunk exercises chunked prefill")
    ap.add_argument("--new-tokens", type=int, default=64)
    ap.add_argument("--fix", action="append", default=[], help="FIELD=VALUE for the serve setup (expert mode)")
    ap.add_argument("--hardware", help="saved hardware profile (default: probe this machine)")
    ap.add_argument("--vram-gib", type=float, help="device budget for the plan (default: free now)")
    ap.add_argument("--ram-gib", type=float, help="host budget for the plan (default: available now)")
    ap.add_argument("--observations", action="append", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--env", action="append", default=[], help="E4B_PAGED_*=VALUE passed to the server as is (e.g. the "
                    "solver's E4B_PAGED_VRAM_GB / _DRAM_GB / _HOT_ROWS, which ServeSetup does not carry)")
    ap.add_argument("--measure-refused", action="store_true",
                    help="build and measure even when the plan refuses (a calibration run for an unpriced setup)")
    a = ap.parse_args()

    from loggetta import Constraints, Workload, describe_model, plan
    from loggetta.cli import _parse_fixed
    from loggetta.execution import load_observations
    from loggetta.hardware import HardwareProfile, probe

    baked = None
    if not os.path.exists(a.arena):
        if not a.bake:
            sys.exit(f"{a.arena} does not exist (pass --bake to write it)")
        kw = {k: (tuple(v.split(",")) if "," in v else v) for k, v in (x.split("=", 1) for x in a.bake_kw)}
        baked = {**bake(a.model, a.arena, **kw), "bake_kw": kw}
    hw = HardwareProfile.from_dict(json.load(open(a.hardware)), origin=a.hardware) if a.hardware else probe()
    obs = [o for d in a.observations for o in load_observations(d)]
    w = Workload(kind="serve", context_len=a.context, concurrency=a.concurrency)
    gib = lambda x: None if x is None else int(x * 2**30)  # noqa: E731
    p = plan(describe_model(a.model), hw, w, Constraints(fixed=_parse_fixed(a.fix), vram_budget=gib(a.vram_gib),
                                                       ram_budget=gib(a.ram_gib)), observations=obs)
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
    if p.status != "feasible" and not a.measure_refused:
        receipt.update(status="REFUSED", refusal=p.refusal)
        json.dump(shareable.record(receipt), open(out_path, "w"), indent=1)
        print("refused; receipt", out_path)
        return

    from experts4bit_qlora.serve_recipe import ServeSetup
    from loggetta.backends.experts4bit_serve import measure_serve
    from loggetta.measure import changed_since

    cand = p.selected or (p.alternatives[0] if p.alternatives else None)
    if cand is None:
        sys.exit("no candidate setup to build")
    setup = ServeSetup(**cand.setup)
    extra = dict(kv.split("=", 1) for kv in a.env)
    env = {**setup.to_env(), **extra, "E4B_PAGED_MODEL": a.model, "E4B_PAGED_ARENA": a.arena,
           "E4B_PAGED_CALIB": a.calib}
    # the build, decode and measurement are the package's serve executor's (one place for both)
    out = measure_serve(env, concurrency=a.concurrency, context_len=a.context, prompt_tokens=a.prompt_tokens,
                        new_tokens=a.new_tokens, trace_path=os.path.join(a.out, run_id + ".steps.jsonl"))
    meas, placement = out["measured"], out["placement"]
    alloc, driver_peak = meas["device_peak_bytes"], meas["driver_process_peak_bytes"]
    gpu = hw.gpu(0)
    est = {ln.name: ln.bytes for ln in cand.lines if ln.where == "device"}
    backend_alloc = sum(b for n, b in est.items() if n not in ("allocator reserve (cached, unallocated blocks)",
                                                               "CUDA context + library workspaces"))
    receipt.update(
        status="OK", setup={**cand.setup, **{k[len("E4B_PAGED_"):].lower(): v for k, v in extra.items()}}, env=env,
        planned=p.status == "feasible", placement=placement, info=out["info"],
        hardware={"gpu": {"name": gpu.name, "driver": gpu.driver.value, "memory_total": gpu.memory_total.value}},
        measured=meas,
        estimate={"device_lines": est, "allocator_estimate_bytes": backend_alloc, "device_total_bytes": cand.device_bytes},
        comparison={"allocator_peak_minus_estimate_gib": round((alloc - backend_alloc) / GiB, 3),
                    "driver_peak_minus_plan_total_gib": round((driver_peak - cand.device_bytes) / GiB, 3)
                    if driver_peak else None},
    )
    receipt["provenance"]["changed_during_run"] = changed_since(PROV)
    json.dump(shareable.record(receipt), open(out_path, "w"), indent=1, default=str)
    print(json.dumps({"measured": meas, "comparison": receipt["comparison"]}, indent=1, default=str))
    print("receipt", out_path)


if __name__ == "__main__":
    main()
