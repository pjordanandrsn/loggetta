"""The experts4bit serve executor: build ``serve_paged``'s engine with exactly a serve plan's environment
(``ServeSetup.to_env``), decode a fixed request set, and report what the run measured.

``measure_serve`` is the GPU part (torch and experts4bit-qlora, imported when it runs); ``run`` is what ``execute`` calls.
It checks the inputs a serve run needs and refuses by name before anything loads, then hands the engine part the
environment and shapes the result. ``bench/serve_validate.py`` uses the same ``measure_serve``.

The NF4 arena is an input, never built here: baking it is an explicit, separate step (``bench/serve_validate.py --bake``).
A receipt names the arena by base name and the sha256 of its first MiB, and the calibration file by base name and
sha256, so a shared receipt carries no path: the server environment it records leaves those two out and names a local
model directory by its name only (``local: <name>``, the rule ``scripts/first_run.sh`` applies).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

GiB = 1 << 30
#: the request set ``run`` decodes when the caller names none: one prompt of this many tokens per sequence, each
#: decoding this many new tokens (bench/serve_validate.py's defaults, so its receipts and these compare)
PROMPT_TOKENS, NEW_TOKENS = 1024, 64


def _sha256(path, limit=None) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read() if limit is None else f.read(limit))
    return h.hexdigest()


def serve_env(setup: dict, model: str, arena: str, calib: str) -> dict:
    """The server environment a serve plan's setup describes, as experts4bit-qlora writes it (``ServeSetup.to_env``),
    plus the model, arena and calibration it serves."""
    from experts4bit_qlora.serve_recipe import ServeSetup

    from .experts4bit import _serve_setup

    return {**_serve_setup(ServeSetup, setup).to_env(), "E4B_PAGED_MODEL": model, "E4B_PAGED_ARENA": arena,
            "E4B_PAGED_CALIB": calib}


def measure_serve(env: dict, *, concurrency: int, context_len: int, prompt_tokens: int, new_tokens: int, seed: int = 0,
                  trace_path: str | None = None, log=print) -> dict:
    """Build the engine under ``env`` on GPU 0, decode ``concurrency`` random prompts of ``prompt_tokens`` tokens for
    ``new_tokens`` each, and return ``{"measured", "info", "placement", "requests"}``. At concurrency 1 with a
    ``trace_path``, the engine's own step trace gives the single-stream decode step (:func:`measure.decode_step_b1`)."""
    import torch

    from ..measure import DriverMemorySampler, decode_step_b1, proc_status

    os.environ.update(env)                       # PagedServeConfig.from_env reads the process environment only
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
        smi.mark(meas["host_anon_after_load_bytes"] or 0)
        tracer = None
        if concurrency == 1 and trace_path:          # the single-stream decode step, from the engine's own step trace
            try:
                from experts4bit_qlora.engines.step_trace import StepTrace
            except ImportError:
                StepTrace = None
            if StepTrace is not None and hasattr(parts.scheduler, "tracer"):
                tracer = StepTrace(trace_path, cuda=True)
                parts.scheduler.tracer = tracer
                if parts.runner is not None and hasattr(parts.runner, "tracer"):
                    parts.runner.tracer = tracer
        g = torch.Generator().manual_seed(seed)
        vocab = int(getattr(parts.tokenizer, "vocab_size", 0) or 32000)
        n_prompt = min(prompt_tokens, context_len - new_tokens - 1)
        for _ in range(concurrency):
            ids = torch.randint(10, vocab - 10, (n_prompt,), generator=g).tolist()
            parts.scheduler.add_request(ids, max_new_tokens=new_tokens)
        t1 = time.time()
        steps = parts.scheduler.run_until_idle()
        torch.cuda.synchronize()
        dt = time.time() - t1
        if tracer is not None:
            tracer.close()
            b1 = decode_step_b1([json.loads(ln) for ln in open(trace_path) if ln.strip()])
            if b1:
                meas.update(b1, step_trace=os.path.basename(trace_path))
    alloc, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
    meas.update(device_peak_bytes=alloc, device_reserved_peak_bytes=reserved, driver_process_peak_bytes=smi.peak or None,
                driver_samples=smi.samples, cuda_context_bytes=(smi.peak - reserved) if smi.peak else None,
                host_required_peak_bytes=smi.required_peak, scheduler_steps=steps, generate_seconds=round(dt, 2),
                tokens_per_s=round(concurrency * new_tokens / dt, 1) if dt else None)
    placement = None
    if cfg.placement != "all-vram":
        from experts4bit_qlora.engines.placement import solve_placement
        from experts4bit_qlora.serve_paged import _bytes_per_expert

        inf = parts.info or {}
        man = solve_placement(n_layers=inf["moe_layers"], n_experts=inf["experts"],
                              bytes_per_expert=_bytes_per_expert(cfg.arena), vram_budget_bytes=int(cfg.vram_gb * GiB),
                              dram_budget_bytes=int(cfg.dram_gb * GiB), calibration=json.load(open(cfg.calib)),
                              profile_path=None, batch=1, top_k=inf.get("top_k"))
        placement = {"tiers": {k: len(v) for k, v in man["tiers"].items()}, "bytes_per_expert": _bytes_per_expert(cfg.arena),
                     "vram_gb": cfg.vram_gb, "dram_gb": cfg.dram_gb, "hot_rows": cfg.hot_rows, "masses": man["masses"]}
    meas.update(host_anon_peak_bytes=smi.anon_peak, host_shmem_peak_bytes=smi.shmem_peak, host_file_peak_bytes=smi.file_peak,
                host_anon_serving_peak_bytes=smi.phase_anon_peak)
    try:                                         # which expert-GEMM route each call took (eager calls and captures)
        from experts4bit_qlora.engines.hot_residency import ROUTE_SEEN
        meas["expert_routes_seen"] = dict(ROUTE_SEEN)
    except ImportError:
        pass
    info = {k: v for k, v in (parts.info or {}).items() if isinstance(v, (int, float, str, bool, dict, list))}
    return {"measured": meas, "info": info, "placement": placement,
            "requests": {"count": concurrency, "prompt_tokens": n_prompt, "new_tokens": new_tokens}}


def model_label(model: str) -> str:
    """A model id as a shared record may carry it: a Hub id (``org/name``) as given, a local model directory (an
    absolute, "."- or "~"-relative path, or an existing directory) by its name only, as ``local: <name>``."""
    if os.path.isabs(model) or model.startswith((".", "~")) or os.path.isdir(model):
        return "local: " + os.path.basename(model.rstrip("/"))
    return model


def check_inputs(plan, arena, calib, prompt_tokens: int, new_tokens: int) -> None:
    """Refuse by name, before anything loads, a serve run that cannot start or would not mean what its plan says."""
    from ..execution import PlanNotExecutable

    if plan.constraints.allow_development_executor is not True:
        from .experts4bit import planned_only_reason

        raise PlanNotExecutable(planned_only_reason("serve"))
    if not arena:
        raise PlanNotExecutable("a serve run needs the model's NF4 arena (--arena): bake it first, as a separate step "
                                "(bench/serve_validate.py --bake); execute never bakes one")
    if not os.path.isfile(arena):
        raise PlanNotExecutable(f"no NF4 arena at {os.path.basename(arena)!r} (--arena): bake it first, as a separate step")
    if not calib or not os.path.isfile(calib):
        raise PlanNotExecutable("a serve run needs the expert calibration file (--calib), which the placement reads")
    ctx = plan.workload.context_len or 0
    if prompt_tokens < 1 or new_tokens < 1 or prompt_tokens + new_tokens >= ctx:
        raise PlanNotExecutable(f"{prompt_tokens} prompt + {new_tokens} new tokens must fit inside the plan's context of "
                                f"{ctx} tokens")


def run(plan, *, seed, log, adapter_dir, run_id, arena=None, calib=None, prompt_tokens=PROMPT_TOKENS,
        new_tokens=NEW_TOKENS, engine=None):
    """Decode a feasible serve plan's request set through experts4bit-qlora's paged engine. ``adapter_dir``'s parent is
    the run's directory (the step trace goes there). ``engine`` replaces :func:`measure_serve` (tests)."""
    check_inputs(plan, arena, calib, prompt_tokens, new_tokens)
    env = serve_env(plan.selected.setup, plan.model["model"], arena, calib)
    run_dir = Path(adapter_dir).parent
    os.makedirs(run_dir, exist_ok=True)
    trace = str(run_dir / "steps.jsonl") if plan.workload.concurrency == 1 else None
    out = (engine or measure_serve)(env, concurrency=plan.workload.concurrency, context_len=plan.workload.context_len,
                                    prompt_tokens=prompt_tokens, new_tokens=new_tokens, seed=seed, trace_path=trace,
                                    log=log)
    files = {"arena": os.path.basename(arena), "arena_head_sha256": _sha256(arena, 1 << 20),
             "calib": os.path.basename(calib), "calib_sha256": _sha256(calib)}
    # the paths stay out: the arena and calibration file appear above by name, and a local model directory by its name
    knobs = {k: v for k, v in env.items() if k not in ("E4B_PAGED_ARENA", "E4B_PAGED_CALIB")}
    knobs["E4B_PAGED_MODEL"] = model_label(knobs.get("E4B_PAGED_MODEL", ""))
    return {"status": "OK", "measured": out["measured"], "correctness": {},
            "engaged": {"placement": out.get("placement"), "info": out.get("info") or {}},
            "data": {**out.get("requests", {}), **files, "server_env": knobs}}
