"""Plan first, execute second: ``execute(plan)`` runs a feasible plan through its backend and writes a receipt.

The receipt keeps the plan's estimate and the run's observations side by side and never merges them: every
comparison is ``measured - estimated`` with both numbers present, so a reader can see how good the estimate was.
Schema ``execution-receipt/1``; field names follow experts4bit-qlora's benchmark-receipt schema where they overlap
(``commit_sha`` lives under ``provenance.sources``, ``status`` uses its vocabulary), so the two can be aggregated.
"""
from __future__ import annotations

import json
import os
import time

from .measure import changed_since, provenance
from .plan import ExecutionPlan

RECEIPT_SCHEMA = "execution-receipt/1"
GiB = 1 << 30


class PlanNotExecutable(RuntimeError):
    pass


def _executor(backend: str):
    if backend == "experts4bit":
        from .backends import experts4bit_train

        return experts4bit_train.run
    raise PlanNotExecutable(f"no executor for backend {backend!r}")


def compare(plan: ExecutionPlan, measured: dict) -> dict:
    lines = plan.selected.lines
    ctx = sum(ln.bytes for ln in lines if ln.where == "device"
              and ln.name.startswith(("CUDA context", "allocator reserve")))
    est_alloc = plan.selected.device_bytes - ctx
    out = {"device_allocator": {"estimated": est_alloc, "measured": measured.get("device_peak_bytes"),
                                "what": "PyTorch allocator peak during training vs the estimate without the context and reserve lines"},
           "device_driver": {"estimated": plan.selected.device_bytes, "measured": measured.get("driver_process_peak_bytes"),
                             "what": "driver-reported process peak vs the full device estimate"},
           "host": {"estimated": plan.selected.host_bytes,
                    "measured": (measured.get("host_required_peak_bytes") or measured.get("host_anon_peak_bytes")
                                 or measured.get("host_peak_bytes")),
                    "what": ("peak of anonymous + shared RSS (pinned memory included, mapped checkpoint pages "
                             "excluded; load transients included) vs the host estimate")
                            if measured.get("host_required_peak_bytes") else
                            "peak RSS (an older receipt: see measured.* for which kind)"}}
    for v in out.values():
        if v["measured"] is not None:
            v["residual"] = v["measured"] - v["estimated"]
            v["ratio"] = round(v["measured"] / v["estimated"], 4) if v["estimated"] else None
    derived = sum(ln.bytes for ln in lines if ln.where == "device" and ln.basis == "derived")
    out["device_allocator"]["derived_part"] = derived
    out["device_allocator"]["heuristic_part"] = sum(ln.bytes for ln in lines if ln.where == "device"
                                                    and ln.basis == "heuristic")
    return out


def execute(plan: ExecutionPlan, *, out_dir: str | None = None, seed: int = 0, log=print,
            prov: dict | None = None) -> dict:
    """Run ``plan``. A refused plan is not executed: it raises ``PlanNotExecutable`` with the refusal's reasons.
    ``prov`` is :func:`measure.provenance` taken at process start (the CLI does); without it, provenance is taken here
    and is only as good as the claim that nothing changed on disk since the code was imported."""
    if plan.status != "feasible":
        raise PlanNotExecutable("refused plan: " + "; ".join(plan.refusal.get("reasons", ())))
    prov = prov or {**provenance(), "taken": "at execute(), after import"}
    t0 = time.time()
    result = _executor(plan.selected.backend)(plan, seed=seed, log=log)
    model_short = plan.model["model"].rstrip("/").split("/")[-1]
    s = plan.selected.setup
    run_id = (f"{model_short}-{s['expert_residency']}-{s['expert_kernel']}{'-attn4' if s['attn_4bit'] else ''}"
              f"-t{plan.workload.tokens_per_microbatch}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}")
    receipt = {
        "schema": RECEIPT_SCHEMA, "run_id": run_id, "status": result["status"],
        "model": plan.model, "workload": {**plan.workload.__dict__,
                                          "tokens_per_microbatch": plan.workload.tokens_per_microbatch},
        "setup": s, "hardware": plan.hardware, "plan": plan.to_dict(),
        "measured": result["measured"], "correctness": result["correctness"], "engaged": result["engaged"],
        "data": result["data"], "comparison": compare(plan, result["measured"]),
        "provenance": {**prov, "runtime_seconds": time.time() - t0, "seed": seed,
                       "changed_during_run": changed_since(prov)},
    }
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"{run_id}.json"), "w") as f:
            json.dump(receipt, f, indent=1, sort_keys=True, default=str)
    return receipt


def load_observations(path: str | None) -> list:
    """Receipts in ``path`` (a directory of ``*.json``), newest first by their recorded start time, for the planner to
    learn runtime overheads from."""
    if not path or not os.path.isdir(path):
        return []
    out = []
    for name in os.listdir(path):
        if name.endswith(".json"):
            try:
                with open(os.path.join(path, name)) as f:
                    r = json.load(f)
            except (OSError, ValueError):
                continue
            if r.get("schema") == RECEIPT_SCHEMA:
                out.append(r)
    return sorted(out, key=lambda r: (r.get("provenance", {}).get("started_at") or "", r.get("run_id") or ""),
                  reverse=True)


def summarize(receipt: dict) -> str:
    m, c, cmp = receipt["measured"], receipt["correctness"], receipt["comparison"]
    g = lambda n: "n/a" if n is None else f"{n / GiB:.2f} GiB"  # noqa: E731
    rows = [f"Run {receipt['run_id']}: {receipt['status']}",
            f"  step time  median {m['s_per_step_median']:.2f} s (steps {m['timed_steps']}), "
            f"{m['tokens_per_s']:.0f} tokens/s  [measured]",
            f"  loss       {c['losses'][0]:.4f} -> {c['losses'][-1]:.4f}; first third {c['loss_first_third_mean']:.4f}, "
            f"last third {c['loss_last_third_mean']:.4f}; finite={c['all_finite']}",
            f"  integrity  frozen expert bytes unchanged={c['frozen_expert_bytes_unchanged']}; "
            f"adapters moved={c['adapters_moved']}",
            "  memory                      estimated   measured   residual"]
    for k, v in cmp.items():
        res = v.get("residual")
        rows.append(f"    {k:24s} {g(v['estimated']):>10s} {g(v['measured']):>10s} "
                    f"{('n/a' if res is None else f'{res / GiB:+.2f} GiB'):>10s}")
    return "\n".join(rows)
