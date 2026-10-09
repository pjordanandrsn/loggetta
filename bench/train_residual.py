"""Where does a training run's allocator peak exceed the plan's estimate? Re-run a committed receipt's model, setup and
workload through the planner and the executor with the caching allocator's history recording (Python stacks), then
replay the trace: the allocations live at the training peak, grouped by the innermost frame that names a mechanism,
beside the plan's itemized device lines.

    python bench/train_residual.py evidence/2026-10-04-rtx-a2000/<receipt>.json --out runs/train-residual/<arm>
    python bench/train_residual.py <receipt>.json --fix expert_kernel=reference --out runs/train-residual/<arm>
    python bench/train_residual.py --summarize runs/train-residual/<arm> ...
    python bench/train_residual.py --regroup runs/train-residual/<arm> ...    # re-key a saved run's groups

The run is today's code on this machine, not the receipt's (receipts before experts4bit-qlora #1292 ran with an fp32
expert absmax; set E4B_ABSMAX_DQ=0 to match them): compare its measured peak with the receipt's first. The
replay says what is live at the peak and which frame allocated it, on this card. It does not say how large the same
term is on another card or shape.
"""
import argparse
import json
import os
import pickle
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
GiB, MiB = 1 << 30, 1 << 20
MECHANISMS = ("experts4bit_qlora", "/kernel/", "gnf4", "grouped_nf4", "peft/", "bitsandbytes/", "transformers/",
              "torch/optim", "loggetta/")


def key_of(frames):
    """The innermost frame that names a mechanism (skipping torch internals and this script)."""
    for f in frames or ():
        fn = f.get("filename", "")
        if "train_residual" in fn:
            continue
        if any(s in fn for s in MECHANISMS):
            return f"{_short(fn)}:{f.get('line')} {f.get('name')}"
    for f in frames or ():
        fn = f.get("filename", "")
        if "train_residual" not in fn:
            return f"{_short(fn)}:{f.get('line')} {f.get('name')}"
    return "?"


def _short(fn):
    """A source path from its package directory on (site-packages, a checkout, or a worktree alike)."""
    for marker in ("experts4bit_qlora/", "kernel/", "site-packages/"):
        if marker in fn:
            return fn[fn.index(marker) + (len(marker) if marker == "site-packages/" else 0):]
    return fn[fn.rindex("loggetta/"):] if "loggetta/" in fn else fn


def replay(trace, start=0):
    """Allocated bytes and the live allocations at the peak of trace[start:], given everything live at ``start``."""
    live, total, peak, at_peak, at = {}, 0, 0, {}, None
    for i, ev in enumerate(trace):
        a, addr, size = ev.get("action"), ev.get("addr"), ev.get("size", 0)
        if a == "alloc":
            live[addr] = (size, ev.get("frames"))
            total += size
        elif a == "free_completed" and addr in live:
            total -= live.pop(addr)[0]
        if i >= start and total > peak:
            peak, at_peak, at = total, dict(live), i
    return peak, at_peak, at


#: frame substring -> what the live allocation is, for the summary (first match wins; unmatched frames are "other")
CATEGORIES = (
    ("_quantize_stack", "frozen expert stacks (resident)"),
    ("_copy_home_to_device", "frozen expert stacks (one layer staged)"),
    ("loader.py:1418 _read", "dense weights (bf16)"),
    ("load_moe_4bit_streaming", "expert LoRA adapters"),
    ("lora.py:780", "attention LoRA adapters"), ("lora.py:781", "attention LoRA adapters"),
    ("torch/optim/adam.py", "optimizer state (adamw)"),
    ("_lora_delta_padded", "padded LoRA delta (gnf4 _lora_delta_padded: G x widest rows)"),
    ("nf4_qlora.py:349 backward", "expert LoRA gradients"),
    ("fused_experts_train_forward", "fused grouped kernel workspaces"), ("fused_grouped_lora", "fused grouped kernel workspaces"),
    ("gemm_4bit_grouped", "fused grouped kernel workspaces"), ("nf4_qlora.py:362 _scaled", "fused grouped kernel workspaces"),
    ("nf4_qlora.py:344 forward", "fused grouped kernel workspaces"),
    ("loss_utils.py", "logits and loss"),
    ("modeling_olmoe.py:402", "checkpointed layer inputs"),
)


def regroup(d):
    """Replay a run's saved snapshot again and rewrite its groups with today's ``key_of`` (the numbers are unchanged)."""
    r = json.load(open(os.path.join(d, "residual.json")))
    with open(os.path.join(d, "snapshot.pickle"), "rb") as f:
        trace = pickle.load(f)["device_traces"][0]
    peak, at_peak, at = replay(trace, start=r["train_mark"])
    if (peak, at) != (r["replayed_peak_bytes"], r["peak_event"]):
        sys.exit(f"{d}: the replay differs from the run's ({peak} at {at}, recorded {r['replayed_peak_bytes']} at "
                 f"{r['peak_event']})")
    groups = defaultdict(lambda: [0, 0, defaultdict(int)])
    for size, frames in at_peak.values():
        g = groups[key_of(frames)]
        g[0] += size
        g[1] += 1
        g[2][size] += 1
    r["groups"] = [{"frame": k, "bytes": b, "count": n, "sizes": {str(s): c for s, c in sizes.items()}}
                   for k, (b, n, sizes) in sorted(groups.items(), key=lambda kv: -kv[1][0])]
    json.dump(r, open(os.path.join(d, "residual.json"), "w"), indent=1, default=str)


def summarize(dirs):
    """A table of each run's live set at its training peak by category, beside the plan's activation item."""
    rows, cats = {}, []
    for d in dirs:
        r = json.load(open(os.path.join(d, "residual.json")))
        where = os.path.join(d, "receipts") if os.path.isdir(os.path.join(d, "receipts")) else d
        rec = json.load(open(next(os.path.join(where, f) for f in sorted(os.listdir(where))
                                  if f.endswith(".json") and f != "residual.json")))
        act = next(ln for ln in rec["plan"]["selected"]["lines"] if ln["name"] == "activations")
        sums = defaultdict(int)
        for g in r["groups"]:
            cat = next((c for sub, c in CATEGORIES if sub in g["frame"]), "other (incl. unattributed frames)")
            sums[cat] += g["bytes"]
            if cat not in cats:
                cats.append(cat)
        est = sum(r["estimate_device"].values())
        rows[os.path.basename(os.path.normpath(d))] = (r, sums, est, act)
    names = list(rows)
    print("| live at the training peak, MiB | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    for c in cats:
        print(f"| {c} | " + " | ".join(f"{rows[n][1].get(c, 0) / MiB:.1f}" for n in names) + " |")
    print("| **replayed peak** | " + " | ".join(f"**{rows[n][0]['replayed_peak_bytes'] / MiB:.1f}**" for n in names) + " |")
    print("| measured allocated peak | " + " | ".join(f"{rows[n][0]['measured']['device_peak_bytes'] / MiB:.1f}" for n in names) + " |")
    print("| plan's allocator estimate | " + " | ".join(f"{rows[n][2] / MiB:.1f}" for n in names) + " |")
    print("| residual (measured − estimate) | " + " | ".join(
        f"{(rows[n][0]['measured']['device_peak_bytes'] - rows[n][2]) / MiB:+.1f}" for n in names) + " |")
    print()
    for n in names:
        r, _s, _e, act = rows[n]
        print(f"- {n}: kernel {r['setup'].get('expert_kernel')}, residency {r['setup'].get('expert_residency')}, "
              f"E4B_ABSMAX_DQ={r.get('env', {}).get('E4B_ABSMAX_DQ')!r}; plan's activations {act['bytes'] / MiB:.1f} MiB: "
              f"{act['detail']}")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--summarize":
        return summarize(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "--regroup":
        for d in sys.argv[2:]:
            regroup(d)
        return None
    ap = argparse.ArgumentParser()
    ap.add_argument("receipt", help="a committed ExecutionReceipt to re-run (its model, setup and workload)")
    ap.add_argument("--fix", action="append", default=[], help="override one setup field, e.g. expert_kernel=reference")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-entries", type=int, default=5_000_000)
    a = ap.parse_args()
    import torch

    from loggetta import Constraints, Workload, describe_model, plan
    from loggetta.execution import execute
    from loggetta.hardware import probe
    from loggetta.measure import provenance

    prov = provenance()
    rec = json.load(open(a.receipt))
    setup = dict(rec["setup"])
    for kv in a.fix:
        k, v = kv.split("=", 1)
        setup[k] = json.loads(v) if v in ("true", "false") or v.lstrip("-").isdigit() else v
    wl = {k: v for k, v in (rec.get("plan") or {}).get("workload", rec["workload"]).items()
          if k in Workload.__dataclass_fields__}
    hw = probe()
    p = plan(describe_model(rec["model"]["model"], revision=rec["model"].get("revision")), hw, Workload(**wl),
             Constraints(fixed=setup))
    if p.status != "feasible":
        sys.exit(f"the plan refused here: {p.refusal['reasons']}")
    lines = [ln for ln in p.selected.lines if ln.where == "device"]
    est = {ln.name: ln.bytes for ln in lines if not ln.name.startswith(("CUDA context", "allocator reserve"))}

    marks = []
    reset = torch.cuda.reset_peak_memory_stats

    def marking_reset(*args, **kwargs):           # the training loop resets the peak right before its first step
        marks.append(len(torch.cuda.memory._snapshot()["device_traces"][0]))
        return reset(*args, **kwargs)

    torch.cuda.set_device(0)
    torch.zeros(1, device="cuda")
    torch.cuda.memory._record_memory_history(max_entries=a.max_entries, stacks="python")
    torch.cuda.reset_peak_memory_stats = marking_reset
    t0 = time.time()
    try:
        receipt = execute(p, out_dir=os.path.join(a.out, "receipts"), hardware=hw, prov=prov)
    finally:
        torch.cuda.reset_peak_memory_stats = reset
    run_s = time.time() - t0
    snap = torch.cuda.memory._snapshot()
    torch.cuda.memory._record_memory_history(enabled=None)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "snapshot.pickle"), "wb") as f:
        pickle.dump(snap, f)
    trace = snap["device_traces"][0]
    if len(trace) >= a.max_entries:
        sys.exit(f"the trace filled its {a.max_entries} entries: early allocations were dropped; raise --max-entries")
    mark = marks[-1] if marks else 0
    peak, at_peak, at = replay(trace, start=mark)
    groups = defaultdict(lambda: [0, 0, defaultdict(int)])
    for size, frames in at_peak.values():
        g = groups[key_of(frames)]
        g[0] += size
        g[1] += 1
        g[2][size] += 1
    top = sorted(groups.items(), key=lambda kv: -kv[1][0])
    m = receipt["measured"]
    est_alloc = sum(est.values())
    print(f"run {receipt['run_id']} status {receipt['status']} ({run_s:.0f}s); E4B_ABSMAX_DQ="
          f"{os.environ.get('E4B_ABSMAX_DQ')!r} PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF')!r}")
    print(f"estimate {est_alloc / GiB:.3f} GiB | measured allocated peak {m['device_peak_bytes'] / GiB:.3f} | replayed "
          f"training peak {peak / GiB:.3f} | residual {(m['device_peak_bytes'] - est_alloc) / MiB:+.1f} MiB | the "
          f"receipt re-run: {rec['measured']['device_peak_bytes'] / GiB:.3f}")
    print("the plan's device items (reserve and context excluded):")
    for name, b in sorted(est.items(), key=lambda kv: -kv[1]):
        print(f"  {b / MiB:9.1f} MiB  {name}")
    print(f"live at the training peak (trace event {at} of {len(trace)}, training from {mark}), by allocating frame:")
    for k, (b, n, sizes) in top[:40]:
        common = ", ".join(f"{s / MiB:.2f} MiB x{c}" for s, c in sorted(sizes.items(), key=lambda kv: -kv[0] * kv[1])[:3])
        print(f"  {b / MiB:9.1f} MiB  x{n:<5d} {k}  [{common}]")
    json.dump({"schema": "train-residual/1", "receipt": a.receipt, "setup": setup, "workload": wl,
               "estimate_device": est, "measured": {k: m.get(k) for k in ("device_peak_bytes",
                                                                          "device_reserved_peak_bytes",
                                                                          "driver_process_peak_bytes")},
               "replayed_peak_bytes": peak, "peak_event": at, "train_mark": mark, "trace_events": len(trace),
               "run_id": receipt["run_id"], "provenance": prov,
               "env": {k: os.environ.get(k) for k in ("E4B_ABSMAX_DQ", "PYTORCH_CUDA_ALLOC_CONF")},
               "groups": [{"frame": k, "bytes": b, "count": n, "sizes": {str(s): c for s, c in sizes.items()}}
                          for k, (b, n, sizes) in top]},
              open(os.path.join(a.out, "residual.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
