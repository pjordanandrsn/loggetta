"""MoE plans against what the driver measured: every receipt that carries a Loggetta plan (or a lane's registered plan
total) and a measured driver peak, read the way DQ7 read the dense plans.

    python bench/plan_vs_driver.py --e4b E4B_CHECKOUT --json evidence/<dir>/plan-vs-driver.json

Per receipt, it reports two comparisons:
- **Plan total vs driver peak.** The plan's device total (allocator estimate + reserve + CUDA context) against the
  process's driver-reported peak. A plan total under the driver peak is flagged.
- **Allocator estimate vs allocated peak.** The estimate without the reserve and context lines against PyTorch's
  allocated peak. An estimate under the allocated peak is flagged.

Sources, read as committed (no number is typed in here):

1. **Loggetta ExecutionReceipts under ``evidence/``** with the plan attached (``plan.selected``). The allocator estimate
   is the plan's device total less its context and reserve lines, as ``execution.compare`` reads it.
2. **experts4bit-qlora lanes SV5, SV6 and SV7.** Each registered the planner's plan total for one serving setup on an
   RTX 4090, as ``PLAN_BYTES`` and ``EST_BYTES`` in its merged reducer (``bench/svN/svN_reduce.py``), read here by
   importing the reducer. Each arm receipt with the registered setup is paired with them. The lane read its plan
   against the ``_long`` arm; the ``_short`` arm ran the same setup.
3. **Loggetta receipts with a driver peak but no plan** (FP1, SV1-SV4, plan-less A2000 runs): the allocator
   comparison only, against the receipt's own estimate. A receipt that source 2 covers is not repeated.

Dense receipts are left out: DQ7 read those (experts4bit-qlora#1359).
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os

GiB = 1 << 30


def _placement(kind, setup):
    if kind == "serve":
        p = setup.get("placement")
        return "resident (all-VRAM)" if p == "all-vram" else f"offload ({p} tiers)" if p else "?"
    r = setup.get("expert_residency")
    return {"device": "resident (experts on device)", "host": "offload (experts on host)"}.get(r, f"? ({r})")


def _row(source, run, card, kind, setup, status, plan_total, driver, est, alloc, note="", parts=None):
    return {"source": source, "run": run, "card": card, "kind": kind, "placement": _placement(kind, setup or {}),
            "status": status, "plan_total_bytes": plan_total, "driver_peak_bytes": driver,
            "allocator_estimate_bytes": est, "allocated_peak_bytes": alloc, "note": note, "parts": parts,
            "plan_under_driver": bool(plan_total and driver and driver > plan_total),
            "estimate_under_allocated": bool(est and alloc and alloc > est)}


def from_evidence(root):
    """Sources 1 and 3: Loggetta's committed ExecutionReceipts."""
    with_plan, without = [], []
    for f in sorted(glob.glob(os.path.join(root, "**", "*.json"), recursive=True)):
        try:
            r = json.load(open(f))
        except (OSError, ValueError):
            continue
        if not isinstance(r, dict) or r.get("schema") != "execution-receipt/1":
            continue
        m = r.get("measured") or {}
        driver = m.get("driver_process_peak_bytes")
        if not driver:
            continue
        kind = (r.get("workload") or {}).get("kind") or "train"
        card = ((r.get("hardware") or {}).get("gpu") or {}).get("name")
        run = os.path.relpath(f, root)[:-5]
        sel = ((r.get("plan") or {}).get("selected")) or None
        if sel is not None:
            if sel.get("backend") == "dense":
                continue
            lines = sel["lines"]

            def line(prefix):
                return sum(ln["bytes"] for ln in lines if ln["where"] == "device" and ln["name"].startswith(prefix))
            reserve, context = line("allocator reserve"), line("CUDA context")
            alloc, reserved = m.get("device_peak_bytes"), m.get("device_reserved_peak_bytes")
            est = sel["device_bytes"] - reserve - context
            # driver - plan = (allocated - estimate) + (reserved - allocated - reserve) + (driver - reserved - context)
            parts = {"allocator": alloc - est, "reserve": reserved - alloc - reserve,
                     "context": driver - reserved - context,
                     "reserve_line": next((ln["basis"] + ": " + ln["detail"] for ln in lines
                                           if ln["name"].startswith("allocator reserve")), None),
                     "context_line": next((ln["basis"] + ": " + ln["detail"] for ln in lines
                                           if ln["name"].startswith("CUDA context")), None)} \
                if alloc and reserved else None
            with_plan.append(_row("loggetta plan", run, card, kind, r.get("setup"), r.get("status"),
                                  sel["device_bytes"], driver, est, alloc, parts=parts))
        else:
            e = r.get("estimate") or {}
            est = e.get("device_bytes") or e.get("device_total") or e.get("allocator_estimate_bytes")
            without.append((r.get("run_id"), _row("receipt's own estimate (no plan)", run, card, kind, r.get("setup"),
                                                  r.get("status"), None, driver, est, m.get("device_peak_bytes"))))
    return with_plan, without


def from_registered(e4b):
    """Source 2: SV5-SV7's registered plan totals beside their committed arm receipts."""
    rows, covered = [], set()
    for lane in ("sv5", "sv6", "sv7"):
        path = os.path.join(e4b, "bench", lane, f"{lane}_reduce.py")
        spec = importlib.util.spec_from_file_location(f"{lane}_reduce", path)
        red = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(red)
        for f in sorted(glob.glob(os.path.join(e4b, "bench", lane, "receipts", "*", "receipts", "*.json"))):
            r = json.load(open(f))
            arm = os.path.basename(f)[:-5]
            run_id = f"{f.split(os.sep)[-3]}/{arm}"
            if r.get("schema") != red.SCHEMA or arm not in red.ARMS:
                continue
            same = all(json.dumps(r.get("setup", {}).get(k)) == json.dumps(v) for k, v in red.SETUP.items())
            m = r.get("measured") or {}
            note = ("the lane's registered reading" if arm.endswith("_long") else "same setup, shorter prompts")
            if not same:
                note += "; setup differs from the registered one"
            if not m.get("driver_process_peak_bytes"):
                note += f"; no driver reading ({m.get('driver_samples', 0)} samples)"
            rows.append(_row(f"{lane.upper()} registered plan", run_id, r.get("gpu"), "serve", r.get("setup"),
                             r.get("status"), red.PLAN_BYTES, m.get("driver_process_peak_bytes"), red.EST_BYTES,
                             m.get("device_peak_bytes"), note))
            covered.add(run_id)
    return rows, covered


def _g(b):
    return "—" if not b else f"{b / GiB:.3f}"


def _ratio(a, b):
    return "—" if not (a and b) else f"{a / b:.3f}"


def table(rows):
    out = ["| source | run | card | kind | placement | status | plan total | driver peak | driver − plan | driver / plan "
           "| allocator est. | allocated peak | allocated / est. | flags |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        diff = (f"{(r['driver_peak_bytes'] - r['plan_total_bytes']) / GiB:+.3f}"
                if r["plan_total_bytes"] and r["driver_peak_bytes"] else "—")
        flags = [f for f, on in (("**PLAN UNDER DRIVER**", r["plan_under_driver"]),
                                 ("**EST. UNDER ALLOCATED**", r["estimate_under_allocated"])) if on]
        if r["note"]:
            flags.append(r["note"])
        out.append(f"| {r['source']} | {r['run']} | {(r['card'] or '?').replace('NVIDIA ', '')} | {r['kind']} | "
                   f"{r['placement']} | {r['status']} | {_g(r['plan_total_bytes'])} | {_g(r['driver_peak_bytes'])} | "
                   f"{diff} | {_ratio(r['driver_peak_bytes'], r['plan_total_bytes'])} | "
                   f"{_g(r['allocator_estimate_bytes'])} | {_g(r['allocated_peak_bytes'])} | "
                   f"{_ratio(r['allocated_peak_bytes'], r['allocator_estimate_bytes'])} | {'; '.join(flags)} |")
    return "\n".join(out)


def summary(rows):
    out = []
    groups = {}
    for r in rows:
        if r["plan_total_bytes"] and r["driver_peak_bytes"]:
            groups.setdefault((r["card"], r["kind"], r["placement"]), []).append(r)
    out.append("| card | kind | placement | receipts with plan + driver | plan under driver | driver / plan, min–max |")
    out.append("|---|---|---|---|---|---|")
    for (card, kind, placement), rs in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        ratios = [r["driver_peak_bytes"] / r["plan_total_bytes"] for r in rs]
        out.append(f"| {(card or '?').replace('NVIDIA ', '')} | {kind} | {placement} | {len(rs)} | "
                   f"{sum(r['plan_under_driver'] for r in rs)} | {min(ratios):.3f}–{max(ratios):.3f} |")
    return "\n".join(out)


def gaps(rows):
    """Where each plan under its driver peak went short: driver - plan = allocator + reserve + context terms."""
    out = ["| run | driver − plan | allocated − estimate | (reserved − allocated) − reserve line | "
           "(driver − reserved) − context line | the plan's reserve line | the plan's context line |",
           "|---|---|---|---|---|---|---|"]
    for r in rows:
        p = r.get("parts")
        if not (r["plan_under_driver"] and p):
            continue
        out.append(f"| {r['run']} | {(r['driver_peak_bytes'] - r['plan_total_bytes']) / GiB:+.3f} | "
                   f"{p['allocator'] / GiB:+.3f} | {p['reserve'] / GiB:+.3f} | {p['context'] / GiB:+.3f} | "
                   f"{p['reserve_line']} | {p['context_line']} |")
    return "\n".join(out)


def replan_today(root, rows):
    """Each training receipt's own setup, workload and stated hardware, planned again with today's code: once with no
    receipts, once with its sibling receipts on file (its own directory less itself, so it never teaches its own plan).
    Configs come from the local Hugging Face cache (no weights). Returns (table, rows)."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from loggetta import Constraints, Workload, describe_model, plan
    from loggetta.execution import load_observations
    from loggetta.hardware import GPU, Fact, HardwareProfile, Host

    out, data, topologies = [], [], {}
    for r in rows:
        if r["source"] != "loggetta plan" or r["kind"] != "train":
            continue
        path = os.path.join(root, r["run"] + ".json")
        rec = json.load(open(path))
        name = rec["model"]["model"]
        if name not in topologies:
            topologies[name] = describe_model(name, revision=rec["model"].get("revision"))
        h, g = rec["hardware"], rec["hardware"]["gpu"]
        f = lambda v: Fact(v, "stated")  # noqa: E731
        cur_gen, cur_width, max_gen, max_width = g["pcie"]
        gpu = GPU(index=0, vendor="nvidia", name=g["name"], uuid=None, compute_capability=f(tuple(g["compute_capability"])),
                  memory_total=f(g["memory_total"]), memory_free=f(g["memory_free"]), driver=f(g["driver"]),
                  pcie_gen_max=f(max_gen), pcie_width_max=f(max_width), pcie_gen_current=f(cur_gen),
                  pcie_width_current=f(cur_width))
        host = Host(cpu_model=f(h["cpu"]), cpus=f(h["cpus"]), memory_total=f(h["ram_limit"]),
                    memory_available=f(h["ram_available"]), memory_limit=f(h["ram_limit"]))
        hw = HardwareProfile(gpus=(gpu,), host=host, platform=h["platform"])
        wl = {k: v for k, v in rec["plan"]["workload"].items() if k in Workload.__dataclass_fields__}
        siblings = [o for o in load_observations(os.path.dirname(path)) if o.get("run_id") != rec["run_id"]]
        found = {}
        for label, obs in (("none", ()), ("siblings", siblings)):
            p = plan(topologies[name], hw, Workload(**wl), Constraints(fixed=dict(rec["setup"])), observations=obs)
            c = p.selected or (p.alternatives[0] if p.alternatives else None)
            if c is None:
                found[label] = None
                continue
            line = lambda prefix: next((ln for ln in c.lines if ln.name.startswith(prefix)), None)  # noqa: E731
            res, ctx = line("allocator reserve"), line("CUDA context")
            found[label] = {"status": p.status, "plan_total_bytes": c.device_bytes,
                            "allocator_estimate_bytes": c.device_bytes - (res.bytes if res else 0) - (ctx.bytes if ctx else 0),
                            "reserve": f"{res.basis} {res.bytes / GiB:.3f}" if res else "none",
                            "context": f"{ctx.basis} {ctx.bytes / GiB:.3f}" if ctx else "none"}
        data.append({"run": r["run"], "driver_peak_bytes": r["driver_peak_bytes"],
                     "allocated_peak_bytes": r["allocated_peak_bytes"], "today": found})
    out.append("| run | allocated peak | today's allocator est. | driver peak | today, no receipts: plan total "
               "(driver / plan) | today, siblings on file: plan total (driver / plan; reserve; context) |")
    out.append("|---|---|---|---|---|---|")
    for d in data:
        n, sib = d["today"]["none"], d["today"]["siblings"]
        cell = lambda x: "—" if x is None else (f"{_g(x['plan_total_bytes'])} ({_ratio(d['driver_peak_bytes'], x['plan_total_bytes'])})"  # noqa: E731
                                               + ("" if x["status"] == "feasible" else f", {x['status']}"))
        flag = lambda x: " **UNDER**" if x and d["driver_peak_bytes"] > x["plan_total_bytes"] else ""  # noqa: E731
        out.append(f"| {d['run']} | {_g(d['allocated_peak_bytes'])} | {_g(n['allocator_estimate_bytes']) if n else '—'} | "
                   f"{_g(d['driver_peak_bytes'])} | {cell(n)}{flag(n)} | {cell(sib)}{flag(sib)}"
                   + (f"; reserve {sib['reserve']}; context {sib['context']}" if sib else "") + " |")
    return "\n".join(out), data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "evidence"))
    ap.add_argument("--e4b", help="an experts4bit-qlora checkout (for SV5-SV7's registered plan totals)")
    ap.add_argument("--json", help="write the rows here as JSON")
    ap.add_argument("--replan", action="store_true",
                    help="also plan each training receipt's setup again with today's code (needs transformers, "
                         "experts4bit-qlora and the configs in the Hugging Face cache)")
    a = ap.parse_args()
    with_plan, without = from_evidence(a.evidence)
    registered, covered = from_registered(a.e4b) if a.e4b else ([], set())
    plain = [row for run_id, row in without if run_id not in covered]
    rows = with_plan + registered + plain
    print("## Summary (receipts with a plan total and a driver peak)\n\n" + summary(rows) + "\n")
    print("## Where the plans under their driver peak went short (GiB)\n\n" + gaps(rows) + "\n")
    print("## Every receipt\n\n" + table(rows))
    replanned = None
    if a.replan:
        text, replanned = replan_today(a.evidence, rows)
        print("\n## Today's planner on the same training setups\n\n" + text)
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"rows": rows, "replanned_today": replanned}, f, indent=1, sort_keys=True)


if __name__ == "__main__":
    main()
