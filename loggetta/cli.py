"""Command line: ``inspect``, ``plan``, ``train``. The program name is taken from argv, never spelled here."""
from __future__ import annotations

import argparse
import json
import sys

GiB = 1 << 30


def _gib(s):
    return None if s is None else int(float(s) * GiB)


def _parse_fixed(items):
    out = {}
    for it in items or ():
        k, _, v = it.partition("=")
        if not _:
            raise SystemExit(f"--fix expects key=value, got {it!r}")
        low = v.lower()
        out[k] = True if low == "true" else False if low == "false" else int(v) if v.lstrip("-").isdigit() else v
    return out


def _common(p):
    p.add_argument("model", help="hub id or local snapshot directory")
    p.add_argument("--revision")
    p.add_argument("--workload", default="train", choices=("train", "serve"))
    p.add_argument("--seq", type=int, default=512)
    p.add_argument("--micro-batch", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--context", type=int, help="serve: tokens per sequence, prompt + output (default 4096)")
    p.add_argument("--concurrency", type=int, help="serve: sequences decoded together (default 1)")
    p.add_argument("--optimizer", default="adamw", choices=("adamw", "adamw_8bit"))
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--vram", help="device budget in GiB (default: free now)")
    p.add_argument("--ram", help="host budget in GiB (default: available now)")
    p.add_argument("--headroom", help="device headroom in GiB (default: max(0.5, 5%% of budget))")
    p.add_argument("--experts", choices=("any", "device", "host"), default="any",
                   help="where frozen experts may live")
    p.add_argument("--fix", action="append", metavar="FIELD=VALUE",
                   help="fix a backend setup field (expert mode), e.g. --fix expert_kernel=reference")
    p.add_argument("--objective", default="speed", choices=("speed", "min_vram", "min_ram"))
    p.add_argument("--target-s-per-step", type=float,
                   help="refuse setups whose host-to-device traffic alone provably exceeds this step time")
    p.add_argument("--observations", help="directory of earlier receipts to learn runtime overheads from")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--hardware", help="plan for a saved hardware profile (inspect --json) instead of probing this machine")


def _plan(a):
    from . import Constraints, Workload, describe_model, plan, probe
    from .runtime import load_observations

    topo = describe_model(a.model, revision=a.revision, trust_remote_code=a.trust_remote_code)
    if a.hardware:
        from .hardware import HardwareProfile

        with open(a.hardware) as f:
            hw = HardwareProfile.from_dict(json.load(f), origin=a.hardware)
    else:
        hw = probe()
    w = Workload(kind=a.workload, seq_len=a.seq, micro_batch=a.micro_batch, grad_accum=a.grad_accum, steps=a.steps,
                 optimizer=a.optimizer, context_len=a.context or (4096 if a.workload == "serve" else None),
                 concurrency=a.concurrency or (1 if a.workload == "serve" else None))
    c = Constraints(device=a.device, vram_budget=_gib(a.vram), ram_budget=_gib(a.ram), headroom=_gib(a.headroom),
                    expert_residency=None if a.experts == "any" else (a.experts,), fixed=_parse_fixed(a.fix),
                    objective=a.objective, target_s_per_step=a.target_s_per_step)
    return plan(topo, hw, w, c, observations=load_observations(a.observations))


def main(argv=None) -> int:
    from .measure import provenance

    prov = {**provenance(), "taken": "at process start, before the measured code was imported"}
    ap = argparse.ArgumentParser(description="Plan, explain and run MoE workloads on this machine.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pi = sub.add_parser("inspect", help="the hardware inventory, and a model's topology if one is given")
    pi.add_argument("model", nargs="?")
    pi.add_argument("--json", action="store_true")
    pp = sub.add_parser("plan", help="decide how a workload would run here; loads no weights")
    _common(pp)
    pp.add_argument("--json", action="store_true")
    pp.add_argument("--out", help="write the plan as JSON here")
    pp.add_argument("-v", "--verbose", action="store_true")
    pt = sub.add_parser("train", help="plan, then execute a feasible plan and write a receipt")
    _common(pt)
    pt.add_argument("--out", default="receipts", help="receipt directory")
    pt.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    if a.cmd == "inspect":
        from .hardware import describe, probe

        hw = probe()
        if a.json:
            print(json.dumps(hw.to_dict(), indent=1, default=str))
        else:
            print(describe(hw))
        if a.model:
            from .model import describe_model

            t = describe_model(a.model)
            print(json.dumps(t.to_dict(), indent=1, default=str) if a.json else "\nModel\n  " + t.summary())
        return 0

    p = _plan(a)
    if a.cmd == "plan":
        if a.out:
            with open(a.out, "w") as f:
                f.write(p.to_json())
        print(p.to_json() if a.json else p.render(verbose=a.verbose))
        return 0 if p.status == "feasible" else 2

    print(p.render())
    if p.status != "feasible":
        return 2
    from .runtime import execute, summarize

    print("\nExecuting the selected plan...")
    receipt = execute(p, out_dir=a.out, seed=a.seed, prov=prov)
    print("\n" + summarize(receipt))
    print(f"\nreceipt: {a.out}/{receipt['run_id']}.json")
    return 0 if receipt["status"] == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())
