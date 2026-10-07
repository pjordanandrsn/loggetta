"""Plan dense adapter training for real model configs across GPU sizes: the planner's choice per cell, nothing run.

Each model is described from its config (``--configs DIR`` of ``<name>.json`` files, or hub ids, which fetch only the
config), then planned on a stated GPU of each size with an ample host, at one sequence length. The table says what the
plan selects (base, placement), its device total, and for a refusal the first reason. Deterministic for the same configs,
code and receipts; no weights are read.

    python bench/dense_plan_sweep.py --configs notes/cfg --seq 4096 --out evidence/<dir>/dense-plan-sweep.md
"""
from __future__ import annotations

import argparse
import glob
import json
import os

GiB = 1 << 30
#: usable GiB per class: the RTX 4090's 23.52 is measured (an out-of-memory receipt); the others are nominal less ~2%
GPUS = {"16 GB": 15.5, "24 GB (4090)": 23.52, "32 GB (5090)": 31.3, "48 GB": 47.0}


def _hw(gib, ram_gib=120):
    from loggetta.hardware import GPU, Fact, HardwareProfile, Host

    r = lambda v: Fact(v, "stated")  # noqa: E731
    g = GPU(index=0, vendor="nvidia", name=f"stated {gib} GiB", uuid=None, compute_capability=r((8, 9)),
            memory_total=r(int(gib * GiB)), memory_free=r(int(gib * GiB)), driver=r("stated"), pcie_gen_max=r(4),
            pcie_width_max=r(16), pcie_gen_current=r(4), pcie_width_current=r(16))
    host = Host(cpu_model=r("stated"), cpus=r(16), memory_total=r(128 * GiB), memory_available=r(int(ram_gib * GiB)),
                memory_limit=r(128 * GiB))
    return HardwareProfile(gpus=(g,), host=host, platform="stated")


def _configs(spec):
    import transformers as tr

    if os.path.isdir(spec):
        for f in sorted(glob.glob(os.path.join(spec, "*.json"))):
            d = json.load(open(f))
            c = tr.AutoConfig.for_model(**d)
            c._name_or_path = os.path.basename(f)[:-5]
            yield c._name_or_path, c
    else:
        for mid in spec.split(","):
            yield mid, mid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", required=True, help="a directory of config.json files, or comma-separated hub ids")
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--out")
    a = ap.parse_args()

    from loggetta import Workload, plan
    from loggetta.backends import dense

    rows = [f"| model | {' | '.join(GPUS)} |", "|---|" + "---|" * len(GPUS)]
    for name, cfg in _configs(a.configs):
        t = dense.describe(cfg)
        cells = []
        for label, gib in GPUS.items():
            p = plan(t, _hw(gib), Workload(seq_len=a.seq), backends=(dense,))
            if p.status == "feasible":
                s = p.selected.setup
                cells.append(f"{s['base']} {'streamed' if s['placement'] == 'stream' else 'resident'} "
                             f"{p.selected.device_bytes / GiB:.1f} GiB")
            else:
                why = p.refusal["reasons"][0].split(": ", 1)[-1]      # the closest candidate's own rejection
                cells.append(f"refused ({p.refusal.get('closest', '').split(',')[0].replace('dense ', '')}"
                             f"{', streamed' if 'streamed' in p.refusal.get('closest', '') else ''}): {why}")
        rows.append(f"| {name} | " + " | ".join(cells) + " |")
    text = "\n".join(rows)
    print(text)
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "w") as f:
            f.write(f"Dense plans at seq {a.seq}, micro-batch 1 (bench/dense_plan_sweep.py; device totals include the "
                    "inferred 20% reserve and 0.5 GiB context; no receipts).\n\n" + text + "\n")


if __name__ == "__main__":
    main()
