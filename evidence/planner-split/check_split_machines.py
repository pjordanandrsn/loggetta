"""Check, from the private receipt store, that every cohort's held-out boxes ran on distinct machines.

A band that requires two boxes per cohort is asking for two hosts. The drivers that rent the boxes refuse a repeat
machine (HO3's Part A refuses HO2's, Part B a second box on the first box's), and this check confirms it from the
store's own records rather than from those rules.

For every held-out row of a split, the row's box (its ``box``, or its id's first field) is looked up in the store
under ``<store>/receipts/<repo>/<date>/<box>/``:
- the machine is ``instance.json``'s ``machine_id``, or, for a box older than that file, ``receipt.json``'s
  ``environment.vast_machine_id``;
- where both are present they must agree;
- a box that is missing, found under more than one date, or with no recorded machine refuses the check.

The output names boxes and counts only, never a machine id, so it can be committed beside the split:

    python evidence/planner-split/check_split_machines.py SPLIT-v3.json STORE --out SPLIT-v3-machines.json

It exits 1 when a cohort's boxes share a machine, and 2 when a box's machine cannot be read.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys


class Unreadable(Exception):
    pass


def box_of(row):
    return row.get("box") or row["id"].split("/")[0]


def machine(store, repo, box):
    found = sorted(glob.glob(os.path.join(store, "receipts", repo, "*", box)))
    if len(found) != 1:
        raise Unreadable(f"{box}: {len(found)} run directories in the store, not exactly one")
    run = found[0]
    ids = {}
    inst = os.path.join(run, "instance.json")
    if os.path.isfile(inst):
        ids["instance.json"] = (json.load(open(inst)) or {}).get("machine_id")
    rec = os.path.join(run, "receipt.json")
    if os.path.isfile(rec):
        ids["receipt.json"] = ((json.load(open(rec)) or {}).get("environment") or {}).get("vast_machine_id")
    got = {str(v) for v in ids.values() if v not in (None, "")}
    if not got:
        raise Unreadable(f"{box}: no recorded machine")
    if len(got) > 1:
        raise Unreadable(f"{box}: instance.json and receipt.json name different machines")
    return got.pop()


def check(split, store):
    boxes = {}
    for row in split["held_out"]:
        boxes.setdefault(row["cohort"], {}).setdefault(box_of(row), row["repo"])
    out = {}
    for cohort, bx in sorted(boxes.items()):
        machines = {machine(store, repo, b) for b, repo in bx.items()}
        out[cohort] = {"boxes": sorted(bx), "n_boxes": len(bx), "n_machines": len(machines),
                       "distinct_machines": len(machines) == len(bx)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("split")
    ap.add_argument("store")
    ap.add_argument("--out")
    a = ap.parse_args()
    split = json.load(open(a.split))
    try:
        cohorts = check(split, a.store)
    except Unreadable as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    doc = {"schema": "loggetta-split-machines/1", "split": os.path.basename(a.split), "version": split["version"],
           "source": "the private receipt store: each box's instance.json machine_id, else its receipt's vast_machine_id",
           "cohorts": cohorts}
    text = json.dumps(doc, indent=1) + "\n"
    if a.out:
        open(a.out, "w").write(text)
    else:
        sys.stdout.write(text)
    shared = [c for c, v in cohorts.items() if not v["distinct_machines"]]
    if shared:
        print(f"refused: boxes share a machine in {shared}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
