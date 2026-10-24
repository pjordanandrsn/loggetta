"""The planner's held-out bands (``loggetta/planner_bands.json``), built from the frozen split and the receipts it cites.

    python bench/build_planner_bands.py evidence/planner-split/SPLIT-v3.json --e4b PATH/TO/experts4bit-qlora \
        --out loggetta/planner_bands.json

A cohort (one model x one card x one placement) gets a band only when the split gives it at least 2 held-out rows from at
least 2 boxes that ran on distinct machines. A cohort whose rows come from one box gets a named no-band; every other cohort
is unmeasured. For each eligible row the receipt is read from the cited repository AT the split's ``pinned_at`` commit
(``git show``, never the working tree), and refused unless its sha256 equals the row's ``sha256``. The band is the range
of measured / estimate that those rows OBSERVED, per term:
- device driver: the sampled driver peak over the plan's device total;
- device allocator: the allocated peak over the allocator estimate;
- host, for a row with the experts on the host: the host peak over the host estimate.

Nothing is fitted or extrapolated, and the band never changes an estimate. The output carries each row's bytes, so every
ratio recomputes from the file itself, and each row's id, box and sha256, so a printed band traces to its receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys

SCHEMA = "loggetta-planner-bands/1"
PLACEMENT = {"resident (device)": "device", "host offload (experts on host)": "host"}


class Refused(Exception):
    pass


def _at(repo_dir, commit, path):
    r = subprocess.run(["git", "-C", repo_dir, "show", f"{commit}:{path}"], capture_output=True)
    if r.returncode != 0:
        raise Refused(f"{path} is not readable at {commit[:12]}: {r.stderr.decode(errors='replace').strip()[:200]}")
    return r.stdout


def _term(est, meas):
    return {"estimated": est, "measured": meas, "ratio": meas / est}


def row_terms(receipt, placement):
    c = receipt.get("comparison") or {}
    out = {}
    for term, key in (("device_driver", "device_driver"), ("device_allocator", "device_allocator")) + (
            (("host", "host"),) if placement == "host" else ()):
        v = c.get(key) or {}
        est, meas = v.get("estimated"), v.get("measured")
        if not (isinstance(est, int) and est > 0 and isinstance(meas, int) and meas > 0):
            raise Refused(f"the receipt has no {term} estimate and measurement ({est}, {meas})")
        out[term] = _term(est, meas)
    return out


def summarize(rows):
    terms = {}
    for t in rows[0]["terms"]:
        ratios = [r["terms"][t]["ratio"] for r in rows]
        terms[t] = {"min": min(ratios), "max": max(ratios), "rows_over": sum(x > 1 for x in ratios)}
    return terms


def build(split, readers):
    """``readers`` maps a repository name to a function (commit, path) -> bytes."""
    pinned = split["pinned_at"]
    rows_by = {}
    for r in split["held_out"]:
        rows_by.setdefault(r["cohort"], []).append(r)
    cohorts = {}
    for name, info in sorted(split["cohorts"].items()):
        rows, boxes = rows_by.get(name, []), info.get("boxes")
        keys = {(r["model"], r["card"], PLACEMENT[r["placement"]]) for r in rows}
        if len(keys) != 1:
            raise Refused(f"{name}: its rows do not share one model, card and placement ({sorted(keys)})")
        (model, card, placement), = keys
        base = {"held_out_rows": info["held_out_rows"], "boxes": boxes, "distinct_machines": info.get("distinct_machines"),
                "model": model, "card": card, "placement": placement}
        if info["held_out_rows"] < 2:
            cohorts[name] = {**base, "band": None, "why": "unmeasured"}
            continue
        if not boxes or boxes < 2 or info.get("distinct_machines") is not True:
            cohorts[name] = {**base, "band": None, "why": "one box"}
            continue
        out_rows = []
        for r in rows:
            raw = readers[r["repo"]](pinned[r["repo"]], r["file"])
            if hashlib.sha256(raw).hexdigest() != r["sha256"]:
                raise Refused(f"{r['id']}: {r['file']} at {pinned[r['repo']][:12]} is not the split's sha256")
            rec = json.loads(raw)
            got = ((rec.get("model") or {}).get("model"), (rec.get("setup") or {}).get("expert_residency"))
            if got != (model, placement):
                raise Refused(f"{r['id']}: the receipt records {got}, the split {(model, placement)}")
            try:
                terms = row_terms(rec, placement)
            except Refused as e:
                raise Refused(f"{r['id']}: {e}") from None
            out_rows.append({"id": r["id"], "box": r.get("box") or r["id"].split("/")[0], "seq": r["seq"],
                             "file": r["file"], "sha256": r["sha256"], "terms": terms})
        cohorts[name] = {**base, "band": summarize(out_rows), "rows": out_rows}
    return {"schema": SCHEMA, "split": f"SPLIT-v{split['version']}.json", "split_version": split["version"],
            "pinned_at": pinned, "cohorts": cohorts}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("split")
    ap.add_argument("--e4b", required=True, help="an experts4bit-qlora checkout holding the split's pinned_at commit")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    split = json.load(open(a.split))
    try:
        doc = build(split, {"experts4bit-qlora": lambda c, p: _at(a.e4b, c, p)})
    except Refused as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    open(a.out, "w").write(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    for name, c in doc["cohorts"].items():
        print(f"{name}: " + (c["why"] if c["band"] is None else
                             ", ".join(f"{t} {v['min']:.4f}-{v['max']:.4f} ({v['rows_over']} over)" for t, v in c["band"].items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
