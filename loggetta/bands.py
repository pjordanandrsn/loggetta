"""The planner's held-out band: what measured / estimate held-out rows OBSERVED for this model, card and placement.

The band is display-only. It never changes an estimate, a budget, feasibility or which plan is selected. It reads
``planner_bands.json`` (built by ``bench/build_planner_bands.py`` from the frozen split, ``evidence/planner-split/``) and
matches exactly: the model's repository id, the card, and where the experts sit. Nothing generalizes across sizes, siblings
or cards. A band is the range those rows showed, on the machines they ran on. It is not a prediction, and a further
machine may fall outside it.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

MiB = 1 << 20
BANDS_FILE = Path(__file__).with_name("planner_bands.json")
ORDER = ("device_driver", "device_allocator", "host")
TERM_NAME = {"device_driver": "device (driver peak / plan total)", "device_allocator": "device (allocated peak / allocator estimate)",
             "host": "host (host peak / host estimate)"}
SHORT = {"device_driver": "device driver", "device_allocator": "allocator", "host": "host"}
UNMEASURED = "unmeasured family/card: no band"


@lru_cache(maxsize=1)
def table() -> dict:
    return json.loads(BANDS_FILE.read_text(encoding="utf-8"))


def card_name(gpu_name: str | None) -> str | None:
    """The split's card name for a detected GPU name: 'NVIDIA GeForce RTX 5090' -> 'RTX 5090'."""
    if not gpu_name:
        return None
    n = str(gpu_name).strip()
    for prefix in ("NVIDIA ", "GeForce "):
        if n.startswith(prefix):
            n = n[len(prefix):]
    return n


def lookup(model: str | None, gpu_name: str | None, residency: str | None, bands: dict | None = None) -> dict:
    """The band record for one plan: ``status`` is "band", "one box" or "unmeasured"."""
    bands = bands if bands is not None else table()
    card = card_name(gpu_name)
    for name, c in bands["cohorts"].items():
        if c["model"] == model and c["card"] == card and c["placement"] == residency:
            break
    else:
        return {"status": "unmeasured", "split": bands["split"]}
    base = {"cohort": name, "split": bands["split"], "pinned_at": bands["pinned_at"], "held_out_rows": c["held_out_rows"]}
    if c["band"] is None:
        return {**base, "status": "unmeasured" if c["why"] == "unmeasured" else "one box", "boxes": c["boxes"]}
    rows = c["rows"]
    over = {t: max(r["terms"][t]["measured"] - r["terms"][t]["estimated"] for r in rows) for t in c["band"]}
    return {**base, "status": "band", "model": model, "card": card, "placement": residency, "machines": c["boxes"],
            "seqs": sorted({r["seq"] for r in rows}), "rows": [{"id": r["id"], "sha256": r["sha256"]} for r in rows],
            "terms": {t: {**c["band"][t], "max_over_bytes": over[t]} for t in ORDER if t in c["band"]}}


def render(b: dict) -> list:
    """The plan's band line(s), under its device and host totals."""
    label = "held-out band"
    if b["status"] == "unmeasured":
        return [f"  {label:44s} {UNMEASURED}"]
    if b["status"] == "one box":
        return [f"  {label:44s} held-out rows from one box only: no band ({b['held_out_rows']} rows, {b['split']})"]
    seqs = " and ".join(f"{s:,}" for s in b["seqs"])
    ranges = "; ".join(f"{SHORT[t]} {v['min']:.3f}-{v['max']:.3f}" for t, v in b["terms"].items())
    return [f"  {label:44s} observed on {b['held_out_rows']} held-out rows from {b['machines']} machines "
            f"(seq {seqs}; {b['split']}): measured / estimate {ranges}"]


def warnings(b: dict) -> list:
    """One warning per term whose estimate ran low on a held-out row, so the miss is read where the plan is read."""
    if b.get("status") != "band":
        return []
    out = []
    for t, v in b["terms"].items():
        if v["rows_over"]:
            out.append(f"the {TERM_NAME[t]} estimate ran LOW on {v['rows_over']} of {b['held_out_rows']} held-out rows for "
                       f"{b['model']} on {b['card']} (observed up to {v['max']:.3f}x, {v['max_over_bytes'] / MiB:.0f} MiB over, "
                       f"on {b['machines']} machines; a further machine may fall outside): keep headroom for it")
    return out
