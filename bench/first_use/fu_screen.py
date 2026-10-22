#!/usr/bin/env python3
"""T1 step 3's host screen (bench/first_use/PREREG-first_use.md): read-only Vast offer searches, bucketed by the
listed driver into the reducer's bands, then a machine-filtered stability read of each band's cheapest candidates, so a
cell can be pinned to a machine. It rents nothing.

The listing is only the screen: the box's own nvidia-smi read grades the cell (fu_reduce.py), so a host updated since
its listing reads CELL_MISMATCH rather than silently moving the cell.

Vast's unpinned search returns a rotating ~30-offer sample, so the report states its sample ("N offers seen across K
reads"), never "the cheapest on the market". A pin is checked with ``machine_id {eq}`` reads, the shape a launch trusts.

    VAST_API_KEY_FILE=~/.vast/key fu_screen.py [--reads 3] [--stable-reads 6] [--per-band 3] --out screen.json
    fu_screen.py --offers saved.json ...        # offline: bucket a saved offer list, no stability reads
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fu_reduce import BANDS, band_of  # noqa: E402

#: the cheapest classes registered for the cells (PREREG, "Cards")
CLASSES = ("RTX A4000", "RTX 3090", "RTX 4090", "RTX 5090")
#: the launch's own floors for this lane (disk for torch + a 13 GB checkpoint per cell, three cells per box)
MIN_DISK_GB, MIN_RAM_GB = 80, 24


API = "https://console.vast.ai/api"


def offer_filter(gpu: str, *, min_disk_gb: int, min_ram_gb: int) -> dict:
    """The search filter: a verified, rentable, on-demand single-GPU offer of the class. The launch re-reads the pinned
    machine at create time, so this screen only proposes hosts."""
    return {"verified": {"eq": True}, "rentable": {"eq": True}, "external": {"eq": False}, "gpu_name": {"eq": gpu},
            "num_gpus": {"eq": 1}, "disk_space": {"gte": min_disk_gb}, "cpu_ram": {"gte": min_ram_gb * 1024},
            "type": "on-demand", "order": [["dph_total", "asc"]]}


class Transport:
    """Read-only HTTP. The key is a header, never part of a URL, an error or a log line."""

    def __init__(self, key: str):
        self._key = key

    def request(self, method: str, path: str, *, query=None, body=None, timeout: float = 30.0):
        url = API + path + ("?" + urllib.parse.urlencode(query) if query else "")
        req = urllib.request.Request(url, method=method,
                                     headers={"Authorization": f"Bearer {self._key}", "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            return e.code, {}


def bucket(offers: list, per_band: int) -> dict:
    """{band: [cheapest per_band offers, one per machine]}, from the listed driver; offers outside every band are
    counted, not kept."""
    seen, out = {}, {b: [] for b in BANDS}
    for o in sorted(offers, key=lambda o: (o.get("dph_total") or 1e9)):
        m = str(o.get("machine_id"))
        if m in seen:
            continue
        seen[m] = True
        b = band_of(o.get("driver_version"))
        if b and len(out[b]) < per_band:
            out[b].append({"machine_id": m, "offer_id": o.get("id"), "gpu": o.get("gpu_name"),
                           "listed_driver": o.get("driver_version"), "cuda_max_good": o.get("cuda_max_good"),
                           "dph_total": o.get("dph_total"), "inet_down": o.get("inet_down")})
    return out


def search(transport, gpu: str, extra: dict | None = None, *, filter_fn) -> list:
    q = {**filter_fn(gpu, min_disk_gb=MIN_DISK_GB, min_ram_gb=MIN_RAM_GB), **(extra or {})}
    status, body = transport.request("GET", "/v0/bundles/", query={"q": json.dumps(q, sort_keys=True)})
    if status != 200 or not isinstance(body, dict) or not isinstance(body.get("offers"), list):
        raise RuntimeError(f"offer search did not answer with offers: HTTP {status}")   # never a silent zero
    return body["offers"]


def screen(transport, *, filter_fn, reads: int, stable_reads: int, per_band: int, pace_s: float = 0.5,
           sleep=time.sleep) -> dict:
    offers, n_reads = {}, 0
    for gpu in CLASSES:
        for _ in range(reads):
            for o in search(transport, gpu, filter_fn=filter_fn):
                offers[str(o.get("id"))] = o
            n_reads += 1
            sleep(pace_s)
    bands = bucket(list(offers.values()), per_band)
    for cands in bands.values():
        for c in cands:
            hits, drivers = 0, set()
            for _ in range(stable_reads):
                got = [o for o in search(transport, c["gpu"], {"machine_id": {"eq": int(c["machine_id"])}},
                                         filter_fn=filter_fn) if str(o.get("machine_id")) == c["machine_id"]]
                hits += bool(got)
                drivers |= {o.get("driver_version") for o in got}
                sleep(pace_s)
            c.update(stable=f"{hits}/{stable_reads}", stable_ok=hits == stable_reads,
                     driver_reads=sorted(d for d in drivers if d))
    return {"schema": "first-use-screen/1", "taken_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "sample": f"{len(offers)} offers seen across {n_reads} reads ({', '.join(CLASSES)})",
            "bands": {b: list(BANDS[b]) for b in BANDS}, "candidates": bands}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--offers", help="a saved offer list (JSON): bucket it offline, no stability reads")
    ap.add_argument("--reads", type=int, default=3)
    ap.add_argument("--stable-reads", type=int, default=6)
    ap.add_argument("--per-band", type=int, default=3)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    if a.offers:
        with open(a.offers) as f:
            data = json.load(f)
        offers = data["offers"] if isinstance(data, dict) else data
        out = {"schema": "first-use-screen/1", "sample": f"{len(offers)} offers from {os.path.basename(a.offers)}",
               "bands": {b: list(BANDS[b]) for b in BANDS}, "candidates": bucket(offers, a.per_band)}
    else:
        kf = os.environ.get("VAST_API_KEY_FILE") or ""
        key = open(os.path.expanduser(kf)).read().strip() if kf and os.path.exists(os.path.expanduser(kf)) else ""
        if not key:
            ap.error("VAST_API_KEY_FILE must name a file holding the key (never pass the key on a command line)")
        out = screen(Transport(key), filter_fn=offer_filter, reads=a.reads, stable_reads=a.stable_reads,
                     per_band=a.per_band)
    text = json.dumps(out, indent=1, sort_keys=True)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text + "\n")
    print(out["sample"])
    for b, cands in out["candidates"].items():
        for c in cands:
            print(f"{b:8s} machine {c['machine_id']:>7s} {c['gpu']:10s} driver {c['listed_driver']} "
                  f"${c['dph_total']}/h stable {c.get('stable', '-')}")
        if not cands:
            print(f"{b:8s} no candidate in this sample")
    return 0


if __name__ == "__main__":
    sys.exit(main())
