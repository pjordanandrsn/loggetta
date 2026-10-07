"""Record ``setup.bulk_kv`` on serve observations imported before the field existed, with the basis for each value.

experts4bit-qlora's ``ServeSetup`` gained ``bulk_kv`` with #1247 (2026-10-06), so every serve plan's setup now names it,
and a receipt imported earlier does not. Whole-setup and same-shape reserve matching then never fired for those receipts.
The planner cannot fill one default for them: the server's default changed with #1200 (merged 2026-10-05T19:32:02Z),
and lane SV2 ran three hours after that from a branch without it.

Each value below comes from the code the run used:
- for lanes, whether their recorded experts4bit-qlora commit contains #1200 (``2ee15a1c``); none set
  ``E4B_PAGED_BULK_KV``, so they ran the default of their commit;
- for seat runs on 2026-10-05, that they ran before #1200 existed, when bulk bookkeeping was opt-in and nothing opted in.

The A2000 family runs of 2026-10-06 record no commit, so they are left without the field, and they keep matching only
by their key fields. Idempotent: an observation that already names ``bulk_kv`` is left alone.

    python bench/annotate_bulk_kv.py            # dry run: prints what it would set
    python bench/annotate_bulk_kv.py --write
"""
import argparse
import glob
import json
import os

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "evidence")
DEFAULT_ON = "experts4bit-qlora #1200 (2ee15a1c, merged 2026-10-05T19:32:02Z): E4B_PAGED_BULK_KV defaults to 1"
TABLE = {
    "2026-10-04-p109-rtx5090-serve": (False, "lane P109 ran experts4bit-qlora 3185b46a, which lacks " + DEFAULT_ON),
    "2026-10-05-rtx-a2000-serve": (False, "seat runs 2026-10-04T23:58Z-10-05T11:58Z, before " + DEFAULT_ON +
                                   "; bulk bookkeeping was opt-in and no run opted in"),
    "2026-10-05-sv1-rtx5090": (False, "lane SV1 ran experts4bit-qlora 547599b, which lacks " + DEFAULT_ON),
    "2026-10-05-a2000-int4-serve": (False, "seat runs 2026-10-05T17:12-17:22Z, before " + DEFAULT_ON +
                                    "; bulk bookkeeping was opt-in and no run opted in"),
    "2026-10-05-sv2-rtx5090": (False, "lane SV2 ran experts4bit-qlora 467f25d, which lacks " + DEFAULT_ON),
    "2026-10-06-sv3-rtx5090": (True, "lane SV3 ran experts4bit-qlora c500651, which contains " + DEFAULT_ON +
                               "; the receipt sets no override"),
    "2026-10-06-sv4-rtx4090": (True, "lane SV4 ran experts4bit-qlora e80b04a, which contains " + DEFAULT_ON +
                               "; the receipt sets no override"),
    "2026-10-06-sv5-rtx4090": (True, "lane SV5 ran experts4bit-qlora f49006f5, which contains " + DEFAULT_ON +
                               "; the receipt sets no override"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    for d, (value, basis) in sorted(TABLE.items()):
        for f in sorted(glob.glob(os.path.join(HERE, d, "*.json"))):
            r = json.load(open(f))
            if r.get("schema") != "execution-receipt/1" or r.get("workload", {}).get("kind") != "serve":
                continue
            setup = r.get("setup")
            if not isinstance(setup, dict) or "bulk_kv" in setup:
                continue
            print(f"{'set' if a.write else 'would set'} bulk_kv={value} in {d}/{os.path.basename(f)}")
            if a.write:
                setup["bulk_kv"] = value
                r.setdefault("provenance", {}).setdefault("setup_inferred", {})["bulk_kv"] = basis
                with open(f, "w") as fh:
                    json.dump(r, fh, indent=1)


if __name__ == "__main__":
    main()
