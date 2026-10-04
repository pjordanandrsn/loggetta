"""Turn experts4bit-qlora lane FP1's arm receipts (``fp1-arm/1``, measured on a rented box) into planner observations
(``execution-receipt/1``), so its allocator slack, context and host baseline replace extrapolated figures for that GPU.

    python bench/import_fp1.py RUN_DIR/tc1 --out evidence/<date>-fp1/   # RUN_DIR = the launcher's run directory

Only what FP1 measured is carried over; nothing is invented. The GPU name and driver come from the box's forensics line.
"""
import argparse
import glob
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fetched", help="the fetched box directory (holds receipts/ and forensics.txt)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-id", required=True, help="the launcher's run id, e.g. fp1-5090-1")
    a = ap.parse_args()
    forensics = open(os.path.join(a.fetched, "forensics.txt")).read().splitlines()[0].split(",")
    gpu = {"name": forensics[0].strip(), "driver": forensics[2].strip(), "memory_total": forensics[1].strip()}
    launcher = os.path.join(os.path.dirname(os.path.abspath(a.fetched)), "receipt.json")
    started = json.load(open(launcher)).get("started_at", "") if os.path.exists(launcher) else ""
    os.makedirs(a.out, exist_ok=True)
    for f in sorted(glob.glob(os.path.join(a.fetched, "receipts", "*.json"))):
        r = json.load(open(f))
        if r.get("schema") != "fp1-arm/1":
            continue
        arm = os.path.basename(f)[:-5]
        args = r["args"]
        obs = {
            "schema": "execution-receipt/1",
            "run_id": f"{a.run_id}/{arm}",
            "status": r["status"],
            "source": f"experts4bit-qlora lane FP1 ({a.run_id}), arm {arm}",
            "model": {"model": args["model"], "revision": args["revision"]},
            "workload": {"seq_len": args["seq"], "micro_batch": args["mb"], "steps": args["steps"],
                         "tokens_per_microbatch": args["seq"] * args["mb"]},
            "setup": r["setup"],
            "hardware": {"gpu": gpu},
            "measured": {k: r["measured"].get(k) for k in (
                "device_peak_bytes", "device_reserved_peak_bytes", "driver_process_peak_bytes", "cuda_context_bytes",
                "host_baseline_bytes", "host_anon_after_load_bytes", "host_required_peak_bytes", "link_h2d_gbps",
                "s_per_step_median", "tokens_per_s", "load_seconds")},
            "estimate": r["estimate"],
            "correctness": r.get("correctness"),
            "provenance": {"started_at": started, "imported_from": os.path.abspath(f),
                           "launcher_receipt": launcher if started else None},
        }
        with open(os.path.join(a.out, f"{a.run_id}-{arm}.json"), "w") as fh:
            json.dump(obs, fh, indent=1)
        print(arm, r["status"])


if __name__ == "__main__":
    main()
