"""Turn experts4bit-qlora lane SV1's arm receipts (``sv1-arm/1``: serve_paged built in-process on a rented sm_89+ card,
all-VRAM, decode graphs and the prefill graph on or off) into planner observations of kind ``serve``, and print the
lane's readings S1-S4.

    python bench/import_sv1.py RUN_DIR/tc1 --run-id sv1-5090-1 --out evidence/2026-10-05-sv1-rtx5090
"""
import argparse
import glob
import json
import os

GiB, MiB = 1 << 30, 1 << 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fetched", help="the fetched box directory (holds receipts/ and forensics.txt)")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    forensics = open(os.path.join(a.fetched, "forensics.txt")).read().splitlines()[0].split(",")
    gpu = {"name": forensics[0].strip(), "memory_total": forensics[1].strip(), "driver": forensics[2].strip()}
    launcher = os.path.join(os.path.dirname(os.path.abspath(a.fetched)), "receipt.json")
    started = json.load(open(launcher)).get("started_at", "") if os.path.exists(launcher) else ""
    os.makedirs(a.out, exist_ok=True)
    arms = {}
    for f in sorted(glob.glob(os.path.join(a.fetched, "receipts", "*.json"))):
        r = json.load(open(f))
        if r.get("schema") != "sv1-arm/1":
            continue
        tag = os.path.basename(f)[:-5]
        arms[tag] = r
        m = r.get("measured", {})
        obs = {
            "schema": "execution-receipt/1", "run_id": f"{a.run_id}/{tag}", "status": r.get("status"),
            "source": f"experts4bit-qlora lane SV1 ({a.run_id}), arm {tag}",
            "model": {"model": r["args"]["model"], "revision": r["args"]["revision"]},
            "workload": {"kind": "serve", "context_len": r["setup"]["max_tokens_per_seq"],
                         "concurrency": r["setup"]["max_seqs"], "prompt_tokens": r["args"]["prompt_tokens"],
                         "new_tokens": r["args"]["new_tokens"]},
            "setup": {**r["setup"], "buckets": list(r["setup"]["buckets"])},
            "hardware": {"gpu": gpu},
            "measured": {k: m.get(k) for k in ("device_peak_bytes", "device_reserved_peak_bytes", "driver_process_peak_bytes",
                                                "load_device_peak_bytes", "load_reserved_peak_bytes", "load_seconds",
                                                "tokens_per_s", "requests_done")},
            "estimate": r.get("estimate"), "graph_status": r.get("graph_status"), "prefill_graph": r.get("prefill_graph"),
            "provenance": {"started_at": started, "imported_from": os.path.abspath(f), "versions": r.get("versions")},
        }
        if m.get("driver_process_peak_bytes") and m.get("device_reserved_peak_bytes"):
            obs["measured"]["cuda_context_bytes"] = m["driver_process_peak_bytes"] - m["device_reserved_peak_bytes"]
        with open(os.path.join(a.out, f"{a.run_id}-{tag}.json"), "w") as fh:
            json.dump(obs, fh, indent=1)

    def peak(tag, key="device_peak_bytes"):
        return (arms.get(tag) or {}).get("measured", {}).get(key)

    for tag, r in arms.items():
        m, e = r.get("measured", {}), r["estimate"]["device_total"]
        print(f"{tag:14s} {r.get('status'):10s} estimate {e / GiB:6.3f} GiB  peak {(m.get('device_peak_bytes') or 0) / GiB:6.3f}"
              f"  reserved {(m.get('device_reserved_peak_bytes') or 0) / GiB:6.3f}  driver {(m.get('driver_process_peak_bytes') or 0) / GiB:6.3f}"
              f"  prefill_graph {(r.get('prefill_graph') or {}).get('status')} pool {(r.get('prefill_graph') or {}).get('pool_mib')} MiB")
    for model in ("olmoe", "qwen3"):
        e, g, p = peak(f"{model}_eager"), peak(f"{model}_graphs"), peak(f"{model}_prefill")
        if e and g:
            print(f"S2 {model}: decode graphs add {(g - e) / MiB:.0f} MiB allocated, "
                  f"{(peak(f'{model}_graphs', 'device_reserved_peak_bytes') - peak(f'{model}_eager', 'device_reserved_peak_bytes')) / MiB:.0f} MiB reserved")
        if g and p:
            print(f"S3 {model}: the prefill graph adds {(p - g) / MiB:.0f} MiB allocated, "
                  f"{(peak(f'{model}_prefill', 'device_reserved_peak_bytes') - peak(f'{model}_graphs', 'device_reserved_peak_bytes')) / MiB:.0f} MiB reserved")


if __name__ == "__main__":
    main()
