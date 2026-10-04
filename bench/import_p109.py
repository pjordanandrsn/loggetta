"""Turn experts4bit-qlora lane P109's arm receipts (``serve_paged`` all-VRAM on one RTX 5090, Qwen3-30B-A3B,
16 x 4096) into planner observations of workload kind ``serve``, so serve plans on that card read a measured
allocator slack and context instead of the stated default.

    python bench/import_p109.py ../e4b/bench/p109/receipts/p109-5090-2 --out evidence/2026-10-04-p109-rtx5090-serve

Only what P109 recorded is carried over. Its memory was read once after load and once after the runs, not
sampled, so:
* ``device_peak_bytes`` is ``torch.cuda.max_memory_allocated()`` after the runs (a true peak);
* ``device_reserved_peak_bytes`` is ``memory_reserved()`` after the runs (the reserve at that moment; the caching
  allocator does not return blocks unprompted, so it is a lower bound on the reserved peak);
* ``cuda_context_bytes`` is (total - free) - reserved at that same moment: everything the device held that the
  allocator did not, on a rented box running nothing else (stated as such in ``notes``).
"""
import argparse
import glob
import json
import os

SETUP_KEYS = ("placement", "max_seqs", "max_tokens_per_seq", "chunk_tokens", "graphs", "buckets", "kv_groups")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("receipts")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    run = os.path.basename(os.path.normpath(a.receipts))
    for f in sorted(glob.glob(os.path.join(a.receipts, "arm_*.json"))):
        r = json.load(open(f))
        cfg, after = r["config"], r["mem_after_runs"]
        name, used_mib, total_mib, driver = [x.strip() for x in r["nvidia_smi"].split(",")]
        arm = r.get("tag") or os.path.basename(f)[4:-5]
        if r.get("status") != "ok" or r["census"].get("int4_expert_layers") or r["census"].get("int4_attn_projections"):
            print(arm, "skipped:", r.get("status"), "or an int4 stack the estimate does not price")
            continue
        reserved = after["memory_reserved"]
        obs = {
            "schema": "execution-receipt/1",
            "run_id": f"{run}/{arm}",
            "status": "OK",
            "source": f"experts4bit-qlora lane P109 ({run}), arm {arm}",
            "model": {"model": r["model"], "revision": r["revision"]},
            "workload": {"kind": "serve", "context_len": cfg["max_tokens_per_seq"], "concurrency": cfg["max_seqs"]},
            "setup": {k: (list(cfg[k]) if k == "buckets" else cfg[k]) for k in SETUP_KEYS},
            "hardware": {"gpu": {"name": name, "driver": driver, "memory_total": total_mib}},
            "measured": {"device_peak_bytes": after["max_memory_allocated"], "device_reserved_peak_bytes": reserved,
                         "cuda_context_bytes": (after["total"] - after["free"]) - reserved,
                         "load_device_peak_bytes": r["mem_after_load"]["max_memory_allocated"],
                         "load_seconds": r.get("load_s")},
            "notes": ["memory read after load and after the runs, not sampled: reserved is the end-of-run reserve "
                      "(a lower bound on its peak); the context is (total - free) - reserved at that moment",
                      f"nvidia-smi after the runs: {used_mib} used"],
            "provenance": {"started_at": "", "imported_from": os.path.abspath(f), "e4b_sha": r.get("e4b_sha"),
                           "gnf4_sha": r.get("gnf4_sha"), "torch": r.get("torch")},
        }
        with open(os.path.join(a.out, f"{run}-{arm}.json"), "w") as fh:
            json.dump(obs, fh, indent=1)
        m = obs["measured"]
        print(arm, f"graphs={cfg['graphs']} alloc {m['device_peak_bytes'] / 2**30:.2f} GiB reserved "
                   f"{reserved / 2**30:.2f} slack {reserved / m['device_peak_bytes'] - 1:.4f} "
                   f"context {m['cuda_context_bytes'] / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
