"""Turn experts4bit-qlora lane SV4's arm receipts (``sv4-arm/1``: serve_paged built in-process on a rented RTX 4090;
Qwen3-30B-A3B all-VRAM and on the solver's tiers) into planner observations of kind ``serve``, and print the lane's
readings X1-X5 against bench/sv4/SV4-PREREG.md.

    python bench/import_sv4.py RUN_DIR/tc1 --run-id sv4-4090-1 --out evidence/2026-10-06-sv4-rtx4090
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
        if r.get("schema") != "sv4-arm/1":
            continue
        tag = os.path.basename(f)[:-5]
        arms[tag] = r
        m = r.get("measured", {})
        obs = {
            # an arm whose decode buckets did not all capture failed the prereg's integrity: its memory is not that
            # setup's, so the planner must not learn from it
            "schema": "execution-receipt/1", "run_id": f"{a.run_id}/{tag}",
            "status": "ALARM" if any(v != "graph" for v in (r.get("graph_status") or {}).values()) else r.get("status"),
            "source": f"experts4bit-qlora lane SV4 ({a.run_id}), arm {tag}",
            "model": {"model": r["args"]["model"], "revision": r["args"]["revision"]},
            "workload": {"kind": "serve", "context_len": r["setup"]["max_tokens_per_seq"],
                         "concurrency": r["setup"]["max_seqs"], "prompt_tokens": r["args"]["prompt_tokens"],
                         "new_tokens": r["args"]["new_tokens"]},
            "setup": {**r["setup"], "buckets": list(r["setup"]["buckets"])},
            "hardware": {"gpu": gpu},
            "measured": {k: m.get(k) for k in (
                "device_peak_bytes", "device_reserved_peak_bytes", "driver_process_peak_bytes", "load_device_peak_bytes",
                "load_reserved_peak_bytes", "load_seconds", "tokens_per_s", "requests_done", "host_anon_start_bytes",
                "host_anon_load_peak_bytes", "host_anon_after_load_bytes", "host_anon_peak_bytes",
                "host_anon_serving_peak_bytes", "host_shmem_peak_bytes")},
            "estimate": r.get("estimate"), "graph_status": r.get("graph_status"), "prefill_graph": r.get("prefill_graph"),
            "expert_routes_seen": r.get("expert_routes_seen"), "info": r.get("info"), "linear_state": r.get("linear_state"), "server_tiers": r.get("server_tiers"), "error": r.get("error"),
            "provenance": {"started_at": started, "imported_from": os.path.abspath(f), "versions": r.get("versions")},
        }
        if m.get("driver_process_peak_bytes") and m.get("device_reserved_peak_bytes"):
            obs["measured"]["cuda_context_bytes"] = m["driver_process_peak_bytes"] - m["device_reserved_peak_bytes"]
        with open(os.path.join(a.out, f"{a.run_id}-{tag}.json"), "w") as fh:
            json.dump(obs, fh, indent=1)

    def m(tag, key):
        return (arms.get(tag) or {}).get("measured", {}).get(key)

    def est(tag, name=None, where="device"):
        e = (arms.get(tag) or {}).get("estimate") or {}
        if name is None:
            return e.get("device_total")
        return sum(i["bytes"] for i in e.get("items", ()) if i["where"] == where and i["name"].startswith(name))

    for tag, r in arms.items():
        i, pg = r.get("info") or {}, r.get("prefill_graph") or {}
        print(f"{tag:14s} {r.get('status'):8s} est {(est(tag) or 0) / GiB:6.3f}  peak {(m(tag, 'device_peak_bytes') or 0) / GiB:6.3f}"
              f"  load {(m(tag, 'load_device_peak_bytes') or 0) / GiB:6.3f}  reserved {(m(tag, 'device_reserved_peak_bytes') or 0) / GiB:6.3f}"
              f"  driver {(m(tag, 'driver_process_peak_bytes') or 0) / GiB:6.3f} GiB | host anon load-peak "
              f"{(m(tag, 'host_anon_load_peak_bytes') or 0) / 1e9:.2f} after {(m(tag, 'host_anon_after_load_bytes') or 0) / 1e9:.2f}"
              f" serving {(m(tag, 'host_anon_serving_peak_bytes') or 0) / 1e9:.2f} GB | int4 {i.get('int4_expert_layers')}/"
              f"{i.get('int4_attn_projections')} | requests {m(tag, 'requests_done')} | tok/s {m(tag, 'tokens_per_s')}"
              f" | prefill graph {pg.get('status')} replays {pg.get('replays')} pool {pg.get('pool_mib')} MiB"
              f" | decode graphs {'all graph' if r.get('graph_status') and all(v == 'graph' for v in r['graph_status'].values()) else r.get('graph_status')}")

    def held(ok):
        return "HELD" if ok else "MISSED"

    def graphs_ok(tag):
        gs = (arms.get(tag) or {}).get("graph_status") or {}
        return all(v == "graph" for v in gs.values())

    for x, tag in (("X1", "t4_all1"), ("X2", "t4_plan8"), ("X3", "t4_deep4")):
        if m(tag, "device_peak_bytes"):
            rel = m(tag, "device_peak_bytes") / est(tag) - 1
            ok = graphs_ok(tag)
            print(f"{x} {tag} peak vs its estimate: {rel * 100:+.2f}% (expected within +-5%): "
                  + (held(abs(rel) <= 0.05) if ok else "ALARM (a decode bucket did not capture)"))
    for tag in ("t4_plan8", "t4_deep4"):
        r = arms.get(tag) or {}
        items = {i["name"]: i["bytes"] for i in (r.get("estimate") or {}).get("items", ())}
        bpe = None
        priced = {}
        for name, b in items.items():
            if name.startswith("expert stacks, VRAM tier"):
                priced["vram"] = b
            elif name.startswith("expert stacks, DRAM tier"):
                priced["dram"] = b
            elif name.startswith("expert rows on NVMe"):
                priced["nvme"] = b
        st = r.get("server_tiers") or {}
        if st and priced:
            total_rows = sum(st.values())
            bpe = sum(priced.values()) // total_rows if total_rows else None
            est_rows = {k: v // bpe for k, v in priced.items()} if bpe else {}
            print(f"X4 {tag} split: server {st} vs estimate {est_rows}: {held(all(st.get(k) == est_rows.get(k) for k in st))}")
    if m("t4_deep4", "host_shmem_peak_bytes"):
        sh = m("t4_deep4", "host_shmem_peak_bytes") / MiB
        landing = est("t4_deep4", "cold tier landing", where="host") / MiB
        print(f"X5 t4_deep4 pinned host peak {sh:.0f} MiB vs the {landing:.0f} MiB landing (>= it, within +256): "
              f"{held(landing <= sh <= landing + 256)}")
    for tag in arms:
        mm = (arms[tag].get("measured") or {})
        if mm.get("device_peak_bytes") and mm.get("device_reserved_peak_bytes"):
            print(f"   slack {tag}: {100 * (mm['device_reserved_peak_bytes'] / mm['device_peak_bytes'] - 1):.2f}%")

if __name__ == "__main__":
    main()
