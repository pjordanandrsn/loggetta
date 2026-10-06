"""Turn experts4bit-qlora lane SV3's arm receipts (``sv3-arm/1``: serve_paged built in-process on a rented sm_89+ card;
Qwen3.6-35B-A3B and gpt-oss-20b all-VRAM, decode graphs and buckets per arm) into planner observations of kind
``serve``, and print the lane's readings W1-W5 against bench/sv3/SV3-PREREG.md.

    python bench/import_sv3.py RUN_DIR/tc1 --run-id sv3-5090-1 --out evidence/2026-10-06-sv3-rtx5090
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
        if r.get("schema") != "sv3-arm/1":
            continue
        tag = os.path.basename(f)[:-5]
        arms[tag] = r
        m = r.get("measured", {})
        obs = {
            # an arm whose decode buckets did not all capture failed the prereg's integrity: its memory is not that
            # setup's, so the planner must not learn from it
            "schema": "execution-receipt/1", "run_id": f"{a.run_id}/{tag}",
            "status": "ALARM" if any(v != "graph" for v in (r.get("graph_status") or {}).values()) else r.get("status"),
            "source": f"experts4bit-qlora lane SV3 ({a.run_id}), arm {tag}",
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
                "host_anon_serving_peak_bytes")},
            "estimate": r.get("estimate"), "graph_status": r.get("graph_status"), "prefill_graph": r.get("prefill_graph"),
            "expert_routes_seen": r.get("expert_routes_seen"), "info": r.get("info"), "linear_state": r.get("linear_state"),
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

    for w, tag in (("W1", "q36_e16"), ("W3", "q36_g16"), ("W5", "gptoss_g16")):
        if m(tag, "device_peak_bytes"):
            rel = m(tag, "device_peak_bytes") / est(tag) - 1
            print(f"{w} {tag} peak vs its estimate: {rel * 100:+.2f}% (expected within +-5%): {held(abs(rel) <= 0.05)}")
    pools = {t: (arms[t].get("linear_state") or {}) for t in arms if t.startswith("q36")}
    print("W2 the state pool, exact:", held(all(p.get("equal") for p in pools.values())),
          {t: (p.get("nbytes"), p.get("n_slots")) for t, p in pools.items()})
    if m("q36_g1_default", "device_peak_bytes") and m("q36_g1_capped", "device_peak_bytes"):
        d = (m("q36_g1_default", "device_peak_bytes") - m("q36_g1_capped", "device_peak_bytes")) / MiB
        priced = (est("q36_g1_default") - est("q36_g1_capped")) / MiB
        ok = graphs_ok("q36_g1_default") and graphs_ok("q36_g1_capped")
        print(f"W4 the bucket cap: {d:+.1f} MiB vs priced {priced:+.1f} (-32/+256): "
              + (held(-32 <= d - priced <= 256) if ok else "ALARM (a decode bucket did not capture; not read)"))

if __name__ == "__main__":
    main()
