"""Turn experts4bit-qlora lane SV2's arm receipts (``sv2-arm/1``: serve_paged built in-process on a rented sm_89+ card,
Qwen3-30B-A3B all-VRAM with decode graphs, the int4 serving levers on or off) into planner observations of kind
``serve``, and print the lane's readings V1-V6 against the expectations registered in bench/sv2/SV2-PREREG.md.

    python bench/import_sv2.py RUN_DIR/tc1 --run-id sv2-5090-1 --out evidence/2026-10-05-sv2-rtx5090
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
        if r.get("schema") != "sv2-arm/1":
            continue
        tag = os.path.basename(f)[:-5]
        arms[tag] = r
        m = r.get("measured", {})
        obs = {
            "schema": "execution-receipt/1", "run_id": f"{a.run_id}/{tag}", "status": r.get("status"),
            "source": f"experts4bit-qlora lane SV2 ({a.run_id}), arm {tag}",
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
            "expert_routes_seen": r.get("expert_routes_seen"), "info": r.get("info"),
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

    if arms.get("q_exp") and m("q_exp", "device_peak_bytes"):
        rel = m("q_exp", "device_peak_bytes") / est("q_exp") - 1
        print(f"V1 q_exp peak vs its estimate: {rel * 100:+.1f}% (expected within +-5%): {held(abs(rel) <= 0.05)}")
    if m("q_exp", "load_device_peak_bytes") and m("q_nf4", "load_device_peak_bytes"):
        d = (m("q_exp", "load_device_peak_bytes") - m("q_nf4", "load_device_peak_bytes")) / MiB
        priced = (est("q_exp") - est("q_nf4")) / MiB
        sd = (m("q_exp", "device_peak_bytes") - m("q_nf4", "device_peak_bytes")) / MiB
        print(f"V2 int4 stores vs NF4: load peak {d:+.1f} MiB vs priced {priced:+.1f} (+-32): {held(abs(d - priced) <= 32)};"
              f" serving peak {sd:+.1f} MiB (reported)")
    if m("q_both", "device_peak_bytes") and m("q_exp", "device_peak_bytes"):
        d = (m("q_both", "device_peak_bytes") - m("q_exp", "device_peak_bytes")) / MiB
        priced = (est("q_both") - est("q_exp")) / MiB
        print(f"V3 int4 attention: {d:+.1f} MiB vs priced {priced:+.1f} (+-64): {held(abs(d - priced) <= 64)}")
    if m("q_exp_prefill", "device_peak_bytes") and m("q_exp", "device_peak_bytes"):
        d = (m("q_exp_prefill", "device_peak_bytes") - m("q_exp", "device_peak_bytes")) / MiB
        dr = (m("q_exp_prefill", "device_reserved_peak_bytes") - m("q_exp", "device_reserved_peak_bytes")) / MiB
        print(f"V4 the prefill graph at int4: {d:+.1f} MiB allocated, {dr:+.1f} MiB reserved; runner's pool "
              f"{(arms['q_exp_prefill'].get('prefill_graph') or {}).get('pool_mib')} MiB (measured)")
    if m("q_exp", "host_anon_load_peak_bytes") and m("q_nf4", "host_anon_after_load_bytes"):
        d = m("q_exp", "host_anon_load_peak_bytes") - m("q_nf4", "host_anon_after_load_bytes")
        priced = est("q_exp", "int4 repack", where="host")
        numel = 128 * 3 * 2048 * 768
        print(f"V5 repack host peak: {d / MiB:.0f} MiB vs priced {priced / MiB:.0f} (at or under): {held(d <= priced)};"
              f" {d / numel:.1f} B per parameter of a layer")
    for tag in ("q_exp", "q_both"):
        if m(tag, "host_anon_after_load_bytes") and m("q_nf4", "host_anon_after_load_bytes"):
            d = (m(tag, "host_anon_after_load_bytes") - m("q_nf4", "host_anon_after_load_bytes")) / 1e9
            print(f"V6 {tag} after-load host vs q_nf4: {d:+.2f} GB (within +0.25): {held(d <= 0.25)}")


if __name__ == "__main__":
    main()
