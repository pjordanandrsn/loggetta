"""Tables for docs/RESULTS.md, generated from receipts (never typed by hand).

    python bench/summarize_receipts.py RECEIPT_DIR > results.md
"""
import glob
import json
import os
import sys

G = 2 ** 30


def gib(v):
    return "n/a" if v is None else f"{v / G:.2f}"


def main(d):
    rs = []
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        r = json.load(open(f))
        if r.get("schema") == "execution-receipt/1":
            r["_file"] = os.path.basename(f)
            rs.append(r)
    planned = [r for r in rs if "plan" in r]
    direct = [r for r in rs if r.get("arm", "").startswith("direct")]
    print("| run | setup | s/step (median, steps 3+) | tokens/s | loss step 1 -> 12 | frozen bytes unchanged | "
          "adapters moved | load s |")
    print("|---|---|---|---|---|---|---|---|")
    for r in planned + direct:
        s, m, c = r["setup"], r["measured"], r["correctness"]
        label = (f"{s['expert_residency']}, {s['expert_kernel']}" + (", NF4 attn" if s.get("attn_4bit") else "")
                 + ("" if s.get("pin", True) or s["expert_residency"] == "device" else ", pageable"))
        name = ("direct (no planner) " if r in direct else "") + r["run_id"].split("-t")[0].split("-0924-")[0]
        print(f"| `{r['run_id'][:60]}` | {label} | {m['s_per_step_median']:.2f} | {m['tokens_per_s']:.0f} | "
              f"{c['losses'][0]:.4f} -> {c['losses'][-1]:.4f} | {c['frozen_expert_bytes_unchanged']} | "
              f"{c['adapters_moved']} | {m.get('load_seconds', 0):.0f} |")
    print()
    print("| run | allocator est / meas (GiB) | driver est / meas (GiB) | host est / meas (GiB) | measured context, "
          "reserve slack | link H2D GB/s | overhead basis in plan |")
    print("|---|---|---|---|---|---|---|")
    for r in planned:
        cmp, m = r["comparison"], r["measured"]
        basis = {ln["name"].split(" ")[0]: ln["basis"] for ln in r["plan"]["selected"]["lines"]
                 if ln["name"].startswith(("CUDA", "allocator"))}
        slack = (m["device_reserved_peak_bytes"] - m["device_peak_bytes"]) / m["device_peak_bytes"]
        link = m.get("link_h2d_gbps")
        print(f"| `{r['run_id'][:60]}` | {gib(cmp['device_allocator']['estimated'])} / "
              f"{gib(cmp['device_allocator']['measured'])} | {gib(cmp['device_driver']['estimated'])} / "
              f"{gib(cmp['device_driver']['measured'])} | {gib(cmp['host']['estimated'])} / {gib(cmp['host']['measured'])} | "
              f"{gib(m.get('cuda_context_bytes'))}, {slack:.1%} | {'n/a' if link is None else f'{link:.2f}'} | {basis} |")
    bounds = [(r["run_id"], r["plan"]["selected"].get("bounds"), r["measured"]["s_per_step_median"]) for r in planned
              if r["plan"]["selected"].get("bounds")]
    if bounds:
        print()
        print("| run | transfer lower bound s/step (basis) | measured s/step | bound holds |")
        print("|---|---|---|---|")
        for rid, b, meas in bounds:
            print(f"| `{rid[:60]}` | {b['s_per_step_lower_bound']:.2f} ({b['link_gbps_basis']}, {b['link_gbps']:.2f} GB/s) | "
                  f"{meas:.2f} | {meas >= b['s_per_step_lower_bound']} |")
    print()
    print("Provenance per run (commit, dirty):")
    for r in planned + direct:
        src = {k: (v.get("commit", "")[:10] or "untracked", v.get("dirty")) for k, v in
               r["provenance"]["sources"].items()}
        print(f"- `{r['run_id']}`: {src}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "receipts")
