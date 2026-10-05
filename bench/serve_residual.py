"""Where does a served model's allocator peak exceed the estimate? Build ``serve_paged``'s engine with the caching
allocator's history recording (Python stacks), generate, then replay the trace: the allocations live at the
generation peak, grouped by the first frame inside experts4bit-qlora / grouped-nf4-gemm / transformers / torch.

    python bench/serve_residual.py allenai/OLMoE-1B-7B-0924 --arena ARENA --calib CALIB --out runs/residual
"""
import argparse
import json
import os
import pickle
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
GiB, MiB = 1 << 30, 1 << 20
OURS = ("experts4bit_qlora", "kernel/", "gnf4", "transformers", "torch/nn", "torch/_")


def key_of(frames):
    """The innermost frame that names a mechanism (skipping torch internals and this script)."""
    for f in frames or ():
        fn = f.get("filename", "")
        if "serve_residual" in fn:
            continue
        if any(s in fn for s in ("experts4bit_qlora", "/kernel/", "gnf4_native", "transformers/")):
            short = fn.split("site-packages/")[-1].split("loggetta-ws/")[-1]
            return f"{short}:{f.get('line')} {f.get('name')}"
    for f in frames or ():
        fn = f.get("filename", "")
        if "serve_residual" not in fn:
            return f"{fn.split('site-packages/')[-1]}:{f.get('line')} {f.get('name')}"
    return "?"


def replay(trace, start=0):
    """Live allocations at the peak of trace[start:], given everything live at ``start``."""
    live, total, peak, at_peak = {}, 0, 0, {}
    for i, ev in enumerate(trace):
        a, addr, size = ev.get("action"), ev.get("addr"), ev.get("size", 0)
        if a == "alloc":
            live[addr] = (size, ev.get("frames"))
            total += size
        elif a in ("free_completed",) and addr in live:
            total -= live.pop(addr)[0]
        if i >= start and total > peak:
            peak, at_peak = total, dict(live)
    return peak, at_peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--arena", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--context", type=int, default=512)
    ap.add_argument("--prompt-tokens", type=int, default=128)
    ap.add_argument("--new-tokens", type=int, default=16)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--placement", default="all-vram")
    ap.add_argument("--vram-gb", type=float, default=1.2)
    ap.add_argument("--dram-gb", type=float, default=6.0)
    ap.add_argument("--hot-rows", type=int, default=64)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch

    from experts4bit_qlora.arch.topology import describe_moe
    from experts4bit_qlora.serve_recipe import ServeSetup, estimate_serve_footprint

    setup = ServeSetup(max_seqs=a.concurrency, max_tokens_per_seq=a.context, graphs=False, prefill_graph="0",
                       placement=a.placement, vram_gb=a.vram_gb, dram_gb=a.dram_gb, hot_rows=a.hot_rows)
    est = estimate_serve_footprint(describe_moe(a.model), setup)
    est_dev = {i.name: i.bytes for i in est.items if i.where == "device"}
    os.environ.update({**setup.to_env(), "E4B_PAGED_MODEL": a.model, "E4B_PAGED_ARENA": a.arena, "E4B_PAGED_CALIB": a.calib,
                       "E4B_PAGED_TORCH_THREADS": "2"})
    torch.cuda.set_device(0)
    torch.zeros(1, device="cuda")
    torch.cuda.memory._record_memory_history(max_entries=2_000_000, stacks="python")
    from experts4bit_qlora.serve_paged import PagedServeConfig, build_engine

    parts = build_engine(PagedServeConfig.from_env())
    torch.cuda.synchronize()
    load_peak = torch.cuda.max_memory_allocated()
    mark = len(torch.cuda.memory._snapshot()["device_traces"][0])
    torch.cuda.reset_peak_memory_stats()
    g = torch.Generator().manual_seed(0)
    vocab = int(getattr(parts.tokenizer, "vocab_size", 0) or 32000)
    for _ in range(a.concurrency):
        parts.scheduler.add_request(torch.randint(10, vocab - 10, (a.prompt_tokens,), generator=g).tolist(),
                                    max_new_tokens=a.new_tokens)
    t0 = time.time()
    parts.scheduler.run_until_idle()
    torch.cuda.synchronize()
    gen_s = time.time() - t0
    run_peak = torch.cuda.max_memory_allocated()
    snap = torch.cuda.memory._snapshot()
    torch.cuda.memory._record_memory_history(enabled=None)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "snapshot.pickle"), "wb") as f:
        pickle.dump(snap, f)
    trace = snap["device_traces"][0]
    peak, at_peak = replay(trace, start=mark)
    groups = defaultdict(lambda: [0, 0])
    for size, frames in at_peak.values():
        k = key_of(frames)
        groups[k][0] += size
        groups[k][1] += 1
    top = sorted(groups.items(), key=lambda kv: -kv[1][0])
    est_alloc = sum(est_dev.values())
    print(f"estimate {est_alloc / GiB:.3f} GiB | load peak {load_peak / GiB:.3f} | run peak {run_peak / GiB:.3f} | "
          f"replayed peak {peak / GiB:.3f} | residual {(run_peak - est_alloc) / MiB:.0f} MiB | generate {gen_s:.1f}s")
    print("live at the generation peak, by allocating frame:")
    for k, (b, n) in top[:30]:
        print(f"  {b / MiB:9.1f} MiB  x{n:<5d} {k}")
    json.dump({"schema": "serve-residual/1", "model": a.model, "setup": setup.to_dict(), "estimate_device": est_dev,
               "load_peak_bytes": load_peak, "run_peak_bytes": run_peak, "replayed_peak_bytes": peak,
               "trace_events": len(trace), "mark": mark,
               "groups": [{"frame": k, "bytes": b, "count": n} for k, (b, n) in top]},
              open(os.path.join(a.out, "residual.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
