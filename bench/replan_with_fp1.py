"""Before/after: plan Qwen3-30B-A3B with the A2000 receipts only, then with lane FP1's RTX 5090 receipts added.

Two hardware profiles: the seat's RTX A2000 (saved by `inspect --json`) and the RTX 5090 FP1 ran on (rebuilt from the box's
forensics line; every fact labelled as coming from that receipt). The workload is FP1's: seq 512 x micro-batch 2.

    python bench/replan_with_fp1.py --a2000 evidence/2026-10-04-rtx-a2000/hardware-profile.json \
        --old evidence/2026-10-04-rtx-a2000 --fp1 evidence/2026-10-04-fp1-rtx5090 --out evidence/fp1-replan.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def rtx5090_profile(fp1_dir):
    from loggetta.hardware import GPU, Fact, HardwareProfile, Host

    obs = json.load(open(sorted(os.path.join(fp1_dir, f) for f in os.listdir(fp1_dir))[0]))
    g = obs["hardware"]["gpu"]
    src = f"receipt {obs['run_id']} forensics"
    r = lambda v: Fact(v, "reported", src)  # noqa: E731
    mib = int(g["memory_total"].split()[0])
    gpu = GPU(index=0, vendor="nvidia", name=g["name"], uuid=None, compute_capability=r((12, 0)),
              memory_total=r(mib << 20), memory_free=Fact(mib << 20, "inferred", "a rented box: the whole card free"),
              driver=r(g["driver"]), pcie_gen_max=r(4), pcie_width_max=r(16), pcie_gen_current=r(4),
              pcie_width_current=r(16))
    host = Host(cpu_model=r("AMD EPYC 7B13"), cpus=Fact(16, "inferred", "a typical rental slice"),
                memory_total=r(2101193408 * 1024), memory_available=Fact(64 << 30, "user", "planning as if 64 GiB were ours"),
                memory_limit=Fact(None, "unknown"))
    return HardwareProfile(gpus=(gpu,), host=host, platform="Linux x86_64", notes=(f"rebuilt from {src}",))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a2000", required=True)
    ap.add_argument("--old", required=True, help="the A2000 receipts")
    ap.add_argument("--fp1", required=True, help="the imported FP1 observations")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from loggetta import Constraints, Workload, describe_model, plan
    from loggetta.hardware import HardwareProfile
    from loggetta.execution import load_observations

    old = load_observations(a.old)
    new = old + load_observations(a.fp1)
    topo = describe_model("Qwen/Qwen3-30B-A3B")
    w = Workload(seq_len=512, micro_batch=2, steps=12)
    profiles = {"RTX A2000 (seat)": HardwareProfile.from_dict(json.load(open(a.a2000)), origin=a.a2000),
                "RTX 5090 (FP1's box)": rtx5090_profile(a.fp1)}
    rows = []
    for hw_name, hw in profiles.items():
        for residency in ("device", "host"):
            for label, obs in (("before", old), ("after", new)):
                c = Constraints(expert_residency=(residency,), fixed={"attn_4bit": False, "pin": True},
                                vram_budget=hw.gpu(0).memory_total.value)
                p = plan(topo, hw, w, c, observations=obs)
                cand = p.selected or min(p.alternatives, key=lambda x: x.device_bytes)
                lines = {ln.name: ln for ln in cand.lines}
                res = lines["allocator reserve (cached, unallocated blocks)"]
                ctx = lines["CUDA context + library workspaces"]
                rows.append({"hardware": hw_name, "residency": residency, "observations": label, "status": p.status,
                             "device_gib": round(cand.device_bytes / 2**30, 2), "reserve_gib": round(res.bytes / 2**30, 2),
                             "reserve_basis": res.basis, "reserve_detail": res.detail,
                             "context_gib": round(ctx.bytes / 2**30, 2), "context_basis": ctx.basis,
                             "transfer_bound_s": round(cand.bounds["s_per_step_lower_bound"], 2) if cand.bounds else None})
                print(f"{hw_name:22s} {residency:6s} {label:6s} {p.status:9s} device {rows[-1]['device_gib']:6.2f} GiB  "
                      f"reserve {rows[-1]['reserve_gib']:5.2f} [{res.basis}]  context {rows[-1]['context_gib']:4.2f} "
                      f"[{ctx.basis}]  bound {rows[-1]['transfer_bound_s']}")
    json.dump({"schema": "replan-comparison/1", "model": "Qwen/Qwen3-30B-A3B", "workload": "seq 512 x mb 2",
               "rows": rows}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
