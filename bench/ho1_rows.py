"""HO1 row set: experts4bit-qlora training arms on cards Loggetta was never calibrated on, fixed before any plan.

    python bench/ho1_rows.py --e4b E4B_CHECKOUT --commit SHA --json evidence/<dir>/rows.json

Reads the arm receipts committed at ``--commit`` (``git archive``; the working tree is not read). An arm is a row when
its framework is ``e4b``, its GPU is an RTX 4090, RTX 5090 or H100, it records ``peak_vram_gb``, its status is ``ok``,
and it records model, ``seq``, ``micro_batch``, ``arm``, ``attn_4bit`` and ``offload``. Rows carry the setup Loggetta
needs to re-plan and the two measured peaks (allocator, and the driver's from the arm's nvidia-smi sidecar). They carry
no plan. Each row is marked ``primary`` or excluded from the primary set with the reasons (see HO1-PREREG.md).
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import subprocess
import tarfile

CARDS = ("RTX 4090", "RTX 5090", "H100")
DIRS = ("bench/h2h-2026-10-02", "bench/p67")
MiB = 1 << 20


def _env_overrides(obj, path=""):
    """Every non-null ``*_env`` value inside an A/B record: an override the run set on purpose."""
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else k
            if k.endswith("_env") and v not in (None, "", "0") and not isinstance(v, (dict, list)):
                out.append(f"{p}={v}")
            out += _env_overrides(v, p)
    return out


def _driver_peak(arm_json_path):
    """Max ``memory.used`` (MiB) over the arm's 1 Hz nvidia-smi series, or None when the sidecar is absent."""
    d, stem = os.path.split(arm_json_path)
    side = os.path.join(d, "vram_" + stem[: -len(".json")] + ".txt")
    if not os.path.exists(side):
        return None, None
    vals = []
    for ln in open(side):
        parts = [p.strip() for p in ln.replace(",", " ").split()]
        if len(parts) >= 2 and parts[1].isdigit():
            vals.append(int(parts[1]))
    return (max(vals) * MiB if vals else None), len(vals)


def rows(root):
    out = []
    for f in sorted(glob.glob(os.path.join(root, "**", "*.json"), recursive=True)):
        try:
            d = json.load(open(f))
        except (OSError, ValueError):
            continue
        if not isinstance(d, dict) or d.get("framework") != "e4b" or "peak_vram_gb" not in d:
            continue
        env = d.get("env") or {}
        gpu = env.get("gpu") or ""
        if not any(c in gpu for c in CARDS) or d.get("status") != "ok":
            continue
        need = ("model", "seq", "micro_batch", "arm", "attn_4bit", "offload")
        if any(d.get(k) is None for k in need):
            continue
        why = []
        if d["arm"] not in ("fused", "reference"):
            why.append(f"arm {d['arm']} (not a kernel the planner prices)")
        route = (d.get("route_ab") or {}).get("gnf4_train_gemm")
        if route not in (None, "fused"):
            why.append(f"gnf4_train_gemm={route}")
        if (d.get("ckpt_offload_layers") or 0) != 0:
            why.append(f"checkpoint offload of {d['ckpt_offload_layers']} layers (not priced)")
        if (d.get("chunked_lm_loss") or {}).get("env") not in (None,):
            why.append(f"E4B_CHUNKED_LM_LOSS={d['chunked_lm_loss']['env']}")
        for k in ("keep_ab", "lean_ab", "prebind_ab", "reuse_ab", "rms_ab", "sync_ab", "tile_ab"):
            why += [f"{k}:{o}" for o in _env_overrides(d.get(k))]
        if d.get("absmax_dq"):
            why.append("absmax_dq (not a planner setup field)")
        if d.get("adapter_dtype") not in ("fp32", "native"):
            why.append(f"adapter_dtype={d.get('adapter_dtype')}")
        drv, n = _driver_peak(f)
        if drv is None:
            why.append("no driver sidecar")
        out.append({
            "arm_receipt": os.path.relpath(f, root), "gpu": gpu, "driver": (env.get("host") or {}).get("driver_version"),
            "host_mem_gib": (env.get("host") or {}).get("host_mem_gib"),
            "e4b_version": env.get("experts4bit-qlora"), "gnf4_version": env.get("grouped-nf4-gemm"),
            "torch": env.get("torch"), "model": d["model"], "revision": d.get("revision"),
            "seq": d["seq"], "micro_batch": d["micro_batch"], "accum": d.get("accum"), "optimizer": d.get("optimizer"),
            "r": d.get("r"), "alpha": d.get("alpha"),
            "setup": {"expert_kernel": "grouped_nf4" if d["arm"] == "fused" else "reference",
                      "expert_residency": "host" if d["offload"] else "device",
                      "attn_4bit": bool(d["attn_4bit"]),
                      "adapter_dtype": "fp32" if d.get("adapter_dtype") == "fp32" else "bf16"},
            "allocated_peak_bytes": round(d["peak_vram_gb"] * 1e9),
            "driver_peak_bytes": drv, "driver_samples": n,
            "primary": not why, "excluded_because": why,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e4b", required=True)
    ap.add_argument("--commit", required=True)
    ap.add_argument("--json", required=True)
    a = ap.parse_args()
    tmp = os.path.join(os.path.dirname(os.path.abspath(a.json)), ".e4b-archive")
    os.makedirs(tmp, exist_ok=True)
    blob = subprocess.run(["git", "-C", a.e4b, "archive", a.commit, *DIRS], check=True, capture_output=True).stdout
    tarfile.open(fileobj=io.BytesIO(blob)).extractall(tmp, filter="data")
    rs = rows(tmp)
    json.dump({"e4b_commit": a.commit, "dirs": DIRS, "rows": rs}, open(a.json, "w"), indent=1, sort_keys=True)
    prim = [r for r in rs if r["primary"]]
    print(f"{len(rs)} rows, {len(prim)} primary; by card:",
          {c: sum(c in r["gpu"] for r in prim) for c in CARDS})


if __name__ == "__main__":
    main()
