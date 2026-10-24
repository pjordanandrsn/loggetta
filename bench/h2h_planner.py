"""The planner head-to-head at a release (bench/h2h-planner/PREREG-h2h-planner.md): this checkout's planner against the
measured held-out rows, beside one published memory estimator.

    python bench/h2h_planner.py evidence/planner-split/SPLIT-v3.json --e4b E4B_CHECKOUT --out OUT_DIR

For every held-out row of the split:
- its receipt is read from the cited repository AT the split's ``pinned_at`` (``git show``), and refused unless its
  sha256 equals the row's;
- the row is planned again through ``bench/insample_replan.py``'s own functions (``_plan``; the dense rows through
  ``dense.describe`` from the lane's committed config, as its ``dense`` mode does), with no receipts on file;
- the plan's estimates are graded against the receipt's own measured peaks with ``execution.compare``, the function a
  receipt's ``comparison`` comes from, and set beside the estimates the receipt recorded when it ran.

Then, once per model, Hugging Face Accelerate's own CLI, exactly as a user runs it:
``accelerate estimate-memory MODEL --library_name transformers --dtypes float16 int4``. Its command, version and full
stdout are kept. It sizes the weights from the model's config (at the Hub's current revision; the CLI takes no revision)
and prints a full-precision Adam training figure; it does not model QLoRA, LoRA, activations or offload, so it is a
reference column, never like-for-like. Its sizes are read as printed, in its own 1024-based units.

Nothing here changes a plan, a constant or a band. Writes ``OUT_DIR/h2h-planner.json`` and ``OUT_DIR/accelerate.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
#: the releases this row is registered at (PREREG "The subject"), always checked; each must be an index wheel, never a
#: checkout, so a development tree that reports the same version string can never stand in for the release
REGISTERED = {"loggetta": "0.7.0", "experts4bit-qlora": "0.58.0", "grouped-nf4-gemm": "0.51.0", "accelerate": "1.15.0"}
UNITS = {"bytes": 1, "KB": 1 << 10, "MB": 1 << 20, "GB": 1 << 30, "TB": 1 << 40}
ROW = re.compile(r"^\W*(float32|float16|int8|int4)\W+([\d.]+ \w+)\W+([\d.]+ \w+)\W+([\d.]+ \w+|N/A)\W*$")


class Refused(Exception):
    pass


def _replan_module():
    spec = importlib.util.spec_from_file_location("insample_replan", os.path.join(HERE, "insample_replan.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def git_show(repo_dir, commit, path):
    r = subprocess.run(["git", "-C", repo_dir, "show", f"{commit}:{path}"], capture_output=True)
    if r.returncode != 0:
        raise Refused(f"{path} is not readable at {commit[:12]}: {r.stderr.decode(errors='replace').strip()[:200]}")
    return r.stdout


def read_row(row, pinned, reader):
    raw = reader(pinned[row["repo"]], row["file"])
    if hashlib.sha256(raw).hexdigest() != row["sha256"]:
        raise Refused(f"{row['id']}: {row['file']} at {pinned[row['repo']][:12]} is not the split's sha256")
    return json.loads(raw)


def size(text):
    """A size as Accelerate prints it ('40.65 GB'), in bytes; None for 'N/A'."""
    if text == "N/A":
        return None
    n, unit = text.split()
    return round(float(n) * UNITS[unit])


def parse_accelerate(stdout):
    """{dtype: {largest_layer, total, training_adam}} from estimate-memory's table, each as printed and in bytes."""
    out = {}
    for line in stdout.splitlines():
        m = ROW.match(line.strip())
        if m:
            dt, largest, total, train = m.groups()
            out[dt] = {"printed": {"largest_layer": largest, "total": total, "training_adam": train},
                       "largest_layer_bytes": size(largest), "total_bytes": size(total), "training_adam_bytes": size(train)}
    if not out:
        raise Refused("accelerate estimate-memory printed no size table")
    return out


def run_accelerate(model, exe="accelerate"):
    cmd = [exe, "estimate-memory", model, "--library_name", "transformers", "--dtypes", "float16", "int4"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise Refused(f"{' '.join(cmd)} exited {r.returncode}: {r.stderr.strip()[-300:]}")
    ver = subprocess.run([exe, "--version"], capture_output=True, text=True).stdout.strip()
    return {"command": " ".join(cmd), "version": ver, "stdout": r.stdout, "sizes": parse_accelerate(r.stdout)}


def installed(dist):
    from importlib import metadata
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def installed_from(dist):
    """How ``dist`` was installed: "index" for a wheel from a package index (no ``direct_url.json``), else the
    direct-URL kind ("editable", "local file", "vcs"), which is a checkout, not a release."""
    from importlib import metadata
    try:
        raw = metadata.distribution(dist).read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        return None
    if not raw:
        return "index"
    d = json.loads(raw)
    if (d.get("dir_info") or {}).get("editable"):
        return "editable"
    return "vcs" if "vcs_info" in d else "local file"


def check_versions(expect=None, version_of=installed, source_of=installed_from):
    """Refused unless every registered release (``expect``, default :data:`REGISTERED`) is installed at exactly its
    version AND from a package index. A missing release, another version, or a checkout reporting the same version
    string reads "not read today"; whatever happens to be installed never stands in."""
    expect = REGISTERED if expect is None else expect
    got = {d: (version_of(d), source_of(d)) for d in expect}
    wrong = [f"{d} wants {v}, installed {got[d][0]}" for d, v in sorted(expect.items()) if got[d][0] != v]
    wrong += [f"{d} {v} is installed from a {got[d][1]} install, not a release" for d, v in sorted(expect.items())
              if got[d][0] == v and got[d][1] != "index"]
    if wrong:
        raise Refused("not read today: " + "; ".join(wrong))
    return {d: {"version": v, "installed_from": src} for d, (v, src) in got.items()}


def outcome(cmp, host_row):
    """The planner's own per-row rule (HO2's and HO3's reducers): COVERED when the plan total bounds the driver peak and
    the allocator estimate bounds the allocated peak, and, on a host-offload row, the host estimate bounds the host
    peak; otherwise each failed bound by name (UNDER, SHORT, HOST_UNDER). A missing term voids the row."""
    terms = ("device_driver", "device_allocator") + (("host",) if host_row else ())
    missing = [t for t in terms if not (isinstance((cmp or {}).get(t), dict) and isinstance(cmp[t].get("estimated"), int)
                                        and isinstance(cmp[t].get("measured"), int) and cmp[t]["estimated"] > 0)]
    if missing:
        return "VOID: no " + ", ".join(missing)
    failed = [name for t, name in (("device_driver", "UNDER"), ("device_allocator", "SHORT"), ("host", "HOST_UNDER"))
              if t in terms and cmp[t]["measured"] > cmp[t]["estimated"]]
    return "+".join(failed) or "COVERED"


def grade(cmp):
    """measured / estimated for each term the comparison carries."""
    return {t: (v["measured"] / v["estimated"] if v.get("estimated") and v.get("measured") else None)
            for t, v in cmp.items() if isinstance(v, dict) and "estimated" in v}


def replan_row(rec, ir, config_reader, topos):
    """Plan ``rec`` again through its own backend. A dense read (``backend`` "dense", DQ7's) is described from the lane's
    committed config, read at the split's pin, as ``insample_replan.py dense`` does; a MoE receipt from the Hub config at
    the receipt's recorded revision."""
    from loggetta.execution import compare

    if rec.get("backend") == "dense":
        import transformers as tr
        from loggetta.backends import dense as be
        model = os.path.basename(str(rec["model"]["model"]))
        if ("dense", model) not in topos:
            cfg = json.loads(config_reader(f"bench/dq7/configs/{model}.json"))
            c = tr.AutoConfig.for_model(**cfg)
            c._name_or_path = model
            topos[("dense", model)] = be.describe(c)
        topo = topos[("dense", model)]
    else:
        be = importlib.import_module("loggetta.backends.experts4bit")
        key = ("moe", rec["model"]["model"], rec["model"].get("revision"))
        if key not in topos:
            topos[key] = be.describe(key[1], revision=key[2]) if key[2] else be.describe(key[1])
        topo = topos[key]
    p = ir._plan(topo, be, rec)
    now = compare(p, rec["measured"]) if p.status == "feasible" else None
    then = rec.get("comparison") or {}
    host_row = (rec.get("setup") or {}).get("expert_residency") == "host"
    return {"status": p.status, "residency": (p.selected.setup or {}).get("expert_residency") if p.selected else None,
            "now": now, "then": then, "now_over": grade(now) if now else None, "then_over": grade(then),
            "now_outcome": outcome(now, host_row) if now else f"REFUSED: the plan is {p.status}",
            "then_outcome": outcome(then, host_row)}


def build(split, e4b, reader=None, accelerate="accelerate"):
    ir = _replan_module()
    reader = reader or (lambda c, p: git_show(e4b, c, p))
    pin = split["pinned_at"]["experts4bit-qlora"]
    rows, topos, models = [], {}, {}
    for row in split["held_out"]:
        rec = read_row(row, split["pinned_at"], reader)
        r = replan_row(rec, ir, lambda path: reader(pin, path), topos)
        rows.append({"id": row["id"], "cohort": row["cohort"], "box": row.get("box") or row["id"].split("/")[0],
                     "model": row["model"], "card": row["card"], "placement": row["placement"], "seq": row["seq"],
                     "file": row["file"], "sha256": row["sha256"], **r})
        models.setdefault(row["model"], None)
    acc = {m: run_accelerate(m, accelerate) for m in models}
    return ({"split": f"SPLIT-v{split['version']}.json", "pinned_at": split["pinned_at"], "rows": rows}, acc)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("split")
    ap.add_argument("--e4b", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--accelerate", default="accelerate", help="the accelerate CLI to run (default: the one on PATH)")
    a = ap.parse_args()
    try:
        versions = check_versions()
        doc, acc = build(json.load(open(a.split)), a.e4b, accelerate=a.accelerate)
    except Refused as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    os.makedirs(a.out, exist_ok=True)
    doc["versions"] = versions
    open(os.path.join(a.out, "h2h-planner.json"), "w").write(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n")
    open(os.path.join(a.out, "accelerate.json"), "w").write(json.dumps(acc, indent=1, sort_keys=True) + "\n")
    for r in doc["rows"]:
        print(r["id"], r["now_outcome"], {k: (round(v, 4) if v else v) for k, v in (r["now_over"] or {}).items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
