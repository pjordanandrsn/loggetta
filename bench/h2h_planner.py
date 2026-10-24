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
``accelerate estimate-memory MODEL --library_name transformers --dtypes float16 int4``, run as
``python -m accelerate.commands.accelerate_cli`` (what the ``accelerate`` entry point calls) from this interpreter, so the
CLI is the Accelerate the version check read. Its argv, version and full stdout are kept. It sizes the weights from the model's config (at the Hub's current revision; the CLI takes no revision)
and prints a full-precision Adam training figure; it does not model QLoRA, LoRA, activations or offload, so it is a
reference column, never like-for-like. Its sizes are read as printed, in its own 1024-based units.

Nothing here changes a plan, a constant or a band. Writes ``OUT_DIR/h2h-planner.json`` and ``OUT_DIR/accelerate.json``.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import importlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from urllib.parse import urlsplit

HERE = os.path.dirname(os.path.abspath(__file__))
#: the releases this row is registered at (PREREG "The subject"), always checked; each must be an index wheel or a git
#: install at exactly its tag's commit, never a checkout, so a development tree that reports the same version string can
#: never stand in for the release
REGISTERED = {"loggetta": "0.7.0", "experts4bit-qlora": "0.58.0", "grouped-nf4-gemm": "0.51.0", "accelerate": "1.15.0"}
#: each registered dist's own repository; a git install counts only from here, and its tag is read only from here (never
#: from the URL the install recorded, which whoever made the install chose, tag and all)
CANONICAL = {"loggetta": "github.com/pjordanandrsn/loggetta", "experts4bit-qlora": "github.com/pjordanandrsn/experts4bit-qlora",
             "grouped-nf4-gemm": "github.com/pjordanandrsn/grouped-nf4-gemm", "accelerate": "github.com/huggingface/accelerate"}
UNITS = {"bytes": 1, "KB": 1 << 10, "MB": 1 << 20, "GB": 1 << 30, "TB": 1 << 40}
ROW = re.compile(r"^\W*(float32|float16|int8|int4)\W+([\d.]+ \w+)\W+([\d.]+ \w+)\W+([\d.]+ \w+|N/A)\W*$")


class Refused(Exception):
    pass


@functools.lru_cache(maxsize=1)
def _release_proof():
    """bench/release_proof.py, loaded once, so its ReleaseProofError is one class."""
    spec = importlib.util.spec_from_file_location("release_proof", os.path.join(HERE, "release_proof.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def prove_release(dist, version):
    """``release_proof.prove``: the installed index wheel's bytes against the wheel PyPI publishes for ``version``."""
    return _release_proof().prove(dist, version)


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


def accelerate_argv(model, python=sys.executable):
    """``accelerate estimate-memory`` as the ``accelerate`` entry point runs it (``accelerate_cli:main``), but from this
    interpreter, so the CLI that runs is the dist :func:`check_versions` read, never another env's ``accelerate`` on PATH."""
    return [python, "-m", "accelerate.commands.accelerate_cli", "estimate-memory", model,
            "--library_name", "transformers", "--dtypes", "float16", "int4"]


def run_accelerate(model, version, python=sys.executable, run=subprocess.run):
    """One estimate-memory run. ``version`` is the one check_versions read for this interpreter's dist (the 1.15.0 CLI
    has no ``--version``)."""
    argv = accelerate_argv(model, python)
    r = run(argv, capture_output=True, text=True)
    if r.returncode != 0:
        raise Refused(f"{' '.join(argv)} exited {r.returncode}: {r.stderr.strip()[-300:]}")
    return {"argv": argv, "version": version, "stdout": r.stdout, "sizes": parse_accelerate(r.stdout)}


def installed(dist):
    from importlib import metadata
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def _direct_url(dist):
    from importlib import metadata
    raw = metadata.distribution(dist).read_text("direct_url.json")
    return json.loads(raw) if raw else None


def installed_from(dist):
    """How ``dist`` was installed: "index" for a wheel from a package index (no ``direct_url.json``), else the
    direct-URL kind ("editable", "local file", "vcs"). Only an index wheel, or a VCS install at the release's tag
    (:func:`vcs_at_tag`), is the release."""
    from importlib import metadata
    try:
        d = _direct_url(dist)
    except metadata.PackageNotFoundError:
        return None
    if d is None:
        return "index"
    if (d.get("dir_info") or {}).get("editable"):
        return "editable"
    return "vcs" if "vcs_info" in d else "local file"


def tag_commit(url, version):
    """The commit tag ``v<version>`` names at ``url`` (the peeled commit of an annotated tag), or None."""
    r = subprocess.run(["git", "ls-remote", url, f"refs/tags/v{version}", f"refs/tags/v{version}^{{}}"],
                       capture_output=True, text=True)
    refs = dict(reversed(line.split("\t")) for line in r.stdout.splitlines() if "\t" in line)
    return refs.get(f"refs/tags/v{version}^{{}}") or refs.get(f"refs/tags/v{version}")


def repo_of(url):
    """``host/owner/name`` for an https or ssh git URL (user, port, a trailing "/" and ".git" dropped); None for any
    other scheme, file:// included."""
    u = urlsplit(url or "")
    if u.scheme not in ("https", "ssh", "git+https", "git+ssh") or not u.hostname:
        return None
    path = u.path.rstrip("/")
    path = path[:-4] if path.endswith(".git") else path
    return f"{u.hostname.lower()}{path}"


def vcs_at_tag(dist, version, direct_url=_direct_url, tag_of=tag_commit):
    """(commit, None) when ``dist`` was installed from git, from its :data:`CANONICAL` repository, at exactly the commit
    that repository's tag ``v<version>`` names; else (commit or None, why not). The tag is read from the canonical
    repository, never from the URL the install recorded."""
    d = direct_url(dist) or {}
    commit, url = (d.get("vcs_info") or {}).get("commit_id"), d.get("url")
    if not commit or not url:
        return None, "records no commit"
    canonical = CANONICAL.get(dist)
    if canonical is None:
        return commit, "has no canonical repository registered"
    if repo_of(url) != canonical:
        return commit, f"is from {url}, not {canonical}"
    tag = tag_of(f"https://{canonical}", version)
    if tag is None:
        return commit, f"has no tag v{version}"
    if tag != commit:
        return commit, f"is at {commit[:12]}, not tag v{version}'s {tag[:12]}"
    return commit, None


def check_versions(expect=None, version_of=installed, source_of=installed_from, at_tag=vcs_at_tag, proof=prove_release):
    """Refused unless every registered release (``expect``, default :data:`REGISTERED`) is installed at exactly its
    version AND as that release: an index wheel whose installed bytes ``release_proof`` matches to the wheel PyPI
    publishes (no ``direct_url.json`` only proves SOME index served it), or a git install at exactly the commit its
    canonical repository's tag names. A missing
    release, another version, or a checkout reporting the same version string reads "not read today"; whatever happens to
    be installed never stands in."""
    expect = REGISTERED if expect is None else expect
    got = {d: (version_of(d), source_of(d)) for d in expect}
    wrong = [f"{d} wants {v}, installed {got[d][0]}" for d, v in sorted(expect.items()) if got[d][0] != v]
    commits, proofs = {}, {}
    for d, v in sorted(expect.items()):
        if got[d][0] != v:
            continue
        if got[d][1] == "index":
            err = _release_proof().ReleaseProofError
            try:
                proofs[d] = proof(d, v)
            except err as e:
                wrong.append(f"{d} {v} is an index install whose bytes are not the published release: {e}")
            continue
        if got[d][1] != "vcs":
            wrong.append(f"{d} {v} is installed from a {got[d][1]} install, not a release")
            continue
        commit, why = at_tag(d, v)
        if why:
            wrong.append(f"{d} {v} is a git install that {why}, not a release")
        commits[d] = commit
    if wrong:
        raise Refused("not read today: " + "; ".join(wrong))
    return {d: {"version": v, "installed_from": src, **({"commit": commits[d]} if d in commits else {}),
                **({"release_proof": proofs[d]} if d in proofs else {})} for d, (v, src) in got.items()}


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


def build(split, e4b, accelerate_version, reader=None, run=subprocess.run):
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
    acc = {m: run_accelerate(m, accelerate_version, run=run) for m in models}
    return ({"split": f"SPLIT-v{split['version']}.json", "pinned_at": split["pinned_at"], "rows": rows}, acc)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("split")
    ap.add_argument("--e4b", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    try:
        versions = check_versions()
        doc, acc = build(json.load(open(a.split)), a.e4b, versions["accelerate"]["version"])
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
