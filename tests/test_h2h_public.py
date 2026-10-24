"""bench/h2h-planner/project_public.py: the planner head-to-head's public copies differ from its raw outputs only in their local paths.

Every value that is not a path is byte-identical between raw and public (so the projection cannot edit a number), and no absolute
path is left. The raw pair lives in the private receipt store. With it to hand (``H2H_RAW_DIR``) the committed copies are re-derived
from it and checked against the sha256 the evidence names; without it, the committed copies are checked as they stand.
"""
import hashlib
import importlib.util
import json
import os
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
EVID = ROOT / "evidence" / "2026-10-24-h2h-planner"


def _tool():
    spec = importlib.util.spec_from_file_location("project_public", ROOT / "bench" / "h2h-planner" / "project_public.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _leaves(obj, where=()):
    if isinstance(obj, dict):
        yield where, "dict", tuple(sorted(obj))
        for k in sorted(obj):
            yield from _leaves(obj[k], where + (k,))
    elif isinstance(obj, list):
        yield where, "list", len(obj)
        for i, v in enumerate(obj):
            yield from _leaves(v, where + (i,))
    else:
        yield where, type(obj).__name__, obj


def _only_paths_differ(t, raw, public, python, pycache):
    """Same structure, every non-string leaf equal in type and value, and every string either equal or the raw string projected."""
    a, b = list(_leaves(raw)), list(_leaves(public))
    assert [(w, k) for w, k, _ in a] == [(w, k) for w, k, _ in b], "the projection changed the structure"
    changed = 0
    for (w, kind, x), (_, _, y) in zip(a, b):
        if kind != "str":
            assert x == y, f"{w}: {x!r} -> {y!r}, a value that is not a path changed"
        elif x != y:
            assert y == t.project_str(x, python, pycache) and re.search(r"<(python|site-packages|pycache)>", y), (w, x, y)
            changed += 1
    return changed


#: synthetic host paths, assembled at run time so the source carries no literal home or temp path
_U = "/".join(("", "Users", "u", "v"))
_T = "/".join(("", "var", "folders", "ab", "T", "release-proof-pycache-x1"))
_H = "/".join(("", "home", "someone"))
RAW = {"versions": {"loggetta": {"version": "0.7.0", "installed_from": "index", "release_proof": {
           "pycache_prefix": _T, "imports": {"loggetta": _U + "/lib/python3.14/site-packages/loggetta/__init__.py"},
           "files": 30, "sha256": "e" * 64, "record": "RECORD equal"}}},
       "rows": [{"id": "ho3-a4000-1/ernie/host/1024", "now": {"host": {"estimated": 15451815936, "measured": 15775776768}}, "now_over": {"host": 1.021},
                 "note": "a plain string, no path", "flag": True, "none": None}]}
ACC = {"m": {"argv": [_U + "/bin/python", "-m", "accelerate.commands.accelerate_cli", "estimate-memory", "m"], "version": "1.15.0",
             "stdout": "| float16 | 1.48 GB | 40.65 GB |", "sizes": {"float16": {"total_bytes": 43647273370}}}}


def test_the_projection_changes_only_paths():
    t = _tool()
    python, pycache = t.tokens_of(RAW, ACC)
    assert (python, pycache) == (_U + "/bin/python", _T)
    pr, pa = t.project(RAW, python, pycache), t.project(ACC, python, pycache)
    assert _only_paths_differ(t, RAW, pr, python, pycache) == 2 and _only_paths_differ(t, ACC, pa, python, pycache) == 1
    assert pr["versions"]["loggetta"]["release_proof"]["imports"]["loggetta"] == "<site-packages>/loggetta/__init__.py"
    assert pr["versions"]["loggetta"]["release_proof"]["pycache_prefix"] == "<pycache>" and pa["m"]["argv"][0] == "<python>"
    assert t.leftover_paths(pr) == [] and t.leftover_paths(pa) == []


def test_a_path_the_map_does_not_cover_refuses(tmp_path):
    t = _tool()
    raw = json.loads(json.dumps(RAW))
    raw["rows"][0]["note"] = f"read from {_H}/split.json"
    for name, obj in (("h2h-planner.json", raw), ("accelerate.json", ACC)):
        (tmp_path / name).write_text(json.dumps(obj))
    with pytest.raises(t.ProjectionError, match="absolute paths left at"):
        t.project_dir(str(tmp_path))
    assert t.main([str(tmp_path), str(tmp_path / "out")]) == 1


def test_two_interpreters_or_prefixes_refuse():
    t = _tool()
    two = {"a": {"argv": ["/x/python"]}, "b": {"argv": ["/y/python"]}}
    with pytest.raises(t.ProjectionError, match="2 interpreters"):
        t.tokens_of(RAW, two)
    with pytest.raises(t.ProjectionError, match="0 bytecode prefixes"):
        t.tokens_of({"versions": {}}, ACC)


def test_the_committed_copies_hold_no_path_and_are_the_projections_fixed_point():
    t = _tool()
    pub = {name: json.loads((EVID / name).read_text()) for _, name in t.FILES}
    for name, obj in pub.items():
        assert t.leftover_paths(obj) == [], name
        assert (EVID / name).read_text() == t.dumps(obj), f"{name} is not in the driver's layout"
    python, pycache = t.tokens_of(pub["h2h-planner.public.json"], pub["accelerate.public.json"])
    assert (python, pycache) == ("<python>", "<pycache>")
    for name, obj in pub.items():
        assert t.project(obj, python, pycache) == obj, f"projecting {name} again changes it"


def test_the_evidence_names_the_raw_pair():
    names = {ln.split()[1]: ln.split()[0] for ln in (EVID / "STORE-ONLY.sha256").read_text().splitlines() if ln and not ln.startswith("#")}
    assert set(names) == {"h2h-planner.json", "accelerate.json", "run.log"}
    readme = (EVID / "README.md").read_text()
    for name in ("h2h-planner.json", "accelerate.json"):
        assert f"`{name}`: `{names[name]}`" in readme


@pytest.mark.skipif(not os.environ.get("H2H_RAW_DIR"), reason="the raw pair is store-only: set H2H_RAW_DIR to its directory")
def test_the_committed_copies_are_the_raw_pairs_projection():
    t = _tool()
    raw_dir = pathlib.Path(os.environ["H2H_RAW_DIR"])
    names = {ln.split()[1]: ln.split()[0] for ln in (EVID / "STORE-ONLY.sha256").read_text().splitlines() if ln and not ln.startswith("#")}
    for name in ("h2h-planner.json", "accelerate.json"):
        assert hashlib.sha256((raw_dir / name).read_bytes()).hexdigest() == names[name], f"{name} is not the raw file the evidence names"
    files = t.project_dir(str(raw_dir))
    for name, text in files.items():
        assert (EVID / name).read_text() == text, f"{name} is not the projection of the raw pair"
    raw = {n: json.loads((raw_dir / n).read_text()) for n, _ in t.FILES}
    python, pycache = t.tokens_of(raw["h2h-planner.json"], raw["accelerate.json"])
    for n, pub in t.FILES:
        assert _only_paths_differ(t, raw[n], json.loads(files[pub]), python, pycache) > 0
