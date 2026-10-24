"""The planner head-to-head's public copies: its raw outputs with their local paths replaced by tokens, and nothing else changed.

    python bench/h2h-planner/project_public.py RAW_DIR OUT_DIR

``bench/h2h_planner.py`` writes ``h2h-planner.json`` and ``accelerate.json``, and both record absolute paths on the machine that ran
the read: the interpreter Accelerate ran under (its ``argv[0]``), each release proof's import origins (under the venv's
site-packages) and the bytecode prefix the proofs ran under. Those stay in the private receipt store. This writes
``h2h-planner.public.json`` and ``accelerate.public.json`` with exactly those paths replaced:

- the interpreter -> ``<python>``;
- a venv's ``.../site-packages`` -> ``<site-packages>`` (the rest of the path kept);
- the bytecode prefix -> ``<pycache>``.

Every other value is left byte-for-byte as the raw file has it, in the driver's own JSON layout, and the projection refuses when an
absolute path is left anywhere. ``tests/test_h2h_public.py`` holds both.
"""
from __future__ import annotations

import json
import os
import re
import sys

FILES = (("h2h-planner.json", "h2h-planner.public.json"), ("accelerate.json", "accelerate.public.json"))
SITE = re.compile(r"/(?:[^/\s\"']+/)*?site-packages(?=/|$)")
#: the roots an absolute path on a development host or a box starts with; any left after the projection refuses
ABSOLUTE = re.compile(r"(?:^|[\s\"'=(\[,:])/(?:private|Users|var|tmp|home|opt|root|usr|Library|Volumes|mnt|srv|nix|data|workspace)/")


class ProjectionError(Exception):
    pass


def tokens_of(planner, accelerate):
    """The run's own interpreter and bytecode prefix, read from the raw outputs: (python, pycache)."""
    pythons = {(a.get("argv") or [None])[0] for a in accelerate.values()}
    prefixes = {(v.get("release_proof") or {}).get("pycache_prefix") for v in (planner.get("versions") or {}).values()}
    prefixes.discard(None)
    if len(pythons) != 1 or None in pythons:
        raise ProjectionError(f"accelerate.json names {len(pythons)} interpreters, not one")
    if len(prefixes) != 1:
        raise ProjectionError(f"the release proofs name {len(prefixes)} bytecode prefixes, not one")
    return pythons.pop(), prefixes.pop()


def project_str(s, python, pycache):
    """One string with the run's paths made tokens."""
    if s == python:
        return "<python>"
    if s == pycache or s.startswith(pycache.rstrip("/") + "/"):
        return "<pycache>" + s[len(pycache.rstrip("/")):] if s != pycache else "<pycache>"
    return SITE.sub("<site-packages>", s)


def project(obj, python, pycache):
    """``obj`` with every string projected; keys, numbers, booleans, nulls and the structure untouched."""
    if isinstance(obj, dict):
        return {k: project(v, python, pycache) for k, v in obj.items()}
    if isinstance(obj, list):
        return [project(v, python, pycache) for v in obj]
    if isinstance(obj, str):
        return project_str(obj, python, pycache)
    return obj


def leftover_paths(obj, where=""):
    """Every string (or key) in ``obj`` that still holds an absolute path, with where it sits."""
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if ABSOLUTE.search(str(k)) or str(k).startswith("/"):
                out.append(f"{where}/{k} (key)")
            out += leftover_paths(v, f"{where}/{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out += leftover_paths(v, f"{where}[{i}]")
    elif isinstance(obj, str) and (obj.startswith("/") or ABSOLUTE.search(obj)):
        out.append(where)
    return out


def dumps(obj):
    """The driver's own layout (h2h_planner.py's ``json.dumps(..., indent=1, sort_keys=True)``)."""
    return json.dumps(obj, indent=1, sort_keys=True) + "\n"


def project_dir(raw_dir):
    """{public file name: text} for the raw pair in ``raw_dir``."""
    raw = {name: json.load(open(os.path.join(raw_dir, name))) for name, _ in FILES}
    python, pycache = tokens_of(raw["h2h-planner.json"], raw["accelerate.json"])
    out = {}
    for name, public in FILES:
        p = project(raw[name], python, pycache)
        left = leftover_paths(p)
        if left:
            raise ProjectionError(f"{public}: absolute paths left at {left[:3]}")
        out[public] = dumps(p)
    return out


def main(argv):
    if len(argv) != 2:
        print("usage: python bench/h2h-planner/project_public.py RAW_DIR OUT_DIR", file=sys.stderr)
        return 2
    try:
        files = project_dir(argv[0])
    except ProjectionError as e:
        print(f"refused: {e}", file=sys.stderr)
        return 1
    os.makedirs(argv[1], exist_ok=True)
    for name, text in files.items():
        with open(os.path.join(argv[1], name), "w") as f:
            f.write(text)
        print(name)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
