"""What a shared record (a receipt) may carry: no local path, which would name the user's machine layout, home
directory and username.

* :func:`label` names a model or file a record cites: a Hub id (``org/name``) or a URL as given, a local path by its
  name only (``local: <name>``), the rule ``scripts/first_run.sh`` applies.
* :func:`provenance` turns :func:`measure.provenance`'s record into its shareable form: a path argument in ``argv`` by
  its name (and, for a file, its sha256), a source file by its package-relative path, version and sha256.
* :func:`scrub` is the last pass before a record is written: any absolute path left anywhere, as a value or inside a
  sentence, by its name. Matching keeps working: the planner compares models by :func:`label` on both sides.
"""
from __future__ import annotations

import hashlib
import os
import re

#: distribution names of the source modules provenance tracks, where they differ from the module name (this package's
#: own is its module name), for their versions
_DISTS = {"experts4bit_qlora": "experts4bit-qlora", "nf4_grouped": "grouped-nf4-gemm"}
#: an absolute or home-relative path inside a sentence ("profile loaded from /srv/x.json, not probed"); a URL's "//" is
#: not one, since the slash that starts a path may not follow ":", "/" or a word character
_PATH_IN_TEXT = re.compile(r"(?<![\w.:/~])~?/[^\s'\",;()\[\]]+")


def is_local(value) -> bool:
    """An absolute, "."- or "~"-relative path, or an existing file or directory (a bare name that exists here)."""
    return isinstance(value, str) and bool(value) and (
        os.path.isabs(value) or value.startswith((".", "~")) or os.path.exists(value))


def label(value):
    """A model id or file as a shared record may cite it: a local path as ``local: <name>``, anything else as given.
    ``label(label(x)) == label(x)``, so a record already labelled compares equal to a fresh plan of the same model."""
    if isinstance(value, str) and "://" not in value and not value.startswith("local: ") and is_local(value):
        return "local: " + os.path.basename(value.rstrip("/"))
    return value


def _sha256(path) -> str | None:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        return None
    return h.hexdigest()


def _argv(argv) -> tuple:
    """``argv`` with each path argument (``PATH`` or ``--opt=PATH``) by its name, plus ``{name: sha256}`` for files."""
    out, files = [], {}
    for a in argv:
        key, sep, val = a.partition("=") if a.startswith("-") and "=" in a else ("", "", a)
        if "://" not in val and is_local(val):
            named = label(val)
            if os.path.isfile(os.path.expanduser(val)):
                files[named[len("local: "):]] = _sha256(os.path.expanduser(val))
            val = named
        out.append(key + sep + val)
    return out, files


def _source(name: str, src: dict, versions: dict) -> dict:
    """A source module's record with its file named relative to the directory it was imported from."""
    f = src.get("file")
    if not f:
        return src
    root = os.path.dirname(os.path.dirname(f)) if os.path.basename(f) == "__init__.py" else os.path.dirname(f)
    return {**{k: v for k, v in src.items() if k != "file"}, "file": os.path.relpath(f, root),
            "sha256": _sha256(f), "version": versions.get(_DISTS.get(name, name))}


def provenance(prov: dict) -> dict:
    """:func:`measure.provenance`'s record in its shareable form (the original keeps its paths: ``changed_since``
    re-reads the source files at the end of a run)."""
    out = dict(prov)
    if "argv" in prov:
        out["argv"], files = _argv(prov["argv"])
        if files:
            out["argv_files"] = files
    if isinstance(prov.get("sources"), dict):
        out["sources"] = {n: _source(n, s, prov.get("versions") or {}) for n, s in prov["sources"].items()}
    return out


def scrub(obj):
    """``obj`` with every absolute or home-relative path left in it named by its name only, as a value or inside a
    sentence. The last pass before a shared record is written."""
    if isinstance(obj, dict):
        return {k: scrub(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [scrub(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(scrub(v) for v in obj)
    if isinstance(obj, str):
        if obj.startswith(("/", "~/")) and "://" not in obj:
            return label(obj) if is_local(obj) or obj.startswith("/") else obj
        return _PATH_IN_TEXT.sub(lambda m: "local: " + os.path.basename(m.group(0).rstrip("/")), obj)
    return obj


def record(r: dict) -> dict:
    """A receipt in its shareable form: provenance by :func:`provenance`, the model (top level and in the embedded plan)
    by :func:`label`, then :func:`scrub` over everything."""
    out = dict(r)
    if isinstance(out.get("provenance"), dict):
        out["provenance"] = provenance(out["provenance"])
    if isinstance(out.get("model"), dict):
        out["model"] = {**out["model"], "model": label(out["model"].get("model"))}
    if isinstance(out.get("plan"), dict) and isinstance(out["plan"].get("model"), dict):
        out["plan"] = {**out["plan"], "model": {**out["plan"]["model"], "model": label(out["plan"]["model"].get("model"))}}
    return scrub(out)
