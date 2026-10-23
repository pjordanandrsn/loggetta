"""What a shared record (a receipt) may carry: no local path, which would name the user's machine layout, home
directory and username.

* :func:`label` names a file a record cites: a Hub id (``org/name``) or a URL as given, a local path by its name only
  (``local: <name>``), the rule ``scripts/first_run.sh`` applies. :func:`model_label` names a MODEL, adding a short
  fingerprint of a local directory's config so two directories named alike never compare equal.
* :func:`provenance` turns :func:`measure.provenance`'s record into its shareable form: a path argument in ``argv`` by
  its name (and, for a file, its sha256), a source file by its package-relative path, version and sha256.
* :func:`scrub` is the last pass before a record is written: any absolute path left anywhere, as a value or inside a
  sentence, by its name. Matching keeps working: the planner compares models by :func:`model_label` on both sides.
"""
from __future__ import annotations

import functools
import hashlib
import os
import re

#: distribution names of the source modules provenance tracks, where they differ from the module name (this package's
#: own is its module name), for their versions
_DISTS = {"experts4bit_qlora": "experts4bit-qlora", "nf4_grouped": "grouped-nf4-gemm"}
#: an absolute or home-relative path inside a sentence ("profile loaded from /srv/x.json, not probed"); a URL's "//" is
#: not one, since the slash that starts a path may not follow ":", "/" or a word character
# Two segments at least: "/health" or "/s" alone names no machine, while a home, a mount or a checkout has a directory
# in it. A Windows drive path ("C:\\Users\\someone\\x") is one too.
_PATH_IN_TEXT = re.compile(r"(?<![\w.:/~])(?:~/[^\s'\",;()\[\]]+|/[^\s'\",;()\[\]/]+/[^\s'\",;()\[\]]+)"
                           r"|(?<![\w])[A-Za-z]:\\[^\s'\",;()\[\]]+")
#: a file argument up to this size is hashed whole; above it, its size and the sha256 of its first MiB (an NF4 arena is
#: tens of GB, and a receipt is written on paid box time)
FULL_HASH_MAX = 64 << 20
HEAD_BYTES = 1 << 20


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


def _model_fingerprint(d: str) -> str | None:
    """12 hex of the sha256 of a local model directory's ``config.json`` plus its safetensors index (or, with no index,
    its weight files' names and sizes): what tells two directories named alike apart, read without loading weights."""
    cfg = os.path.join(d, "config.json")
    if not os.path.isfile(cfg):
        return None
    h = hashlib.sha256()
    with open(cfg, "rb") as f:
        h.update(f.read())
    idx = os.path.join(d, "model.safetensors.index.json")
    if os.path.isfile(idx):
        with open(idx, "rb") as f:
            h.update(f.read())
    else:
        for name in sorted(os.listdir(d)):
            if name.endswith((".safetensors", ".bin", ".pt", ".gguf")):
                h.update(f"{name}:{os.path.getsize(os.path.join(d, name))}\n".encode())
    return h.hexdigest()[:12]


@functools.lru_cache(maxsize=256)
def model_label(value):
    """A MODEL as a shared record names it and the planner compares it: a Hub id as given, a local model directory as
    ``local: <name>@<12 hex>`` (:func:`_model_fingerprint`). Two directories both called "model" or "checkpoint-500"
    compare equal only with the same architecture and sharding (config.json plus the weight index, or the weight files'
    names and sizes); the weights themselves are not hashed, so two fine-tunes of one base sharded alike DO compare
    equal, which suits shape-driven numbers (memory, step time) but not ones that follow the weights (MoE routing). A
    local path without a readable config.json, or a label already made, by :func:`label` (and :func:`match_key`
    matches it to nothing)."""
    if not isinstance(value, str) or value.startswith("local: ") or "://" in value or not is_local(value):
        return label(value)
    d = os.path.expanduser(value)
    fp = _model_fingerprint(d) if os.path.isdir(d) else None
    return label(value) + (f"@{fp}" if fp else "")


def match_key(value):
    """What the planner compares models by: :func:`model_label`, except that a local model WITHOUT a fingerprint (no
    readable config.json) matches nothing, not even itself, so a run on it is treated as unseen and measured rather than
    lent another same-named directory's numbers."""
    lab = model_label(value)
    if isinstance(lab, str) and lab.startswith("local: ") and "@" not in lab:
        return object()
    return lab


def _sha256(path, limit=None) -> str | None:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            if limit is not None:
                h.update(f.read(limit))
            else:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
    except OSError:
        return None
    return h.hexdigest()


def file_digest(path) -> dict:
    """``{"size", "sha256"}`` for a file up to :data:`FULL_HASH_MAX`; ``{"size", "head_sha256"}`` (its first
    :data:`HEAD_BYTES`) above it, so a receipt never reads a multi-GB arena whole."""
    size = os.path.getsize(path)
    if size <= FULL_HASH_MAX:
        return {"size": size, "sha256": _sha256(path)}
    return {"size": size, "head_sha256": _sha256(path, HEAD_BYTES), "head_bytes": HEAD_BYTES}


def _argv(argv) -> tuple:
    """``argv`` with each path argument (``PATH`` or ``--opt=PATH``) by its name, plus ``{name: sha256}`` for files."""
    out, files = [], {}
    for a in argv:
        key, sep, val = a.partition("=") if a.startswith("-") and "=" in a else ("", "", a)
        if "://" not in val and is_local(val):
            named = label(val)
            if os.path.isfile(os.path.expanduser(val)):
                files[named[len("local: "):]] = file_digest(os.path.expanduser(val))
            val = named
        out.append(key + sep + val)
    return out, files


def _source(name: str, src: dict, versions: dict) -> dict:
    """A source module's record with its file named relative to the directory it was imported from, with forward
    slashes on every OS, so the same source reads the same in a receipt written on Windows."""
    f = src.get("file")
    if not f:
        return src
    root = os.path.dirname(os.path.dirname(f)) if os.path.basename(f) == "__init__.py" else os.path.dirname(f)
    return {**{k: v for k, v in src.items() if k != "file"}, "file": os.path.relpath(f, root).replace(os.sep, "/"),
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
        return {scrub(k) if isinstance(k, str) else k: scrub(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [scrub(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(scrub(v) for v in obj)
    if isinstance(obj, str):
        # the last segment on either separator: a Windows path's backslashes are not os.sep on a POSIX box
        return _PATH_IN_TEXT.sub(lambda m: "local: " + re.split(r"[\\/]", m.group(0).rstrip("/\\"))[-1], obj)
    return obj


def record(r: dict) -> dict:
    """A receipt in its shareable form: provenance by :func:`provenance`, the model (top level and in the embedded plan)
    by :func:`label`, then :func:`scrub` over everything."""
    out = dict(r)
    if isinstance(out.get("provenance"), dict):
        out["provenance"] = provenance(out["provenance"])
    if isinstance(out.get("model"), dict):
        out["model"] = {**out["model"], "model": model_label(out["model"].get("model"))}
    if isinstance(out.get("plan"), dict) and isinstance(out["plan"].get("model"), dict):
        out["plan"] = {**out["plan"],
                       "model": {**out["plan"]["model"], "model": model_label(out["plan"]["model"].get("model"))}}
    return scrub(out)
