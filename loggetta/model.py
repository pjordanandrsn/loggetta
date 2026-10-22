"""Model identity and topology, from the model-family layer that owns it.

The planner never parses a config itself: structure, admission and family conventions belong to the package that
loads the model. Each backend asks its own package (``describe``) and says whether it can plan for the answer
(``refusal``); this module is the one place that asks them.
"""
from __future__ import annotations


class NoModelProvider(RuntimeError):
    pass


class Refused:
    """What :func:`describe_model` returns when more than one backend described the model and every one refused it:
    the first description (its attributes read through, so callers that expect it still work) and each backend's own
    refusal, by backend name, so a plan can say why each one cannot take the model."""

    def __init__(self, first, backend: str, reasons: dict):
        self._first, self.backend, self.reasons = first, backend, reasons

    @property
    def description(self):
        return self._first

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._first, name)


def describe_model(model, *, revision=None, trust_remote_code=False, backends=None):
    """The model's topology without its weights (config.json at most, plus a meta-device module tree).

    Backends are asked in order, and the first description its backend can plan for is returned. When none can, the
    first description is returned anyway: its refusal is the plan's answer. The topology is the backend's own type
    (today experts4bit-qlora's ``MoETopology``)."""
    from .backends import BACKENDS

    described, missing = [], []
    for b in BACKENDS if backends is None else backends:
        try:
            topology = b.describe(model, revision=revision, trust_remote_code=trust_remote_code)
        except NoModelProvider as e:
            missing.append(e)
            continue
        why = b.refusal(topology)
        if why is None:
            return topology
        described.append((b.NAME, topology, why))
    if not described:
        if len(missing) == 1:
            raise missing[0]
        raise NoModelProvider("; ".join(str(e) for e in missing) or "no backend is installed to describe a model")
    if len(described) == 1:
        return described[0][1]
    return Refused(described[0][1], described[0][0], {name: why for name, _t, why in described})


#: the weight files a ``from_pretrained`` load reads: safetensors when the checkpoint has any, else the pickle formats
WEIGHT_SUFFIXES = ((".safetensors",), (".bin", ".pt", ".pth"))


def _weight_files(sizes: dict) -> dict:
    """The top-level weight files of a checkpoint listing (``{name: bytes}``). Subfolders are left out: a load reads
    the repository root, and repositories that also ship another format in a folder (``original/``) would double it."""
    top = {n: s for n, s in sizes.items() if "/" not in n}
    for suffixes in WEIGHT_SUFFIXES:
        picked = {n: s for n, s in top.items() if n.endswith(suffixes)}
        if picked:
            return picked
    return {}


def checkpoint_size(model, *, revision=None) -> dict:
    """The bytes of weight files a load of ``model`` maps from disk, without downloading them: a local directory is
    listed, a Hub id is asked of the Hub API (file metadata only), and the local Hub cache answers when the Hub cannot.

    Returns ``{"bytes", "files", "source"}``; ``bytes`` is None, with ``source`` saying why, when no answer was found.
    It never raises: the size only prices a warning."""
    from pathlib import Path

    path = Path(str(model)).expanduser()
    if path.is_dir():                                    # the directory's own name only: no home path in a receipt
        try:
            files = _weight_files({p.name: p.stat().st_size for p in path.iterdir() if p.is_file()})
        except OSError as e:
            return {"bytes": None, "files": 0, "source": f"local directory {path.name}: {type(e).__name__}"}
        return {"bytes": sum(files.values()) if files else None, "files": len(files),
                "source": f"local directory {path.name}" if files else f"no weight files in local directory {path.name}"}
    errors = []
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(str(model), revision=revision, files_metadata=True, timeout=10)
        files = _weight_files({s.rfilename: s.size for s in info.siblings or () if s.size is not None})
        if files:
            return {"bytes": sum(files.values()), "files": len(files),
                    "source": f"Hub API file metadata, {model}@{info.sha or revision or 'main'}"}
        errors.append("the Hub listed no weight files")
    except Exception as e:  # offline, gated, or no huggingface_hub: fall through to the cache
        errors.append(f"Hub API: {type(e).__name__}")
    try:
        from huggingface_hub import snapshot_download

        snap = Path(snapshot_download(str(model), revision=revision, local_files_only=True))
        files = _weight_files({p.name: p.stat().st_size for p in snap.iterdir() if p.is_file()})
        if files:
            return {"bytes": sum(files.values()), "files": len(files), "source": f"local Hub cache, {model}@{snap.name}"}
        errors.append("the local Hub cache holds no weight files")
    except Exception as e:
        errors.append(f"local Hub cache: {type(e).__name__}")
    return {"bytes": None, "files": 0, "source": "; ".join(errors)}
