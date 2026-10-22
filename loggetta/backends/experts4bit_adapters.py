"""Portable native adapter artifacts for the included runtime, not PEFT-format checkpoints.

The runtime's own structural adapter discovery decides which tensors are adapters; its recipe rebuilds the
selected setup. This module only serializes those trainable tensors, identity/configuration, and the tokenizer.
It never saves a full model state_dict, frozen expert weights, or optimizer state.

Two manifest schemas, both defined by experts4bit-qlora (``docs/ADAPTER_PORTABILITY.md``):
* ``backend-adapter/1``: the setup, the base revision and the tensors.
* ``backend-adapter/2``: v1 plus what the base was quantized to, so a server or :func:`load_adapter` can check the base
  it rebuilt against the one the adapter was trained on. The field list is ``experts4bit_qlora.adapter_schema.V2_FIELDS``
  (one schema, one place). The facts training resolved come from ``prepare_qlora_training``'s report, and the base
  fingerprints from ``adapter_schema.quant_fingerprint``, read where training kept the expert stacks
  (:data:`FINGERPRINT_SOURCE`). Nothing is filled in from defaults: a missing fact refuses the save.

``save_adapter`` writes ``/2`` when the installed experts4bit-qlora provides ``adapter_schema`` and ``/1`` otherwise.
"""
from __future__ import annotations

import json
import shutil
import tempfile
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

from ..data import file_sha256

SCHEMA = SCHEMA_V1 = "backend-adapter/1"
SCHEMA_V2 = "backend-adapter/2"
SCHEMAS = (SCHEMA_V1, SCHEMA_V2)
WEIGHTS = "adapter.safetensors"
MANIFEST = "adapter_manifest.json"
#: the facts ``prepare_qlora_training`` resolved and reports; a /2 manifest copies them from that report
TRAINING_FACTS = ("expert_absmax", "train_fused_qkv", "attn_quant", "versions")
#: ``setup.expert_residency`` -> the ``quant_fingerprint`` source that reads the stacks where training kept them. A
#: run recorded as ``device`` is fingerprinted from its resident tensors, so a stack that turns out to be evicted is
#: refused loudly instead of being read from a host home nobody recorded.
FINGERPRINT_SOURCE = {"device": "resident", "host": "offload_homes"}


class V2Unavailable(ValueError):
    """``backend-adapter/2`` cannot be recorded for this run (named reason). Distinct from a broken artifact: the
    tensors are fine, only the base facts cannot be recorded."""


def adapter_schema():
    """experts4bit-qlora's ``adapter_schema`` module, or None when the installed release predates ``backend-adapter/2``."""
    try:
        from experts4bit_qlora import adapter_schema as mod
    except ImportError:
        return None
    return mod


def fingerprint_source(setup: dict) -> str:
    residency = setup.get("expert_residency")
    if residency not in FINGERPRINT_SOURCE:
        raise V2Unavailable(f"backend-adapter/2: the setup records expert_residency {residency!r}; the fingerprint source "
                         f"is chosen from it ({FINGERPRINT_SOURCE}), so an unrecorded residency is refused")
    return FINGERPRINT_SOURCE[residency]


def fingerprint_scopes(model) -> tuple:
    """The scopes a /2 manifest records, as e4b's reader requires them: ``experts`` always (an experts4bit model has
    expert stacks), ``attention`` whenever the adapter carries attention LoRA."""
    from experts4bit_qlora.lora import trainable_lora_param_ids

    _, attention_ids = trainable_lora_param_ids(model)
    return ("experts", "attention") if attention_ids else ("experts",)


def v2_preflight(model, setup: dict, report: dict, *, run_id: str | None) -> dict:
    """Everything ``backend-adapter/2`` records except the time it is written, or ValueError naming what cannot be
    recorded. The trainer calls it right after ``prepare_qlora_training``, BEFORE the first step, so a run that could
    not save a ``/2`` manifest is refused before it trains, not after (a finished run would otherwise end SAVE_FAILED).
    Every input is known by then: the training facts are in ``report`` (``prepare_qlora_training`` adds them after
    ``enable_fast_train`` resolved them), and the LoRA scale is a module constant. The base fingerprints taken here are
    compared at save with fresh ones (:func:`v2_fields`), which is what proves the base did not change during training."""
    schema = adapter_schema()
    if schema is None:
        raise V2Unavailable("backend-adapter/2 needs experts4bit-qlora's adapter_schema; the installed release predates it")
    if not (isinstance(run_id, str) and run_id.strip()):     # the reader refuses the same set (whitespace-only too)
        # a reader checks only that run_id is present, so a null would pass every verifier unnoticed
        raise V2Unavailable(f"backend-adapter/2 records the run that produced the adapter; run_id {run_id!r} is not one")
    missing = [k for k in TRAINING_FACTS if k not in report]
    if missing:
        raise V2Unavailable(f"backend-adapter/2: the training report lacks {missing}. They come from "
                         f"prepare_qlora_training's report, after enable_fast_train resolved them, never from defaults")
    source = fingerprint_source(setup)
    fields = {k: report[k] for k in TRAINING_FACTS}
    fields["lora_scale_applied"] = schema.lora_scale_applied(model)
    try:
        fields["fingerprints"] = {scope: schema.quant_fingerprint(model, scope, source=source)
                                  for scope in fingerprint_scopes(model)}
    except ValueError as e:                     # e.g. a device run whose stacks are evicted: the base cannot be read
        raise V2Unavailable(f"backend-adapter/2: the base cannot be fingerprinted: {e}") from None
    fields["run_id"] = run_id
    unknown = sorted(set(schema.V2_FIELDS) - set(fields) - {"created_at"})
    if unknown:                       # a field the schema added that this writer does not record yet: refuse, never omit
        raise V2Unavailable(f"backend-adapter/2 names fields this writer does not record: {unknown}")
    return fields


def v2_fields(model, setup: dict, report: dict, *, run_id: str | None, preflight: dict | None = None) -> dict:
    """The fields ``backend-adapter/2`` adds to v1's, taken now, and ``created_at``. With the trainer's ``preflight``
    (taken before its first step), the base is fingerprinted AGAIN here and compared scope by scope: the frozen-digest
    check during training samples only some expert layers and no attention base, so this comparison is what proves the
    base the manifest records is the base the adapter trained on. A scope that differs raises :class:`V2Unavailable`
    naming it, and the trainer keeps the tensors as ``/1``."""
    fields = v2_preflight(model, setup, report, run_id=run_id)
    if preflight is not None:
        if preflight.get("run_id") != run_id:
            raise V2Unavailable(f"the preflight recorded run {preflight.get('run_id')!r}, the save names {run_id!r}")
        before, after = preflight.get("fingerprints") or {}, fields["fingerprints"]
        changed = sorted(s for s in set(before) | set(after)
                         if (before.get(s) or {}).get("sha256") != (after.get(s) or {}).get("sha256")
                         or (before.get(s) or {}).get("modules") != (after.get(s) or {}).get("modules"))
        if changed:
            raise V2Unavailable(f"the base changed during training: the {changed} fingerprint(s) taken before the first "
                                f"step differ from the ones at save")
    fields["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return fields


def trainable_adapters(model) -> dict:
    from experts4bit_qlora.lora import trainable_lora_param_ids

    expert_ids, attention_ids = trainable_lora_param_ids(model)
    allowed = expert_ids | attention_ids
    params = {name: p for name, p in model.named_parameters() if p.requires_grad}
    unexpected = [name for name, p in params.items() if id(p) not in allowed]
    if unexpected:
        raise ValueError(f"refusing adapter-only export: non-adapter trainables {unexpected[:5]}")
    if not params:
        raise ValueError("no trainable adapters to export")
    return params


def validate_target(directory) -> Path:
    """Check a fresh, writable destination before spending time loading or training."""
    target = Path(directory).expanduser().absolute()
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"adapter output already exists; use a new --adapter-out path: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile(dir=target.parent):
        pass
    return target


def save_adapter(model, tokenizer, directory, plan, *, data: dict, report: dict, seed: int,
                 schema: str | None = None, run_id: str | None = None, preflight: dict | None = None) -> dict:
    """Write a fresh artifact directory. Manifest-last prevents partial output looking complete.

    ``schema``: ``backend-adapter/1`` or ``/2``; None picks ``/2`` when the installed experts4bit-qlora defines it and
    ``/1`` otherwise. Everything ``/2`` records (the fingerprints included) is computed BEFORE the directory exists,
    so a refusal leaves nothing behind. ``preflight`` is :func:`v2_preflight`'s result when the trainer took it before
    its first step; a preflight given with ``/2`` resolved selects ``/2``. When ``/2`` was chosen automatically but
    cannot be recorded, the adapter is written as ``/1`` (tensors are never discarded over manifest metadata) and
    the result and the manifest name why (``v2_refused``); an explicit ``schema=/2`` raises instead."""
    import torch
    from safetensors.torch import save_file

    explicit = schema is not None
    if schema is None:
        schema = SCHEMA_V2 if (preflight is not None or adapter_schema() is not None) else SCHEMA_V1
    if schema not in SCHEMAS:
        raise ValueError(f"unknown adapter schema {schema!r}; one of {SCHEMAS}")
    target = validate_target(directory)
    params = trainable_adapters(model)
    extra, refused = {}, None
    if schema == SCHEMA_V2:
        try:
            extra = v2_fields(model, dict(plan.selected.setup), report, run_id=run_id, preflight=preflight)
        except V2Unavailable as e:
            if explicit:
                raise
            # trained tensors are never discarded over manifest metadata: write /1, and say why no /2 exists
            schema, refused = SCHEMA_V1, str(e)
            extra = {"v2_refused": refused}
    tensors = {}
    for name, param in params.items():
        tensor = param.detach().cpu().contiguous().clone()
        if not torch.isfinite(tensor).all():
            raise ValueError(f"refusing to save non-finite adapter tensor {name}")
        tensors[name] = tensor
    # Atomic exclusive creation: never overwrite any user's existing adapter directory.
    target.mkdir(exist_ok=False)
    try:
        save_file(tensors, str(target / WEIGHTS), metadata={"format": schema, "backend": "experts4bit"})
        tokenizer.save_pretrained(str(target))
        # Record checksums for every file produced, including tokenizer configuration.
        files = {str(p.relative_to(target)): {"sha256": file_sha256(p), "bytes": p.stat().st_size}
                 for p in sorted(target.rglob("*")) if p.is_file()}
        manifest = {
            "schema": schema, "backend": "experts4bit", "format": "native adapter tensors (not PEFT)",
            "model": dict(plan.model), "resolved_model_revision": report.get("commit") or plan.model.get("revision"),
            "setup": dict(plan.selected.setup), "workload": asdict(plan.workload), "seed": seed,
            "plan_schema": plan.schema, "data": data,
            "runtime_version": version("experts4bit-qlora"), "weights": WEIGHTS, "files": files,
            "tensors": {name: {"shape": list(t.shape), "dtype": str(t.dtype)} for name, t in tensors.items()},
            "trainable_numel": sum(t.numel() for t in tensors.values()),
            "optimizer_state_included": False,
            "note": "Rebuild the base model with this setup and revision, then load these adapter tensors. "
                    "An unpinned base revision is not reproducible; local base snapshots must remain available.",
            **extra,
        }
        (target / MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        result = {"path": str(target), "manifest": MANIFEST, "format": schema,
                "weights": WEIGHTS, "sha256": files[WEIGHTS]["sha256"],
                "tensor_count": len(tensors), "trainable_numel": manifest["trainable_numel"]}
        if refused is not None:
            result["v2_refused"] = refused       # a /1 written where /2 was expected: always reported
        return result
    except BaseException:
        shutil.rmtree(target)
        raise


def read_manifest(directory) -> dict:
    target = Path(directory).expanduser()
    manifest = json.loads((target / MANIFEST).read_text(encoding="utf-8"))
    if manifest.get("schema") not in SCHEMAS or manifest.get("backend") != "experts4bit":
        raise ValueError("unsupported adapter manifest schema or backend")
    if manifest.get("weights") != WEIGHTS or WEIGHTS not in manifest.get("files", {}):
        raise ValueError("adapter manifest must name adapter.safetensors")
    for name, info in manifest["files"].items():
        file = target / name
        if (Path(name).is_absolute() or ".." in Path(name).parts or file.is_symlink()
                or not file.resolve().is_relative_to(target.resolve())):
            raise ValueError("adapter manifest contains an unsafe file path")
        if not file.is_file() or file.stat().st_size != info["bytes"] or file_sha256(file) != info["sha256"]:
            raise ValueError(f"adapter artifact checksum mismatch: {name}")
    if manifest["schema"] == SCHEMA_V2:
        verify_v2(target)
    return manifest


def verify_v2(directory) -> None:
    """A ``/2`` directory through experts4bit-qlora's own verifier (``adapter_dir.verify_adapter_dir``: the v1 checks,
    every V2 field present, known fingerprint schemes, the applied scale equal to alpha/r). One verifier, the reader's,
    so the writer and the server never disagree about what a valid ``/2`` is."""
    try:
        from experts4bit_qlora.adapter_dir import AdapterRefused, verify_adapter_dir
    except ImportError:
        raise ValueError("this adapter is backend-adapter/2; reading it needs an experts4bit-qlora that defines it "
                         "(adapter_schema / adapter_dir), and the installed release predates it") from None
    try:
        verify_adapter_dir(directory)
    except AdapterRefused as e:
        raise ValueError(str(e)) from None


def verify_base(model, manifest: dict) -> dict:
    """For a ``/2`` manifest: the rebuilt model's base, fingerprinted the same way, against the fingerprints recorded
    at training. ``{scope: {"recorded", "rebuilt", "match"}}``; raises ValueError on any mismatch, naming the scope.
    A v1 manifest records none, so there is nothing to check (``{}``)."""
    if manifest.get("schema") != SCHEMA_V2:
        return {}
    schema = adapter_schema()
    if schema is None:
        raise ValueError("verifying a backend-adapter/2 base needs experts4bit-qlora's adapter_schema")
    source = fingerprint_source(manifest["setup"])
    out, bad = {}, []
    for scope, fp in sorted(manifest["fingerprints"].items()):
        if fp.get("scheme_id") not in schema.KNOWN_SCHEMES:
            raise ValueError(f"fingerprints.{scope}: scheme {fp.get('scheme_id')!r} is not one this runtime knows "
                             f"({list(schema.KNOWN_SCHEMES)})")
        now = schema.quant_fingerprint(model, scope, source=source)
        match = now["sha256"] == fp["sha256"] and now["modules"] == fp["modules"]
        out[scope] = {"recorded": fp["sha256"], "rebuilt": now["sha256"], "modules": now["modules"], "match": match}
        if not match:
            bad.append(scope)
    if bad:
        raise ValueError(f"the rebuilt base differs from the one this adapter was trained on: {bad} fingerprint(s) "
                         f"do not match (e.g. another absmax form, quant_type or blocksize; docs/ADAPTER_PORTABILITY.md)")
    return out


def describe(manifest: dict) -> list:
    """Lines for the CLI's ``inspect --adapter``: the adapter's identity and, for ``/2``, every fact it records about the
    base, so a reader sees which parts a server can verify. A v1 manifest says plainly that it records none."""
    files = manifest.get("files", {})
    lines = [f"schema {manifest.get('schema')}, backend {manifest.get('backend')}",
             f"model {manifest.get('model', {}).get('model')} @ {manifest.get('resolved_model_revision') or 'unpinned'}",
             f"setup {json.dumps(manifest.get('setup'), sort_keys=True)}",
             f"adapter {WEIGHTS} sha256 {files.get(WEIGHTS, {}).get('sha256')}: {len(manifest.get('tensors', {}))} tensors, "
             f"{manifest.get('trainable_numel')} parameters"]
    if manifest.get("schema") != SCHEMA_V2:
        lines.append("base facts: none recorded (backend-adapter/1): the expert absmax form, the attention recipe and the "
                     "base fingerprints are unknown, so no server can verify the base (docs/ADAPTER_PORTABILITY.md F1, F2)")
        return lines
    absmax, attn = manifest["expert_absmax"], manifest["attn_quant"]
    lines += [f"expert absmax: {absmax.get('form')} ({absmax.get('stacks')} double-quantized stack(s))",
              f"attention base: {attn.get('form')}" + (f" {attn.get('quant_type')}, blocksize {attn.get('blocksize')}, "
                                                       f"nested {attn.get('nested')}" if attn.get("form") == "bnb4" else ""),
              f"fused q/k/v in training: {manifest['train_fused_qkv']} module(s)",
              f"LoRA scale applied: {json.dumps(manifest['lora_scale_applied'], sort_keys=True)}",
              "versions: " + ", ".join(f"{k} {v}" for k, v in sorted(manifest["versions"].items()))]
    for scope, fp in sorted(manifest["fingerprints"].items()):
        lines.append(f"fingerprint {scope}: {fp.get('scheme_id')} {fp.get('sha256')} ({fp.get('modules')} modules, "
                     f"source {fp.get('source')}, {fp.get('seconds')} s)")
    lines.append(f"run {manifest.get('run_id') or '(not recorded)'}, written {manifest.get('created_at')}")
    return lines


def load_adapter_weights(model, directory) -> dict:
    """Validate every tensor before copying any: wrong setup or damaged files cannot partially load."""
    import torch
    from safetensors.torch import load_file

    manifest = read_manifest(directory)
    params = trainable_adapters(model)
    tensors = load_file(str(Path(directory).expanduser() / WEIGHTS), device="cpu")
    if set(params) != set(tensors) or set(tensors) != set(manifest["tensors"]):
        raise ValueError("adapter parameter names differ from the rebuilt runtime setup")
    for name, tensor in tensors.items():
        recorded = manifest["tensors"][name]
        if (tuple(params[name].shape) != tuple(tensor.shape) or params[name].dtype != tensor.dtype
                or recorded["shape"] != list(tensor.shape) or recorded["dtype"] != str(tensor.dtype)):
            raise ValueError(f"adapter shape/dtype mismatch: {name}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"non-finite saved adapter tensor: {name}")
    with torch.no_grad():
        for name, tensor in tensors.items():
            params[name].copy_(tensor)
    return manifest


def load_adapter(directory, *, device="cuda"):
    """Rebuild the runtime's recorded setup, load its adapters, and return an evaluation-mode model.

    The tokenizer is saved beside the adapter and can be loaded with AutoTokenizer.from_pretrained(directory).
    This is inference loading, not exact optimizer/scheduler resumption. No remote code trust is enabled here.
    """
    from experts4bit_qlora.recipe import QLoRASetup, prepare_qlora_training

    manifest = read_manifest(directory)
    prep = prepare_qlora_training(manifest["model"]["model"], QLoRASetup(**manifest["setup"]), device=device,
                                  revision=manifest.get("resolved_model_revision"))
    verify_base(prep.model, manifest)           # /2: the rebuilt base must fingerprint as the trained one did
    load_adapter_weights(prep.model, directory)
    if hasattr(prep.model, "gradient_checkpointing_disable"):
        prep.model.gradient_checkpointing_disable()
    if hasattr(prep.model, "config"):
        prep.model.config.use_cache = True
    prep.model.eval()
    return prep.model
