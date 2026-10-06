"""Portable native adapter artifacts for the included runtime, not PEFT-format checkpoints.

The runtime's own structural adapter discovery decides which tensors are adapters; its recipe rebuilds the
selected setup. This module only serializes those trainable tensors, identity/configuration, and the tokenizer.
It never saves a full model state_dict, frozen expert weights, or optimizer state.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

from ..data import file_sha256

SCHEMA = "backend-adapter/1"
WEIGHTS = "adapter.safetensors"
MANIFEST = "adapter_manifest.json"


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


def save_adapter(model, tokenizer, directory, plan, *, data: dict, report: dict, seed: int) -> dict:
    """Write a fresh artifact directory. Manifest-last prevents partial output looking complete."""
    import torch
    from safetensors.torch import save_file

    target = validate_target(directory)
    params = trainable_adapters(model)
    tensors = {}
    for name, param in params.items():
        tensor = param.detach().cpu().contiguous().clone()
        if not torch.isfinite(tensor).all():
            raise ValueError(f"refusing to save non-finite adapter tensor {name}")
        tensors[name] = tensor
    # Atomic exclusive creation: never overwrite any user's existing adapter directory.
    target.mkdir(exist_ok=False)
    try:
        save_file(tensors, str(target / WEIGHTS), metadata={"format": SCHEMA, "backend": "experts4bit"})
        tokenizer.save_pretrained(str(target))
        # Record checksums for every file produced, including tokenizer configuration.
        files = {str(p.relative_to(target)): {"sha256": file_sha256(p), "bytes": p.stat().st_size}
                 for p in sorted(target.rglob("*")) if p.is_file()}
        manifest = {
            "schema": SCHEMA, "backend": "experts4bit", "format": "native adapter tensors (not PEFT)",
            "model": dict(plan.model), "resolved_model_revision": report.get("commit") or plan.model.get("revision"),
            "setup": dict(plan.selected.setup), "workload": asdict(plan.workload), "seed": seed,
            "plan_schema": plan.schema, "data": data,
            "runtime_version": version("experts4bit-qlora"), "weights": WEIGHTS, "files": files,
            "tensors": {name: {"shape": list(t.shape), "dtype": str(t.dtype)} for name, t in tensors.items()},
            "trainable_numel": sum(t.numel() for t in tensors.values()),
            "optimizer_state_included": False,
            "note": "Rebuild the base model with this setup and revision, then load these adapter tensors. "
                    "An unpinned base revision is not reproducible; local base snapshots must remain available.",
        }
        (target / MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {"path": str(target), "manifest": MANIFEST, "format": SCHEMA,
                "weights": WEIGHTS, "sha256": files[WEIGHTS]["sha256"],
                "tensor_count": len(tensors), "trainable_numel": manifest["trainable_numel"]}
    except BaseException:
        shutil.rmtree(target)
        raise


def read_manifest(directory) -> dict:
    target = Path(directory).expanduser()
    manifest = json.loads((target / MANIFEST).read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA or manifest.get("backend") != "experts4bit":
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
    return manifest


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
    load_adapter_weights(prep.model, directory)
    if hasattr(prep.model, "gradient_checkpointing_disable"):
        prep.model.gradient_checkpointing_disable()
    if hasattr(prep.model, "config"):
        prep.model.config.use_cache = True
    prep.model.eval()
    return prep.model
