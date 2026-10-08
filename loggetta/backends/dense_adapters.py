"""PEFT adapter artifacts for the dense backend, with manifest-last publication and checksummed reload."""
from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path

from ..data import file_sha256
from .experts4bit_adapters import MANIFEST, SCHEMA, validate_target

WEIGHTS = "adapter_model.safetensors"


def save_adapter(model, tokenizer, directory, plan, *, data, report, seed):
    import torch

    from .dense_train import adapter_parameters

    target = validate_target(directory)
    params = adapter_parameters(model)
    if any(not torch.isfinite(p).all() for p in params.values()):
        raise ValueError("refusing to export non-finite dense adapter tensors")
    target.mkdir(exist_ok=False)
    try:
        model.save_pretrained(target, safe_serialization=True, save_embedding_layers=False)
        tokenizer.save_pretrained(target)
        if not (target / WEIGHTS).is_file() or not (target / "adapter_config.json").is_file():
            raise ValueError("PEFT did not export the expected adapter weights and configuration")
        files = {str(p.relative_to(target)): {"sha256": file_sha256(p), "bytes": p.stat().st_size}
                 for p in sorted(target.rglob("*")) if p.is_file()}
        manifest = {"schema": SCHEMA, "backend": "dense", "format": "PEFT linear LoRA",
                    "model": dict(plan.model), "resolved_model_revision": report.get("commit") or plan.model.get("revision"),
                    "setup": dict(plan.selected.setup), "workload": asdict(plan.workload), "seed": seed,
                    "plan_schema": plan.schema, "data": data, "weights": WEIGHTS, "files": files,
                    "trainable_numel": sum(p.numel() for p in params.values()), "optimizer_state_included": False,
                    "note": "Inference adapter loading, not optimizer resumption. Local base snapshots must remain "
                            "available; an unpinned Hub revision is not reproducible."}
        (target / MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {"path": str(target), "manifest": MANIFEST, "format": "PEFT linear LoRA", "weights": WEIGHTS,
                "sha256": files[WEIGHTS]["sha256"], "tensor_count": len(params),
                "trainable_numel": manifest["trainable_numel"]}
    except BaseException:
        shutil.rmtree(target)
        raise


def read_manifest(directory):
    target = Path(directory).expanduser()
    manifest = json.loads((target / MANIFEST).read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA or manifest.get("backend") != "dense" or manifest.get("weights") != WEIGHTS:
        raise ValueError("unsupported dense adapter manifest")
    if not {WEIGHTS, "adapter_config.json"}.issubset(manifest.get("files", {})):
        raise ValueError("dense adapter manifest omits its weights or PEFT configuration")
    for name, info in manifest["files"].items():
        file = target / name
        if (Path(name).is_absolute() or ".." in Path(name).parts or file.is_symlink()
                or not file.resolve().is_relative_to(target.resolve())):
            raise ValueError("adapter manifest contains an unsafe file path")
        if not file.is_file() or file.stat().st_size != info["bytes"] or file_sha256(file) != info["sha256"]:
            raise ValueError(f"adapter artifact checksum mismatch: {name}")
    return manifest


def load_adapter(directory, *, device="cuda"):
    import torch
    from peft import PeftModel
    from peft.utils import get_peft_model_state_dict
    from safetensors.torch import load_file

    from .dense_loader import load_base

    manifest = read_manifest(directory)
    model, _, _ = load_base(manifest["model"]["model"], manifest["setup"], device=device,
                            revision=manifest.get("resolved_model_revision"))
    model = PeftModel.from_pretrained(model, directory, is_trainable=False,
                                      autocast_adapter_dtype=manifest["setup"]["adapter_dtype"] == "fp32")
    saved = load_file(str(Path(directory).expanduser() / WEIGHTS), device="cpu")
    restored = get_peft_model_state_dict(model)
    if set(saved) != set(restored) or any(saved[key].dtype != restored[key].dtype or
                                        not torch.equal(saved[key], restored[key].detach().cpu()) for key in saved):
        raise ValueError("PEFT adapter reload differs in keys, shapes, dtype or values from the exported tensors")
    if manifest["setup"]["placement"] == "stream":
        from experts4bit_qlora.engines.dense_offload import enable_dense_offload

        handles = enable_dense_offload(model, device=device)
        if not handles or not sum(h.bytes for h in handles):
            raise ValueError("dense adapter reload did not engage the recorded streamed placement")
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    model.eval()
    return model
