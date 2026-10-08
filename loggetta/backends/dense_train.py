"""Execute a dense plan through transformers, PEFT, and the existing e4b training mechanisms."""
from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass


@dataclass
class Prepared:
    model: object
    trainable: list
    report: dict


def adapter_parameters(model):
    """PEFT linear LoRA parameters by module structure, with no other trainables admitted."""
    from peft.tuners.lora.layer import LoraLayer

    allowed = set()
    for module in model.modules():
        if isinstance(module, LoraLayer):
            for adapters in (module.lora_A, module.lora_B):
                allowed.update(id(p) for p in adapters.parameters())
    params = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if not params or any(id(p) not in allowed for p in params.values()):
        raise ValueError("dense training requires non-empty PEFT linear LoRA parameters and no other trainables")
    return params


def prepare(plan, *, device="cuda"):
    import torch
    from peft import LoraConfig, get_peft_model

    from . import dense
    from .dense_loader import load_base

    setup = dict(plan.selected.setup)
    if not isinstance(setup["r"], int) or isinstance(setup["r"], bool) or setup["r"] <= 0:
        raise ValueError("dense adapter rank must be a positive integer")
    if not isinstance(setup["alpha"], (int, float)) or not math.isfinite(setup["alpha"]) or setup["alpha"] <= 0:
        raise ValueError("dense adapter alpha must be finite and positive")
    topology = dense.describe(plan.model["model"], revision=plan.model.get("revision"))
    reasons = dense.setup_refusals(topology, setup, plan.workload)
    if topology.refusal or reasons:
        raise ValueError(topology.refusal or "; ".join(reasons))
    model, topology, report = load_base(plan.model["model"], setup, device=device, revision=plan.model.get("revision"))
    reasons = dense.setup_refusals(topology, setup, plan.workload)
    if reasons:
        raise ValueError("; ".join(reasons))
    for key in ("model_type", "n_layers", "linear_params", "embedding_params", "head_params", "targets"):
        if dense.summary(topology).get(key) != plan.model.get(key):
            raise ValueError(f"dense checkpoint description changed since planning: {key}; plan again")
    roles = set(setup["targets"])
    relative = {name for role, names in topology.targets.items() if role in roles for name in names}
    targets = [f"{name}.{projection}" for name, _ in model.named_modules() if dense._LAYER.search(name)
               for projection in sorted(relative)]
    if not targets:
        raise ValueError("dense plan selects no adapter targets")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=setup["r"], lora_alpha=setup["alpha"], lora_dropout=0.0,
                                           target_modules=targets, task_type="CAUSAL_LM", bias="none"))
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[setup["adapter_dtype"]]
    params = adapter_parameters(model)
    for param in params.values():
        param.data = param.data.to(device=device, dtype=dtype)
    expected = topology.n_layers * sum(setup["r"] * (lin.in_features + lin.out_features)
                                       for lin in topology.layer_linears if lin.role in roles)
    if sum(p.numel() for p in params.values()) != expected:
        raise ValueError("dense adapter parameter count differs from the plan")
    chunked = 0
    if setup["loss_chunk"]:
        from experts4bit_qlora.engines.chunked_lm_loss import enable_chunked_lm_loss

        chunked = enable_chunked_lm_loss(model.get_base_model(), chunk=setup["loss_chunk"], min_logits_bytes=0)
        if chunked != 1:
            raise ValueError("the planned chunked loss did not engage")
    handles = []
    if setup["placement"] == "stream":
        from experts4bit_qlora.engines.dense_offload import dense_offload_report, enable_dense_offload

        handles = enable_dense_offload(model, device=device, pin=torch.device(device).type == "cuda",
                                       train_prefetch=True)
        if len(handles) != topology.n_layers or not sum(h.bytes for h in handles):
            raise ValueError("the planned dense streaming did not engage for every decoder layer")
        if any(h.streamed_trainable for h in handles):
            raise ValueError("dense streaming attempted to stream trainable adapters")
        if torch.device(device).type == "cuda" and any(h.host_bytes and not h.pinned for h in handles):
            raise ValueError("dense streaming could not pin every host home")
        report["dense_offload"] = dense_offload_report(handles)
    report.update(backend="dense", setup=setup, adapter_targets=targets,
                  adapter_parameters=expected, adapter_dtype=str(dtype), chunked_loss_engaged=chunked,
                  train_prefetch=bool(handles), gradient_checkpointing="non-reentrant",
                  commit=topology.revision)
    return Prepared(model, list(params.values()), report)


def frozen_digest(model):
    """Hash frozen parameters of the first and last decoder layers, including offloaded CPU homes.

    This is sampled integrity, not a hash of the entire checkpoint. Quantization states are also hashed when present.
    """
    import torch

    from . import dense

    layers = [(name, module) for name, module in model.named_modules() if dense._LAYER.search(name)]
    selected = layers[:1] + (layers[-1:] if len(layers) > 1 else [])
    out = {}
    for name, layer in selected:
        module_names = {id(module): relative for relative, module in layer.named_modules()}
        homes = {}
        handle = getattr(layer, "_dense_offload", None)
        if handle is not None:
            for module, attr, is_param, home in handle.slots:
                if is_param:
                    homes[f"{module_names[id(module)]}.{attr}"] = home
        tensors = {key: homes.get(key, param) for key, param in layer.named_parameters() if not param.requires_grad}
        for relative, module in layer.named_modules():
            state = getattr(getattr(module, "weight", None), "quant_state", None)
            if state is not None:
                for key, value in state.as_dict(packed=True).items():
                    if isinstance(value, torch.Tensor):
                        tensors[f"{relative}.quant_state.{key}"] = value
        sha = hashlib.sha256()
        for key, tensor in sorted(tensors.items()):
            sha.update(key.encode())
            sha.update(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
        if not tensors:
            raise ValueError("no frozen dense tensors were checked")
        out[name] = sha.hexdigest()
    if not out:
        raise ValueError("no decoder layer is available for dense integrity checking")
    return out


def run(plan, *, seed=0, warmup=2, log=print, adapter_dir=None):
    from ..execution import PlanNotExecutable
    from . import dense

    if plan.constraints.allow_development_executor is not True:
        raise PlanNotExecutable(dense.planned_only_reason("train"))
    import torch
    from transformers import AutoTokenizer

    from ..data import TrainingData, prepare_data
    from ..measure import DriverMemorySampler, proc_status
    from .dense_adapters import save_adapter
    from .experts4bit_adapters import validate_target
    from .experts4bit_train import link_h2d, train_loop

    target = validate_target(adapter_dir) if adapter_dir is not None else None
    tokenizer = AutoTokenizer.from_pretrained(plan.model["model"], revision=plan.model.get("revision"),
                                              trust_remote_code=False)
    spec = TrainingData.from_dict(plan.workload.data)
    blocks = plan.workload.steps * plan.workload.grad_accum * plan.workload.micro_batch
    epochs = plan.workload.epochs
    start = time.perf_counter()
    with prepare_data(tokenizer, blocks, plan.workload.seq_len, spec, seed=seed,
                      expect=plan.data_profile, max_passes=math.ceil(epochs) if epochs and not spec.repeat else None,
                      cache_dir=str(target.parent) if target is not None else None) as data:
        data_seconds = time.perf_counter() - start
        torch.manual_seed(seed)
        torch.cuda.set_device(plan.constraints.device)
        torch.zeros(1, device="cuda")
        measured = {"host_baseline_bytes": proc_status().get("VmRSS"), **link_h2d(),
                    "data_prepare_seconds": data_seconds, "data_token_cache_bytes": data.info["token_cache_bytes"]}
        torch.cuda.reset_peak_memory_stats()
        with DriverMemorySampler() as sampler:
            start = time.perf_counter()
            prepared = prepare(plan, device=f"cuda:{plan.constraints.device}")
            measured.update(load_seconds=time.perf_counter() - start,
                            load_device_peak_bytes=torch.cuda.max_memory_allocated(),
                            host_anon_after_load_bytes=proc_status().get("RssAnon"))
            result = train_loop(prepared.model, prepared.trainable, plan.model["model"], plan.workload,
                                sampler, measured, seed=seed, warmup=warmup, log=log, prepared_data=data,
                                device=f"cuda:{plan.constraints.device}", frozen_digest=frozen_digest,
                                frozen_kind="dense")
        result["engaged"] = prepared.report
        if target is not None:
            if result["status"] != "OK":
                result["artifact_error"] = "integrity checks failed; no reusable adapter was exported"
            else:
                start = time.perf_counter()
                try:
                    result["artifacts"] = {"adapter": save_adapter(prepared.model, tokenizer, target, plan,
                                                                  data=result["data"], report=prepared.report, seed=seed)}
                except (OSError, ValueError, RuntimeError) as error:
                    result.update(status="SAVE_FAILED", artifact_error=f"{type(error).__name__}: {error}")
                result["measured"]["adapter_save_seconds"] = time.perf_counter() - start
        return result
