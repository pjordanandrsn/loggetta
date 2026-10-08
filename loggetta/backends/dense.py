"""The dense backend: decoder-only models without experts, for LoRA/QLoRA adapter training.

This module is the planner's side of that backend. It DESCRIBES a model -- from its config and a module tree built on
the ``meta`` device (no weights read) it finds the decoder layers and classifies every linear in them by its structural
role, so that LoRA targets are chosen by shape rather than by a per-family list of names -- and it PLANS adapter
training on it: candidate setups, an itemized memory estimate, refusals and the reasons for a choice. Its executor
uses transformers, PEFT and the existing experts4bit-qlora engines; no dense kernel is added here.

The roles:

* ``attn_in``  -- hidden -> query / key / value width (``q_proj``, ``k_proj``, ``v_proj``, or one fused ``qkv_proj``);
* ``attn_out`` -- query width -> hidden (``o_proj``);
* ``mlp_in``   -- hidden -> intermediate, or twice it when gate and up are fused (``gate_proj``, ``up_proj``);
* ``mlp_out``  -- intermediate -> hidden (``down_proj``).

Shapes decide; names only break a tie between roles of the same shape (a square ``q_proj`` and ``o_proj``). A linear
neither settles is *unclassified*: it stays frozen and is reported, never adapted. Family knowledge that structure
cannot show comes from the packages that own it: which attention implementations a model class declares
(transformers), whether the chunked LM loss covers its class (experts4bit-qlora's ``chunked_lm_loss_refusal``).

A refusal is an answer, not an exception: a model this backend cannot describe safely comes back with ``refusal``
set and the reason in words.
"""
from __future__ import annotations

import functools
import re
from dataclasses import asdict, dataclass, field

NAME = "dense"
WORKLOADS = ("train",)
#: setup fields that separate allocator-slack regimes (DQ4: resident 8.8%, streamed 27% under the default allocator)
SLACK_KEYS = {"train": ("base", "placement")}
#: which probed capabilities training uses: the planner reports only those as unusable
KERNELS_FOR = {"train": ("late_bound_4bit", "flash_attention_2", "dense_offload_pricing")}
BASES = ("bf16", "nf4")
PLACEMENTS = ("device", "stream")
ADAPTER_BYTES = {"fp32": 4, "bf16": 2}
#: the default adapter: PEFT's (fp32 adapters beside a bf16 or 4-bit base), rank and scale as experts4bit-qlora's DQ lanes
DEFAULT_ADAPTER = {"r": 16, "alpha": 32, "adapter_dtype": "fp32"}
#: the order a speed objective tries setups in, and the measured evidence for each step (experts4bit-qlora's DQ lanes,
#: Qwen3-32B architecture on an RTX 5090). Not a performance model: no time is predicted.
SPEED_EVIDENCE = {
    "base": "bf16 before NF4: the frozen weights stay exact, and NF4's dequantization costs at most ~10% of linear time at "
            "2,048 tokens per micro-batch and ~5% at 4,096 (DQ1: H(2048) = 0.097, H(4096) = 0.050)",
    "placement": "resident before streamed: streamed NF4 layers ran at 1.0023x (PCIe 5.0) and 1.0050x (PCIe 4.0) a resident "
                 "step at 2,048 tokens per micro-batch (DQ3, DQ5), but cost speed below the break-even of ~0.26 F/B "
                 "tokens (DQ1: ~1,760 on PCIe 4.0), and hold the frozen weights in pinned host memory",
}
#: activation bytes per token = the heuristic's per-token terms (checkpointed layer inputs + one layer's recompute) times
#: this coefficient: DQ4's measured slope over the formula's on Qwen3-32B (1.2557 / 1.0664 GiB per 1,024 tokens), with
#: fp32 PEFT adapters r16 on all seven projections, chunked loss (512) and SDPA (experts4bit-qlora bench/dq4, dq4-5090-2)
ACTIVATION_COEFFICIENT = 1.2557 / 1.0664
#: tokens per micro-batch above which a streamed NF4 layer hides behind its compute on PCIe 4.0 x16 (DQ1's ~0.26 F/B)
STREAM_BREAK_EVEN_TOKENS = 1760
DEFAULT_CHUNK = 512
GiB = 1 << 30
ROLES = ("attn_in", "attn_out", "mlp_in", "mlp_out")
#: name hints that settle a tie between roles of the same shape (the last path component of the linear)
NAME_HINTS = {
    "attn_in": {"q_proj", "k_proj", "v_proj", "qkv_proj", "query_key_value", "wqkv", "c_attn", "query", "key", "value"},
    "attn_out": {"o_proj", "out_proj", "dense", "wo"},
    "mlp_in": {"gate_proj", "up_proj", "gate_up_proj", "w1", "w3", "fc1", "c_fc", "dense_h_to_4h"},
    "mlp_out": {"down_proj", "w2", "fc2", "dense_4h_to_h"},
}
#: config keys that mean routed experts: such a model is the experts4bit backend's
EXPERT_KEYS = ("num_experts", "num_local_experts", "n_routed_experts", "moe_num_experts")
_LAYER = re.compile(r"(?:^|\.)layers\.(\d+)$")


@dataclass(frozen=True)
class Linear:
    """One linear inside a decoder layer."""

    name: str              # path inside the layer, e.g. "self_attn.q_proj"
    in_features: int
    out_features: int
    bias: bool
    role: str | None       # one of ROLES, or None (unclassified)


@dataclass(frozen=True)
class DenseTopology:
    """A dense decoder's structure, without its weights. Every field is data; :meth:`to_dict` is JSON-ready."""

    model: str
    model_type: str | None
    architecture: str | None
    revision: str | None
    #: why this backend cannot describe the model safely, or None
    refusal: str | None
    n_layers: int = 0
    hidden_size: int = 0
    intermediate_size: int = 0
    heads: int = 0
    kv_heads: int = 0
    head_dim: int = 0
    vocab_size: int = 0
    max_positions: int | None = None
    tied_embeddings: bool = False
    #: the linears of one decoder layer (the first; ``uniform`` says whether every layer has the same ones)
    layer_linears: tuple = ()
    uniform: bool = True
    #: parameter counts: every decoder-layer linear, embeddings, the untied head (0 if tied), everything else
    linear_params: int = 0
    embedding_params: int = 0
    head_params: int = 0
    other_params: int = 0
    unclassified: tuple = ()
    #: attention implementations the model class declares, and those that keep its semantics (softcapping, ...)
    attention_impls: tuple = ()
    attention_valid_impls: tuple = ()
    attention_notes: tuple = ()
    #: why experts4bit-qlora's chunked LM loss cannot take this class, or None when it can
    chunked_loss_refusal: str | None = None
    #: the description of a multimodal checkpoint is its text tower's
    text_tower_of: str | None = None
    provenance: dict = field(default_factory=dict)

    @property
    def targets(self) -> dict:
        """role -> the layer-relative names of the linears in it (what an adapter wraps)."""
        out = {r: [] for r in ROLES}
        for lin in self.layer_linears:
            if lin.role:
                out[lin.role].append(lin.name)
        return out

    def to_dict(self) -> dict:
        d = asdict(self)
        d["targets"] = self.targets
        return d

    def summary(self) -> str:
        if self.refusal:
            return f"{self.model} ({self.model_type}): not described by the dense backend: {self.refusal}"
        roles = ", ".join(f"{r} {len(n)}" for r, n in self.targets.items() if n)
        params = (self.linear_params + self.embedding_params + self.head_params + self.other_params) / 1e9
        return (f"{self.model} ({self.model_type}, dense decoder): {self.n_layers} layers, H {self.hidden_size}, "
                f"I {self.intermediate_size}, heads {self.heads}/{self.kv_heads} x {self.head_dim}, vocab "
                f"{self.vocab_size:,}{', tied head' if self.tied_embeddings else ''}; {params:.2f}B params "
                f"({self.linear_params / 1e9:.2f}B in decoder linears); per layer: {roles}"
                + (f"; {len(self.unclassified)} unclassified" if self.unclassified else ""))


def _int(cfg, *keys, default=0):
    for k in keys:
        v = getattr(cfg, k, None)
        if isinstance(v, int) and not isinstance(v, bool):
            return v
    return default


def classify(name: str, in_f: int, out_f: int, *, hidden: int, inter: int, q_width: int, kv_width: int):
    """The structural role of a decoder-layer linear, or None. Shapes first; a name hint only breaks a tie."""
    by_shape = set()
    if in_f == hidden and out_f in (q_width, kv_width, q_width + 2 * kv_width):
        by_shape.add("attn_in")
    if in_f == q_width and out_f == hidden:
        by_shape.add("attn_out")
    if in_f == hidden and out_f in (inter, 2 * inter):
        by_shape.add("mlp_in")
    if in_f == inter and out_f == hidden:
        by_shape.add("mlp_out")
    if len(by_shape) == 1:
        return by_shape.pop()
    leaf = name.rsplit(".", 1)[-1].lower()
    hinted = {r for r in by_shape if leaf in NAME_HINTS[r]}
    return hinted.pop() if len(hinted) == 1 else None


def _refused(model, config, why, prov, **extra):
    archs = getattr(config, "architectures", None) or [None]
    return DenseTopology(model=str(model), model_type=getattr(config, "model_type", None), architecture=archs[0],
                         revision=getattr(config, "_commit_hash", None), refusal=why, provenance=prov, **extra)


def describe(model, *, revision=None, trust_remote_code=False):
    """``model`` (a hub id, a local snapshot or a ``PretrainedConfig``) as a :class:`DenseTopology`, from its config
    and a meta-device module tree; no weight is read."""
    from ..model import NoModelProvider

    try:
        return _describe(model, revision=revision, trust_remote_code=trust_remote_code)
    except NoModelProvider:
        raise
    except Exception as e:
        # Third-party configs and model builders may raise family-specific exception types. Treat every such
        # failure as a refusal; even reading identifying config attributes again could repeat the exception.
        return DenseTopology(model=str(model) if isinstance(model, (str, bytes)) else type(model).__name__,
                             model_type=None, architecture=None, revision=revision,
                             refusal=f"the dense backend cannot describe it safely ({type(e).__name__}: {e})"[:400],
                             provenance={"structure": "description failed before safe admission"})


def _describe(model, *, revision=None, trust_remote_code=False):
    try:
        import torch
        import transformers
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError as e:
        from ..model import NoModelProvider

        raise NoModelProvider(f"the dense backend needs torch and transformers to describe a model ({e})") from e

    prov = {"config": str(getattr(model, "_name_or_path", model)), "transformers": transformers.__version__,
            "structure": "module tree built on meta from the config (no weights read)"}
    if isinstance(model, transformers.PretrainedConfig):
        config, name = model, getattr(model, "_name_or_path", None) or model.model_type
    else:
        name = str(model)
        try:
            config = AutoConfig.from_pretrained(model, revision=revision, trust_remote_code=trust_remote_code)
        except ValueError as e:                # e.g. a model type that needs its own remote modeling code
            return DenseTopology(model=name, model_type=None, architecture=None, revision=None, provenance=prov,
                                 refusal=f"its config cannot be read here ({type(e).__name__}: {e})"[:400])
    q = getattr(config, "quantization_config", None)
    if q:
        method = q.get("quant_method") if isinstance(q, dict) else getattr(q, "quant_method", "?")
        return _refused(name, config, f"a pre-quantized checkpoint ({method}): adapters over it are not supported", prov)
    text = getattr(config, "text_config", None)
    text_tower_of = None
    if text is not None and text is not config:
        text_tower_of = getattr(config, "model_type", None)
        config_t = text
    else:
        config_t = config
    experts = {k: getattr(config_t, k) for k in EXPERT_KEYS if (getattr(config_t, k, 0) or 0) > 1}
    if experts:
        return _refused(name, config, f"routed experts ({experts}): a mixture-of-experts model is the experts4bit "
                                      "backend's", prov)
    try:
        from accelerate import init_empty_weights

        with init_empty_weights():
            tree = AutoModelForCausalLM.from_config(config_t, dtype=torch.bfloat16, trust_remote_code=trust_remote_code)
    except (ValueError, KeyError, TypeError, ImportError) as e:
        return _refused(name, config, f"transformers cannot build it as a causal LM ({type(e).__name__}: {e})"[:400],
                        prov)
    H = _int(config_t, "hidden_size", "n_embd", "d_model")
    nh = _int(config_t, "num_attention_heads", "n_head")
    nkv = _int(config_t, "num_key_value_heads", default=nh) or nh
    hd = _int(config_t, "head_dim") or (H // nh if nh else 0)
    inter = _int(config_t, "intermediate_size", "n_inner", "ffn_dim")
    layers = [(int(m.group(1)), mod) for n, mod in tree.named_modules() if (m := _LAYER.search(n))]
    if not layers:
        return _refused(name, config, "no decoder layers found (modules named '...layers.<i>')", prov)
    per_layer, layer_param_ids = [], set()
    for _i, layer in sorted(layers, key=lambda x: x[0]):
        found = []
        for n, mod in layer.named_modules():
            if isinstance(mod, torch.nn.Linear):
                found.append(Linear(n, mod.in_features, mod.out_features, mod.bias is not None,
                                    classify(n, mod.in_features, mod.out_features, hidden=H, inter=inter,
                                             q_width=nh * hd, kv_width=nkv * hd)))
        per_layer.append(tuple(found))
        layer_param_ids |= {id(p) for p in layer.parameters()}
    first = per_layer[0]
    uniform = all(lin == first for lin in per_layer)       # names, shapes, biases and roles must all agree
    linear_params = sum(lin.in_features * lin.out_features for layer in per_layer for lin in layer)
    emb, head = tree.get_input_embeddings(), tree.get_output_embeddings()
    tied = bool(getattr(config_t, "tie_word_embeddings", getattr(config, "tie_word_embeddings", False))) \
        and head is not None and emb is not None
    emb_n = int(emb.weight.numel()) if emb is not None else 0
    head_n = 0 if tied or head is None else int(head.weight.numel())
    total = sum(p.numel() for p in tree.parameters()) - (int(head.weight.numel()) if tied and head.weight is not
                                                          emb.weight else 0)
    other = int(total - linear_params - emb_n - head_n)
    unclassified = tuple(sorted({f"layers.{i}.{lin.name} ({lin.in_features}->{lin.out_features})"
                                 for i, layer in enumerate(per_layer) for lin in layer if lin.role is None}))
    cls = type(tree)
    impls = tuple(k for k, attr in (("sdpa", "_supports_sdpa"), ("flash_attention_2", "_supports_flash_attn"),
                                    ("flex_attention", "_supports_flex_attn")) if getattr(cls, attr, False)) + ("eager",)
    notes, valid = [], impls
    if getattr(config_t, "attn_logit_softcapping", None):
        valid = tuple(i for i in impls if i != "sdpa")
        notes.append(f"attention logit softcapping ({config_t.attn_logit_softcapping}): transformers' SDPA path does "
                     "not apply it, so only eager, flash or flex attention keep the model's semantics")
    if getattr(config_t, "final_logit_softcapping", None):
        notes.append(f"final logit softcapping ({config_t.final_logit_softcapping}) after the LM head")
    if getattr(config_t, "sliding_window", None) and getattr(config_t, "use_sliding_window", True) is not False:
        notes.append(f"sliding-window attention ({config_t.sliding_window} tokens) in some or all layers")
    try:
        from experts4bit_qlora.engines.chunked_lm_loss import chunked_lm_loss_refusal
    except ImportError:
        chunked = "experts4bit-qlora's chunked LM loss is not installed"
    else:
        chunked = chunked_lm_loss_refusal(tree)
    refusal = None
    if not any(lin.role for lin in first):
        refusal = "no decoder-layer linear classifies as an attention or MLP projection: nothing to adapt"
    archs = getattr(config, "architectures", None) or [None]
    prov.update(roles="classified by shape against the config's hidden/intermediate/head widths; names break ties",
                attention="the model class's _supports_* attributes", chunked_loss="experts4bit-qlora "
                "engines.chunked_lm_loss.chunked_lm_loss_refusal on the meta tree")
    return DenseTopology(
        model=name, model_type=getattr(config_t, "model_type", None), architecture=archs[0],
        revision=getattr(config, "_commit_hash", None), refusal=refusal, n_layers=len(per_layer), hidden_size=H,
        intermediate_size=inter, heads=nh, kv_heads=nkv, head_dim=hd, vocab_size=_int(config_t, "vocab_size"),
        max_positions=_int(config_t, "max_position_embeddings", default=0) or None, tied_embeddings=tied,
        layer_linears=first, uniform=uniform, linear_params=linear_params, embedding_params=emb_n,
        head_params=head_n, other_params=other, unclassified=unclassified, attention_impls=impls,
        attention_valid_impls=valid, attention_notes=tuple(notes), chunked_loss_refusal=chunked,
        text_tower_of=text_tower_of, provenance=prov)


def refusal(topology) -> str | None:
    """Why this backend can plan nothing for ``topology``, or None. A description from another backend is not ours."""
    if not isinstance(topology, DenseTopology):
        return "not a model the dense backend described"
    return topology.refusal


def summary(topology) -> dict:
    """The plan's ``model`` section."""
    t = topology
    return {"model": t.model, "model_type": t.model_type, "revision": t.revision, "summary": t.summary(),
            "n_layers": t.n_layers, "linear_params": t.linear_params, "embedding_params": t.embedding_params,
            "head_params": t.head_params, "targets": t.targets, "unclassified": list(t.unclassified),
            "attention_valid_impls": list(t.attention_valid_impls), "chunked_loss_refusal": t.chunked_loss_refusal,
            "refusal": t.refusal, "provenance": t.provenance}


# ------------------------------------------------------------------------------------------------------------------ planning
@dataclass
class Status:
    available: bool
    reason: str = ""
    versions: dict = field(default_factory=dict)
    #: capability -> (usable here, why)
    kernels: dict = field(default_factory=dict)


def _version(dist):
    try:
        from importlib.metadata import version

        return version(dist)
    except Exception:  # noqa: BLE001 - absence is the answer
        return None


@functools.lru_cache(maxsize=1)
def _late_bound_reason() -> str | None:
    """Why offloaded Linear4bit weights would not be freed in training here (experts4bit-qlora's own check), or None."""
    try:
        from experts4bit_qlora.engines.dense_offload import late_bound_4bit_refusal
    except ImportError:
        return ("this experts4bit-qlora cannot say whether offloaded Linear4bit weights are freed in training "
                "(late_bound_4bit_refusal, experts4bit-qlora#1312)")
    return late_bound_4bit_refusal()


def _flash_installed() -> bool:
    import importlib.util

    return importlib.util.find_spec("flash_attn") is not None


def _offload_plan():
    try:
        from experts4bit_qlora.engines.dense_offload import offload_plan
    except ImportError:
        return None
    return offload_plan


def probe(gpu) -> Status:
    """Is this backend usable, and which of its capabilities hold on ``gpu``."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as e:
        return Status(False, f"torch and transformers are needed ({e})")
    versions = {k: _version(k) for k in ("torch", "transformers", "bitsandbytes", "peft", "experts4bit-qlora")}
    kernels = {}
    if _offload_plan() is None:
        kernels["dense_offload_pricing"] = (False, "this experts4bit-qlora cannot price dense offload "
                                                   "(engines.dense_offload.offload_plan, experts4bit-qlora#1312)")
    else:
        kernels["dense_offload_pricing"] = (True, "experts4bit-qlora engines.dense_offload.offload_plan")
    why = _late_bound_reason()
    kernels["late_bound_4bit"] = (why is None, why or "offloaded Linear4bit weights are released between their forward "
                                                     "and backward (the late-bound backward)")
    fa = _flash_installed()
    kernels["flash_attention_2"] = (fa and gpu is not None, "flash-attn installed" if fa else "flash-attn is not installed")
    return Status(True, "", versions, kernels)


def _attn_choice(topology, status):
    """The attention implementation a candidate uses: the first that keeps the model's semantics and runs here."""
    for impl in ("sdpa", "flash_attention_2", "flex_attention", "eager"):
        if impl in topology.attention_valid_impls and (impl != "flash_attention_2"
                                                       or status.kernels.get(impl, (False,))[0]):
            return impl
    return "eager"


def candidates(topology, workload, constraints, status) -> list:
    """Setups worth estimating, as dicts. Fields in ``constraints.fixed`` are not varied."""
    fixed = dict(constraints.fixed)
    known = {"base", "placement", "attn_impl", "loss_chunk", "r", "alpha", "adapter_dtype", "targets"}
    unknown = set(fixed) - known
    if unknown:
        raise ValueError(f"unknown dense setup fields fixed by the caller: {sorted(unknown)}")
    base = {**DEFAULT_ADAPTER, "targets": list(ROLES), "attn_impl": _attn_choice(topology, status),
            "loss_chunk": DEFAULT_CHUNK if topology.chunked_loss_refusal is None else 0}
    if isinstance(fixed.get("targets"), str):
        fixed["targets"] = [t.strip() for t in fixed["targets"].split(",") if t.strip()]
    out = []
    for b in [fixed["base"]] if "base" in fixed else list(BASES):
        for p in [fixed["placement"]] if "placement" in fixed else list(PLACEMENTS):
            if p == "stream" and b == "bf16" and "placement" not in fixed:
                continue                       # dominated: twice NF4's link bytes, and resident NF4 fits wherever it would
            out.append({**base, **fixed, "base": b, "placement": p})
    return out


def _linear_bytes(n: int, base: str) -> tuple:
    """(bytes that would stream, bytes that stay on the device) for a frozen linear of ``n`` weights. NF4: bitsandbytes
    Linear4bit with nested statistics -- 4-bit codes, an 8-bit absmax per 64 weights, an fp32 second-level scale per 256
    absmax blocks; the codes are the packed parameter, the statistics its quant_state (never streamed)."""
    if base == "nf4":
        return n // 2, n // 64 + (n // (64 * 256)) * 4
    return 2 * n, 0


def setup_refusals(topology, setup, workload) -> tuple:
    out = []
    if setup.get("base") not in BASES:
        out.append(f"base must be one of {BASES}, got {setup.get('base')!r} (8-bit bases are not priced yet)")
    if setup.get("placement") not in PLACEMENTS:
        out.append(f"placement must be one of {PLACEMENTS}, got {setup.get('placement')!r}")
    if setup.get("adapter_dtype") not in ADAPTER_BYTES:
        out.append(f"adapter_dtype must be one of {tuple(ADAPTER_BYTES)}, got {setup.get('adapter_dtype')!r}")
    if not isinstance(setup.get("r"), int) or setup["r"] < 1:
        out.append("r must be a positive integer")
    roles = setup.get("targets") or []
    if not roles or set(roles) - set(ROLES):
        out.append(f"targets must be a non-empty subset of {ROLES}")
    elif not any(lin.role in roles for lin in topology.layer_linears):
        out.append(f"no decoder-layer linear has the roles {roles}")
    if not topology.uniform:
        out.append("decoder layers carry different linears; pricing assumes identical layers")
    impl = setup.get("attn_impl")
    if impl not in topology.attention_valid_impls:
        out.append(f"attention {impl!r} would change this model's semantics or is not declared by its class "
                   f"({'; '.join(topology.attention_notes) or 'declared: ' + ', '.join(topology.attention_impls)})")
    elif impl == "flash_attention_2" and not _flash_installed():
        out.append("flash_attention_2: flash-attn is not installed")
    if setup.get("loss_chunk") and topology.chunked_loss_refusal:
        out.append(f"chunked loss: {topology.chunked_loss_refusal}")
    if topology.max_positions and workload.seq_len > topology.max_positions:
        out.append(f"seq {workload.seq_len} exceeds the model's {topology.max_positions} positions")
    if setup.get("placement") == "stream":
        if _offload_plan() is None:
            out.append("streaming: this experts4bit-qlora cannot price dense offload (offload_plan, "
                       "experts4bit-qlora#1312)")
        elif setup.get("base") == "nf4" and _late_bound_reason() is not None:
            out.append(f"streaming: {_late_bound_reason()}")
    return tuple(out)


def estimate(topology, setup, workload):
    """``(lines, unmodelled, refusals)``; ``lines`` are ``(name, where, bytes, basis, detail)``."""
    refusals = setup_refusals(topology, setup, workload)
    if refusals:
        return [], (), refusals
    t = topology
    T = workload.seq_len * workload.micro_batch
    H, inter, L, V = t.hidden_size, t.intermediate_size, t.n_layers, t.vocab_size
    q, kv = t.heads * t.head_dim, 2 * t.kv_heads * t.head_dim
    base, ab, r = setup["base"], ADAPTER_BYTES[setup["adapter_dtype"]], setup["r"]
    roles = set(setup["targets"])
    lines, unmodelled = [], []
    streamed_layer, stays_layer, lora_layer = [], 0, 0
    for lin in t.layer_linears:
        n = lin.in_features * lin.out_features
        moving, stays = _linear_bytes(n, base)
        streamed_layer.append(moving)
        stays_layer += stays
        if lin.role in roles:
            lora_layer += r * (lin.in_features + lin.out_features)
    weights = L * sum(streamed_layer)
    qdesc = "NF4 with nested statistics" if base == "nf4" else "bf16"
    if setup["placement"] == "device":
        lines.append(("frozen decoder linears", "device", weights + L * stays_layer, "derived",
                      f"{L} layers x {len(t.layer_linears)} linears in {qdesc}"))
    else:
        frozen_layer = [(m, 2, False, True) for m in streamed_layer]
        kept_frozen = _offload_plan()([frozen_layer] * L, pin=True, train_prefetch=True)["stays_on_device"]
        layer = list(frozen_layer)
        layer += [(r * lin.in_features * ab, 2, True, True) for lin in t.layer_linears if lin.role in roles]
        layer += [(lin.out_features * r * ab, 2, True, True) for lin in t.layer_linears if lin.role in roles]
        plan = _offload_plan()([layer] * L, pin=True, train_prefetch=True)
        lines.append(("frozen decoder linears, two layers staged", "device", plan["resident_slots"], "derived",
                      "experts4bit-qlora offload_plan: the layer in use and its prefetched neighbour"))
        if kept_frozen:
            lines.append(("frozen decoder linears, kept on device", "device", kept_frozen, "derived",
                          "experts4bit-qlora offload_plan: frozen codes below its streaming threshold stay on device"))
        if stays_layer:
            lines.append(("frozen decoder linears, quantization statistics", "device", L * stays_layer, "derived",
                          "absmax and second-level scales stay on the device"))
        lines.append(("frozen decoder linears, pinned host homes", "host", plan["host_reserved"], "derived",
                      f"{plan['streamed'] / 1e9:.2f} GB of {qdesc}, each pinned request rounded up to a power of two "
                      "(experts4bit-qlora offload_plan)"))
        lines.append(("frozen layers streamed per micro-batch", "link", plan["link_per_microbatch"], "derived",
                      "each layer copied for its forward and again for its backward (an upper bound)"))
    lines.append(("embeddings + lm head (bf16)", "device", 2 * (t.embedding_params + t.head_params), "derived",
                  "input embeddings" + (" tied to the head" if t.tied_embeddings else " and the untied lm head")))
    other = 2 * t.other_params                  # the topology already includes linear biases here
    if other:
        lines.append(("norms, biases and other parameters (bf16)", "device", other, "derived", ""))
    lora = L * lora_layer
    lines.append(("LoRA adapters", "device", lora * ab, "derived",
                  f"r={r}, {setup['adapter_dtype']}, on {', '.join(sorted(roles))}: {lora:,} parameters"))
    lines.append(("adapter gradients", "device", lora * ab, "derived", "one gradient per trainable parameter"))
    lines.append(("optimizer state (adamw)" if workload.optimizer == "adamw" else f"optimizer state ({workload.optimizer})",
                  "device", int(lora * (2 * ab if workload.optimizer == "adamw" else 2 + 8 / 256)), "derived",
                  "exp_avg + exp_avg_sq per trainable parameter" if workload.optimizer == "adamw" else "8-bit states"))
    boundaries = L * H * 2
    layer_work = 2 * 2 * (H + q + kv + q + H + H + 2 * inter + inter + H)
    if setup["loss_chunk"]:
        try:
            from experts4bit_qlora.engines.chunked_lm_loss import chunked_loss_bytes

            loss_bytes = chunked_loss_bytes(T, V, hidden=H, chunk=setup["loss_chunk"])
            loss_detail = f"experts4bit-qlora chunked_loss_bytes: {setup['loss_chunk']} tokens of logits at a time"
        except ImportError:
            loss_bytes = min(T, setup["loss_chunk"]) * V * 10 + T * H * 2
            loss_detail = "one chunk of logits at 10 B per logit (this experts4bit-qlora cannot state it)"
    else:
        # Generic full-logit CE backward overlaps saved log-softmax, NLL grad and softmax grad:
        # three distinct fp32 [T,V] tensors. DQ7's CPU/CUDA operator census identifies the missing buffer.
        loss_bytes, loss_detail = T * V * 12, "full-logit backward: saved fp32 log-softmax + fp32 NLL grad + fp32 logits grad"
    per_token = ACTIVATION_COEFFICIENT * (boundaries + layer_work)
    if T * layer_work > loss_bytes:
        act = int(T * per_token)
        how = "checkpointed layer inputs + one layer's recompute"
    else:
        act = int(T * ACTIVATION_COEFFICIENT * boundaries) + loss_bytes
        how = "checkpointed layer inputs + the loss's workspace"
    lines.append(("activations", "device", act, "heuristic",
                  f"{how} at {T:,} tokens; per-token terms x {ACTIVATION_COEFFICIENT:.3f}, DQ4's measured slope over "
                  f"the formula's (Qwen3-32B, fp32 adapters); loss: {loss_detail}"))
    if setup["attn_impl"] == "eager":
        lines.append(("eager attention scores", "device", workload.micro_batch * t.heads * workload.seq_len ** 2 * 6,
                      "heuristic", "one layer's scores and probabilities (bf16) and their gradient, materialized"))
    if base == "nf4":
        lines.append(("dequantization transient", "device", 2 * max(lin.in_features * lin.out_features
                                                                   for lin in t.layer_linears), "heuristic",
                      "one weight dequantized to bf16 for its matmul"))
    unmodelled += ["CUDA context, cuBLAS/attention workspaces and allocator fragmentation (the planner adds context "
                   "and reserve)", "load-time transients (quantizing on load; host page cache)",
                   "bitsandbytes and PEFT internal buffers"]
    return lines, tuple(unmodelled), ()


def speed_rank(setup: dict) -> tuple:
    return (setup.get("base") != "bf16", setup.get("placement") == "stream", setup.get("attn_impl") == "eager")


def label(setup: dict) -> str:
    roles = setup.get("targets") or ()
    where = "streamed from host" if setup.get("placement") == "stream" else "resident"
    return (f"dense {setup.get('base')} base, {where}, LoRA r{setup.get('r')} on "
            f"{'all projections' if set(roles) == set(ROLES) else '+'.join(roles)}, {setup.get('attn_impl')}"
            + (", chunked loss" if setup.get("loss_chunk") else ""))


def run_tag(setup: dict) -> str:
    return f"dense-{setup['base']}-{setup['placement']}"


def executor(kind: str):
    """Return the executor, which requires a typed development opt-in pending DQ7."""
    if kind == "train":
        from .dense_train import run

        return run
    return None


def planned_only_reason(kind: str) -> str:
    if kind == "train":
        return ("dense execution is a development executor pending the registered CUDA proof (DQ7); "
                "use --allow-development-executor (Constraints.allow_development_executor=True) to run it; "
                "see docs/DENSE.md")
    return f"the dense backend does not execute {kind!r} workloads"


def residency(setup: dict) -> str | None:
    return {"device": "device", "stream": "host"}.get(setup.get("placement"))


# Conservative admission policy after DQ7, separate from every estimator coefficient and reserve fraction.
DQ7_STREAM_HEADROOM = 2_400_000_000  # decimal bytes, rounded above the observed maximum 2,395,904,403 B
DQ7_RESULTS = "https://github.com/pjordanandrsn/experts4bit-qlora/blob/main/bench/dq7/RESULTS-dq7.md"


def planning_warnings(topology) -> list:
    return ["Dense estimates are not calibrated out of sample: DQ7 measured driver device use up to 2.4 GB "
            "above the full streamed plan and found Llama allocator underestimates. Streamed plans require "
            "at least max(2.4 GB, 20% of the full device plan) headroom, even if less is requested. This empirical refusal "
            "margin is not a general fit guarantee; the development executor opt-in remains. Evidence: " + DQ7_RESULTS]


def memory_policy(topology, gpu, setup, workload, constraints, allocator_bytes):
    """Explicit registered hypothesis pricing; invalid selection refuses rather than falling back."""
    from ..dense_policy import lines

    return lines(topology, gpu, setup, workload, constraints, allocator_bytes)


def minimum_headroom(setup, device_bytes=0) -> int:
    """Mandatory streamed admission margin; never an allocator estimate or a calibrated reserve."""
    return max(DQ7_STREAM_HEADROOM, -(-device_bytes//5)) if setup.get("placement") == "stream" else 0


def scope_warnings(topology, setup, workload) -> list:
    if setup.get("placement") != "stream":
        return []
    shape = (topology.model_type, topology.hidden_size, topology.intermediate_size,
             topology.n_layers, topology.heads, topology.kv_heads, topology.head_dim, topology.vocab_size)
    shapes = {("qwen3", 5120, 17408, 40, 40, 8, 128, 151936),
              ("qwen3", 5120, 25600, 64, 64, 8, 128, 151936),
              ("llama", 4096, 14336, 32, 32, 8, 128, 128256)}
    upper = 2048 if shape == ("qwen3", 5120, 25600, 64, 64, 8, 128, 151936) else 4096
    lower = 2048 if upper == 2048 else 512
    if (shape not in shapes or not lower <= workload.seq_len <= upper
            or workload.micro_batch != 1 or workload.grad_accum != 1
            or setup.get("base") != "nf4" or setup.get("adapter_dtype") != "fp32"
            or setup.get("r") != 16 or setup.get("alpha") != 32 or set(setup.get("targets", ())) != set(ROLES)
            or setup.get("attn_impl") != "sdpa" or workload.optimizer != "adamw"
            or setup.get("loss_chunk") != (0 if topology.model_type == "llama" else 512)):
        return ["Streamed plan is outside DQ7's measured subject/sequence/recipe range (Qwen3-14B and Llama-3.1-8B "
                "512-4096, Qwen3-32B 2048 only; micro-batch1/accum1, NF4 fp32 r16 SDPA AdamW). "
                "The empirical headroom floor cannot establish fit beyond that range."]
    return []


def plan_warnings(setup: dict, gpu) -> list:
    if setup.get("placement") == "stream" and gpu.pcie_width_current.value and gpu.pcie_width_max.value and \
            gpu.pcie_width_current.value < gpu.pcie_width_max.value:
        return [f"frozen layers stream over PCIe, and the driver reports the link at x{gpu.pcie_width_current.value} "
                f"of x{gpu.pcie_width_max.value} right now"]
    return []


def relaxed_candidates(topology, workload, constraints, status) -> list:
    from dataclasses import replace

    out = []
    if constraints.fixed.get("placement") == "device":
        relaxed = replace(constraints, fixed={k: v for k, v in constraints.fixed.items() if k != "placement"})
        out += [("allow streaming the frozen layers from host memory", s)
                for s in candidates(topology, workload, relaxed, status) if s["placement"] == "stream"]
    if constraints.fixed.get("base") == "bf16":
        relaxed = replace(constraints, fixed={k: v for k, v in constraints.fixed.items() if k != "base"})
        out += [("allow an NF4 base (QLoRA)", s) for s in candidates(topology, workload, relaxed, status)
                if s["base"] == "nf4" and s["placement"] == "device"]
    return out


def isolation_refusal(topology) -> str | None:
    if topology.unclassified:
        return (f"{len(topology.unclassified)} decoder-layer linears are unclassified; a token mixer among them may carry "
                "state across packed examples")
    return None


def policy_notes(topology, constraints) -> list:
    out = []
    if topology.unclassified:
        out.append(f"{len(topology.unclassified)} unclassified decoder-layer linears stay frozen and take no adapter")
    for n in topology.attention_notes:
        out.append(f"attention: {n}")
    return out


def explain(sel, feasible, infeasible, budget, status, constraints, workload) -> list:
    s, out = sel.setup, []
    every = feasible + infeasible
    bf16 = min((c for c in every if c.setup.get("base") == "bf16" and c.setup.get("placement") == "device"),
               key=lambda c: c.device_bytes, default=None)
    resident = min((c for c in every if c.setup.get("base") == s["base"] and c.setup.get("placement") == "device"),
                   key=lambda c: c.device_bytes, default=None)
    if s["base"] == "bf16":
        out.append("bf16 base: the frozen weights stay exact (LoRA, not QLoRA)")
    elif bf16 is not None and bf16 not in feasible:
        out.append(f"NF4 base (QLoRA): a bf16 base needs {bf16.device_bytes / GiB:.2f} GiB + "
                   f"{budget['headroom'] / GiB:.2f} GiB headroom, over the {budget['device'] / GiB:.2f} GiB budget"
                   if not bf16.rejected or bf16.rejected[0].startswith("device") else
                   f"NF4 base (QLoRA): {'; '.join(bf16.rejected)}")
    else:
        out.append(f"NF4 base (QLoRA), by objective {constraints.objective} or a fixed field")
    T = workload.seq_len * workload.micro_batch
    if s["placement"] == "stream":
        need = f"{resident.device_bytes / GiB:.2f} GiB" if resident is not None else "more than the budget"
        out.append(f"frozen layers streamed from pinned host memory, two staged at a time: resident needs {need} + "
                   "headroom, over the budget")
        if T < STREAM_BREAK_EVEN_TOKENS:
            out.append(f"{T:,} tokens per micro-batch is under the measured break-even (~{STREAM_BREAK_EVEN_TOKENS:,} on "
                       "PCIe 4.0 x16, DQ1): streaming will slow each step, by an amount not modelled")
    else:
        out.append(f"frozen layers resident: {sel.device_bytes / GiB:.2f} GiB estimated + {budget['headroom'] / GiB:.2f} "
                   f"GiB headroom fits the {budget['device'] / GiB:.2f} GiB budget")
    lora = next((ln for ln in sel.lines if ln.name == "LoRA adapters"), None)
    if lora:
        out.append(f"LoRA r{s['r']} alpha {s['alpha']} ({s['adapter_dtype']}) on every classified projection role: "
                   f"{lora.detail.split(': ')[-1]}" if set(s["targets"]) == set(ROLES) else f"LoRA: {lora.detail}")
    out.append(f"attention {s['attn_impl']}" + ("" if s["attn_impl"] != "eager" else
                                                 ": the scores are materialized (priced)"))
    out.append(f"loss chunked {s['loss_chunk']} tokens at a time (experts4bit-qlora chunked_lm_loss)" if s["loss_chunk"]
               else "loss over full logits (experts4bit-qlora's chunked loss does not cover this model class)")
    if constraints.objective == "speed":
        out += [f"ordering, {k}: {v}" for k, v in SPEED_EVIDENCE.items()]
    if constraints.fixed:
        out.append(f"fixed by the caller: {constraints.fixed}")
    return out
