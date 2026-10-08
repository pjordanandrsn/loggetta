"""The dense backend: decoder-only models without experts, for LoRA/QLoRA adapter training (planned in a later step).

This module is the planner's side of that backend. Today it only DESCRIBES a model: from its config and a module tree
built on the ``meta`` device (no weights read), it finds the decoder layers and classifies every linear in them by its
structural role, so that LoRA targets are chosen by shape rather than by a per-family list of names:

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

import re
from dataclasses import asdict, dataclass, field

NAME = "dense"
#: nothing is planned yet: describing comes first (adapter training follows in a later release)
WORKLOADS = ()
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
    uniform = all(tuple((x.name, x.role) for x in lin) == tuple((x.name, x.role) for x in first) for lin in per_layer)
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
