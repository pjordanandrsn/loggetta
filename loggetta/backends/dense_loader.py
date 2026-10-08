"""Dense checkpoint loading without a whole bf16 model or resident streamed base.

Transformers supplies the meta tree. Safetensors supplies one checkpoint tensor at a time; bitsandbytes quantizes
each decoder linear. For streaming, packed weights remain on CPU while their small quantization states stay on the
execution device, ready for experts4bit-qlora's dense-offload handles. No checkpoint tensor is synthesized.
"""
from __future__ import annotations

from pathlib import Path


def snapshot(model, revision=None):
    directory = Path(model).expanduser()
    if directory.is_dir():
        return directory
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model, revision=revision,
                                  allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"]))


def load_base(model, setup, *, device="cuda", revision=None):
    """Return a frozen dense model and loader provenance. Each NF4 quantization uses one linear's bf16 weights."""
    import torch
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForCausalLM

    from . import dense

    directory = snapshot(model, revision)
    from experts4bit_qlora.engines.dense_offload import MIN_BYTES
    config = AutoConfig.from_pretrained(directory, trust_remote_code=False)
    topology = dense.describe(config)
    if topology.refusal or not topology.uniform:
        raise ValueError(topology.refusal or "non-uniform decoder layers cannot execute this dense estimate")
    text_config = getattr(config, "text_config", None) or config
    with init_empty_weights():
        tree = AutoModelForCausalLM.from_config(text_config, dtype=torch.bfloat16,
                                              attn_implementation=setup["attn_impl"], trust_remote_code=False)
    files = sorted(directory.glob("*.safetensors"))
    if not files:
        raise ValueError("dense execution needs a safetensors checkpoint; no safetensors files were found")
    locations = {}
    for file in files:
        with safe_open(file, framework="pt", device="cpu") as shard:
            for name in shard.keys():
                if name in locations:
                    raise ValueError(f"checkpoint tensor occurs in multiple shards: {name}")
                locations[name] = file

    def key_for(name):
        if name in locations:
            return name
        # A multimodal checkpoint can prefix its text tower. Accept only a unique, exact suffix match.
        matches = [key for key in locations if key.endswith("." + name)]
        if len(matches) != 1:
            raise ValueError(f"checkpoint has no unique tensor for {name}")
        return matches[0]

    def read(name):
        key = key_for(name)
        with safe_open(locations[key], framework="pt", device="cpu") as shard:
            return shard.get_tensor(key)

    expected = dict(tree.state_dict())
    linears = {}
    for layer_name, layer in tree.named_modules():
        if dense._LAYER.search(layer_name):
            for relative, module in layer.named_modules():
                if isinstance(module, torch.nn.Linear):
                    linears[f"{layer_name}.{relative}"] = module
    input_names = [name for name, module in tree.named_modules() if module is tree.get_input_embeddings()]
    head_names = [name for name, module in tree.named_modules() if module is tree.get_output_embeddings()]
    aliases = {}
    if topology.tied_embeddings and input_names and head_names:
        aliases[head_names[0] + ".weight"] = input_names[0] + ".weight"
        for head, embedding in aliases.items():
            explicit = head in locations or any(key.endswith("." + head) for key in locations)
            if explicit and not torch.equal(read(head), read(embedding)):
                raise ValueError("checkpoint has different weights for embeddings and a configured tied head")
    # Check names and shapes before putting any checkpoint weight on the execution device.
    keys = {}
    for name, tensor in expected.items():
        source = aliases.get(name, name)
        key = key_for(source)
        with safe_open(locations[key], framework="pt", device="cpu") as shard:
            shape = tuple(shard.get_slice(key).get_shape())
            dtype = shard.get_tensor(key).dtype
        if shape != tuple(tensor.shape):
            raise ValueError(f"checkpoint shape differs for {name}: {shape} versus {tuple(tensor.shape)}")
        if tensor.is_floating_point() and dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            raise ValueError(f"checkpoint tensor {name} uses {dtype}; packed or scaled checkpoint formats are refused")
        keys[name] = source
    quantized, handled = [], set()
    if setup["base"] == "nf4":
        if torch.device(device).type != "cuda":
            raise ValueError("dense NF4 execution requires CUDA; the CPU correctness path uses bf16")
        import bitsandbytes as bnb

        for name, linear in linears.items():
            if type(linear) is not torch.nn.Linear:
                raise ValueError(f"NF4 replacement is unverified for the custom linear class at {name}")
            weight = read(keys[name + ".weight"]).to(torch.bfloat16)
            quant = bnb.nn.Linear4bit(linear.in_features, linear.out_features, bias=linear.bias is not None,
                                     compute_dtype=torch.bfloat16, compress_statistics=True, quant_type="nf4",
                                     device="meta")
            quant.weight = bnb.nn.Params4bit(weight, requires_grad=False, compress_statistics=True,
                                            quant_type="nf4", blocksize=64)
            if linear.bias is not None:
                quant.bias = torch.nn.Parameter(read(keys[name + ".bias"]).to(torch.bfloat16), requires_grad=False)
                handled.add(name + ".bias")
            quant = quant.to(device)
            if setup["placement"] == "stream" and quant.weight.numel() * quant.weight.element_size() >= MIN_BYTES:
                # Keep quant_state on the GPU. Only packed codes move to the offload handles' CPU homes.
                quant.weight.data = quant.weight.data.cpu()
            parent_name, leaf = name.rsplit(".", 1)
            setattr(tree.get_submodule(parent_name), leaf, quant)
            handled.add(name + ".weight")
            quantized.append(name)
            del weight
        tree.is_loaded_in_4bit = True
    elif setup["base"] != "bf16":
        raise ValueError(f"unsupported dense base {setup['base']!r}")
    for name in expected:
        if name in handled or name in aliases:
            continue
        tensor = read(keys[name])
        dtype = torch.bfloat16 if tensor.is_floating_point() else tensor.dtype
        layer_weight = name.endswith(".weight") and name.rsplit(".", 1)[0] in linears
        streamable = layer_weight and tensor.numel() * torch.empty((), dtype=dtype).element_size() >= MIN_BYTES
        destination = "cpu" if setup["placement"] == "stream" and streamable else device
        set_module_tensor_to_device(tree, name, destination, value=tensor.to(dtype))
    tree.tie_weights()
    # Non-persistent buffers (e.g. rotary frequencies) are initialized by the model class rather than checkpointed.
    for name, buffer in list(tree.named_buffers()):
        if buffer.is_meta:
            raise ValueError(f"model left a meta buffer after checkpoint loading: {name}")
        if buffer.device != torch.device(device):
            set_module_tensor_to_device(tree, name, device, value=buffer)
    if any(p.is_meta for p in tree.parameters()):
        raise ValueError("model left meta parameters after checkpoint loading")
    tree.requires_grad_(False)
    return tree, topology, {"checkpoint": str(directory), "revision": revision,
                            "quantized_linears": quantized, "text_tower_of": topology.text_tower_of,
                            "loading": "one safetensors tensor at a time; one linear per NF4 quantization"}
