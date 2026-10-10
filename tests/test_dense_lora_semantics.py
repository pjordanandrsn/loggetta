"""Actual PEFT/e4b adapter bridge, CPU only; the arithmetic/equality bars were fixed before execution.

Both scale AFTER B(A(x)); PEFT's zero dropout is Identity. Native evaluates delta before the independent frozen base,
PEFT evaluates base first. For fp32 inputs/base/adapters, casts are no-ops and native's unit-scaling skip is exact;
bitwise outputs, input/adapter gradients and accumulated AdamW updates/state are required, with no tolerance.

Mixed bf16 base/input + fp32 adapters differ for plain PEFT Linear: native rounds delta BEFORE adding, PEFT adds in
fp32 then rounds the result. PEFT's bnb wrapper instead rounds delta before adding when autocast is off. The fixed
base=1, delta=0.003907 counterexample pins that distinction instead of declaring a tolerance or an NF4 result.
"""
import copy
from collections import OrderedDict

import pytest
import torch
from peft import LoraConfig, get_peft_model
from peft.tuners.lora.bnb import Linear4bit as PeftLinear4bit
from torch import nn
from torch.utils.checkpoint import checkpoint

from experts4bit_qlora.lora import LoRALinear
from test_dense_accumulated_parity import _assert_equal

SHAPES = {"q": (64, 64), "k": (64, 32), "v": (64, 32), "o": (64, 64),
          "gate": (64, 160), "up": (64, 160), "down": (160, 64)}


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _pair(in_features, out_features, rank, alpha, nonzero_b, base_dtype=torch.float32):
    torch.manual_seed(23)
    base = nn.Linear(in_features, out_features, bias=False, dtype=base_dtype)
    native = LoRALinear(copy.deepcopy(base), rank, alpha, torch.float32)
    holder = nn.Sequential(OrderedDict(proj=copy.deepcopy(base)))
    reference = get_peft_model(holder, LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.0,
                                                 target_modules=["proj"], bias="none")).get_base_model().proj
    with torch.no_grad():
        if nonzero_b:
            native.lora_B.normal_(0, 0.1)
        reference.lora_A["default"].weight.copy_(native.lora_A)
        reference.lora_B["default"].weight.copy_(native.lora_B)
    assert isinstance(reference.lora_dropout["default"], nn.Identity)
    assert reference.active_adapters == ["default"] and not reference.merged
    assert not reference.lora_variant and reference.scaling["default"] == native.scaling
    return native, reference


def _run(module, adapters, in_features, out_features, checkpoint_mode):
    module.train()
    assert all(parameter.dtype == torch.float32 for parameter in adapters)
    base = module.base if isinstance(module, LoRALinear) else module.base_layer
    frozen = base.weight.detach().clone()
    optimizer = torch.optim.AdamW(adapters, lr=1e-3)
    generator = torch.Generator().manual_seed(42)
    records = []
    for _step in range(2):
        optimizer.zero_grad(set_to_none=True)
        microbatches = []
        for _microbatch in range(2):
            x = torch.randn(3, in_features, generator=generator, requires_grad=True)
            dy = torch.randn(3, out_features, generator=generator)
            y = module(x) if checkpoint_mode == "none" else checkpoint(
                module, x, use_reentrant=checkpoint_mode == "reentrant")
            (y * dy).sum().div(2).backward()
            assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in adapters)
            microbatches.append({"y": y.detach().clone(), "dx": x.grad.clone(),
                                 "dA": adapters[0].grad.clone(), "dB": adapters[1].grad.clone()})
        optimizer.step()
        records.append({"microbatches": microbatches, "adapters": [parameter.detach().clone() for parameter in adapters],
                        "optimizer": copy.deepcopy(optimizer.state_dict())})
    assert torch.equal(base.weight, frozen) and base.weight.grad is None
    return records


def _compare(native, reference, in_features, out_features, checkpoint_mode):
    got = _run(native, [native.lora_A, native.lora_B], in_features, out_features, checkpoint_mode)
    want = _run(reference, [reference.lora_A["default"].weight, reference.lora_B["default"].weight],
                in_features, out_features, checkpoint_mode)
    _assert_equal(got, want)


@pytest.mark.parametrize("role", SHAPES)
@pytest.mark.parametrize("rank,alpha", ((4, 4), (3, 6), (3, 2)))
@pytest.mark.parametrize("nonzero_b", (False, True))
@pytest.mark.parametrize("checkpoint_mode", ("none", "nonreentrant", "reentrant"))
def test_fp32_native_and_actual_peft_match_through_gradients_and_optimizer_state(
        role, rank, alpha, nonzero_b, checkpoint_mode):
    in_features, out_features = SHAPES[role]
    native, reference = _pair(in_features, out_features, rank, alpha, nonzero_b)
    _compare(native, reference, in_features, out_features, checkpoint_mode)


def test_a_scaling_mismatch_fails_the_same_equality_bar():
    native, reference = _pair(64, 32, 4, 4, True)
    native.scaling *= 2
    with pytest.raises(AssertionError, match="parity"):
        _compare(native, reference, 64, 32, "nonreentrant")


def test_mixed_bf16_base_and_fp32_adapters_have_distinct_rounding_contracts():
    native, reference = _pair(1, 1, 1, 1, True, torch.bfloat16)
    with torch.no_grad():
        native.base.weight.fill_(1)
        reference.base_layer.weight.fill_(1)
        native.lora_A.fill_(1)
        reference.lora_A["default"].weight.fill_(1)
        native.lora_B.fill_(0.003907)
        reference.lora_B["default"].weight.fill_(0.003907)
    x = torch.ones(1, 1, dtype=torch.bfloat16)
    got, want = native(x), reference(x)
    assert got.dtype == want.dtype == torch.bfloat16
    assert got.item() == 1.0 and want.item() == 1.0078125
    assert not torch.equal(got, want), "a mixed-precision matched-work comparison must normalize delta-add rounding"


def test_peft_bnb_wrapper_rounds_delta_before_add_with_autocast_off():
    """Exercise the actual wrapper forward over a frozen bf16 linear fixture; this tests no packed quantization."""
    assert not torch.is_autocast_enabled()
    native, plain = _pair(1, 1, 1, 1, True, torch.bfloat16)
    config = LoraConfig(r=1, lora_alpha=1, lora_dropout=0.0, bias="none")
    wrapped = PeftLinear4bit(copy.deepcopy(native.base), "default", config=config, r=1, lora_alpha=1)
    wrapped.lora_A["default"].float()
    wrapped.lora_B["default"].float()
    with torch.no_grad():
        for base in (native.base, plain.base_layer, wrapped.base_layer):
            base.weight.fill_(1)
        for a, b in ((native.lora_A, native.lora_B),
                     (plain.lora_A["default"].weight, plain.lora_B["default"].weight),
                     (wrapped.lora_A["default"].weight, wrapped.lora_B["default"].weight)):
            assert a.dtype == b.dtype == torch.float32
            a.fill_(1)
            b.fill_(0.003907)
    x = torch.ones(1, 1, dtype=torch.bfloat16)
    assert torch.equal(wrapped(x), native(x)) and wrapped(x).item() == 1.0
    assert plain(x).item() == 1.0078125
