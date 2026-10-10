"""``dense_train.frozen_digest`` takes a rank-0 frozen parameter, and every ranked digest is what it was before the fix
(a byte view of a rank-0 tensor used to refuse; flattening first leaves ranked bytes unchanged)."""
import hashlib

import pytest

torch = pytest.importorskip("torch")
from loggetta.backends.dense_train import frozen_digest  # noqa: E402


def _model(dtype, scalar=None):
    """Two decoder layers (``layers.0``, ``layers.1``) with frozen, ranked parameters from arithmetic sequences (no RNG,
    so the same on every platform and torch); optionally a rank-0 parameter ``scale`` on the first."""
    root = torch.nn.Module()
    root.layers = torch.nn.ModuleList()
    for i in range(2):
        layer = torch.nn.Module()
        layer.proj = torch.nn.Linear(4, 3, bias=True)
        layer.proj.weight = torch.nn.Parameter(((torch.arange(12, dtype=torch.float32).reshape(3, 4) - 5.5) * 0.37 + i)
                                               .to(dtype))
        layer.proj.bias = torch.nn.Parameter((torch.arange(3, dtype=torch.float32) * 0.25 - i).to(dtype))
        layer.norm = torch.nn.Module()
        layer.norm.weight = torch.nn.Parameter(torch.arange(5, dtype=torch.float32).to(dtype))
        root.layers.append(layer)
    if scalar is not None:
        root.layers[0].scale = torch.nn.Parameter(torch.tensor(scalar, dtype=dtype))
    for p in root.parameters():
        p.requires_grad_(False)
    return root


#: computed with the code before the fix (loggetta main at 2026-10-10); ranked bytes must not move
PINNED = {
    torch.float32: ("3148cf6a5672d757d383850a34a02ede6d2ac7f9620fa4d17d0a1ebc6f3bc0e3",
                    "e584ccbdf294c2019e94fc82739a0c227bf84e365b2a5321bf1562df1e3c9f14"),
    torch.bfloat16: ("859670580805c3dce841c3086412477c8537ebfe6e44418ac29dd85addd2a012",
                     "d6a2312d42dba7005a80d8a04398cd2476ce446f124be57ed7af7b302b48cf8f"),
    torch.float16: ("37a3e21602982bb38d5c0c5623e667eb08b433299688199e934946260fd527e1",
                    "7aa71db3baee3e5870988855e7eb03655190f05c4f92f00ef744788451c65580"),
}


@pytest.mark.parametrize("dtype", list(PINNED))
def test_ranked_digests_are_unchanged(dtype):
    assert frozen_digest(_model(dtype)) == dict(zip(("layers.0", "layers.1"), PINNED[dtype]))


def _independent(layer):
    """The digest from first principles: key, then the tensor's raw bytes (an integer view of the same width)."""
    ints = {1: torch.uint8, 2: torch.int16, 4: torch.int32}
    sha = hashlib.sha256()
    for key, t in sorted((k, p) for k, p in layer.named_parameters() if not p.requires_grad):
        sha.update(key.encode())
        sha.update(t.detach().reshape(1, -1).contiguous().view(ints[t.element_size()]).numpy().tobytes())
    return sha.hexdigest()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_a_rank0_parameter_is_digested(dtype):
    model = _model(dtype, scalar=2.5)
    got = frozen_digest(model)
    assert got["layers.0"] == _independent(model.layers[0])
    assert got["layers.1"] == PINNED[dtype][1]                     # the other layer is untouched
    other = _model(dtype, scalar=-2.5)
    assert frozen_digest(other)["layers.0"] != got["layers.0"]      # the scalar's value is covered
