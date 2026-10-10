"""``dense_train.frozen_digest`` takes a rank-0 frozen parameter, and every ranked digest is what it was before the fix
(a byte view of a rank-0 tensor used to refuse; flattening first leaves ranked bytes unchanged)."""
import hashlib

import pytest

torch = pytest.importorskip("torch")
from loggetta.backends.dense_train import frozen_digest  # noqa: E402


def _model(dtype, scalar=None):
    """Two decoder layers (``layers.0``, ``layers.1``) with deterministic, frozen, ranked parameters; optionally a
    rank-0 parameter ``scale`` on the first."""
    torch.manual_seed(0)
    root = torch.nn.Module()
    root.layers = torch.nn.ModuleList()
    for _ in range(2):
        layer = torch.nn.Module()
        layer.proj = torch.nn.Linear(4, 3, bias=True).to(dtype)
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
    torch.float32: ("e027bbab16813fcb313f8447ac06828e26adbdc23f3f24add17cd6de8fd04ff0",
                    "5217c4c964b6fdb8f2aa3ac43bbc29a586284ce6166cb9e79b2f6200d3ad177b"),
    torch.bfloat16: ("cf902b1f7e1a07bf355e223d46241bcaa51756330c84c762e80a029541dd2a2c",
                     "a5c6348b559584a3756849adb9a28d1c907581cc3d0d1a0975ce4f2810b93240"),
    torch.float16: ("e05418af085ee07539fa184bdd8ba372c5a299a65152a1b8c1dc6c8561a93eb4",
                    "9ac4e6f25d4f6683aea065985d4f47628e06ed4b0a96e145e8eb54209e2e2262"),
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
