"""Dense plans do not move with the grouped_nf4 MoE training terms (#43).

The terms live in the experts4bit backend's ``estimate``. This prices a matrix of dense setups (three Qwen3 shapes, four
card sizes, three workloads; every candidate, line by line) with the terms in place and with them switched off, and
requires the two to be byte-identical. ``_snapshot()`` is also what the PR compared between loggetta 50c1553 (main
before the change) and the branch.
"""
import hashlib
import json
from dataclasses import asdict

import pytest

tr = pytest.importorskip("transformers")

from loggetta import Constraints, Workload, plan  # noqa: E402
from loggetta.backends import dense  # noqa: E402
from loggetta.hardware import GPU, Fact, HardwareProfile, Host  # noqa: E402

GiB = 1 << 30
CONFIGS = {
    "small": dict(hidden_size=64, intermediate_size=160, num_hidden_layers=2, num_attention_heads=4,
                  num_key_value_heads=2, vocab_size=128, max_position_embeddings=256),
    "qwen3-14b": dict(hidden_size=5120, intermediate_size=17408, num_hidden_layers=40, num_attention_heads=40,
                      num_key_value_heads=8, head_dim=128, vocab_size=151936, max_position_embeddings=40960,
                      tie_word_embeddings=False),
    "qwen3-32b": dict(hidden_size=5120, intermediate_size=25600, num_hidden_layers=64, num_attention_heads=64,
                      num_key_value_heads=8, head_dim=128, vocab_size=151936, max_position_embeddings=40960,
                      tie_word_embeddings=False),
}
CARDS_GIB = (12, 24, 48, 80)
WORKLOADS = ((512, 1), (2048, 1), (1024, 4))


def _hw(gib):
    r = lambda v: Fact(v, "reported")  # noqa: E731
    g = GPU(index=0, vendor="nvidia", name="Test GPU", uuid=None, compute_capability=r((8, 9)),
            memory_total=r(int(gib * GiB)), memory_free=r(int(gib * GiB)), driver=r("595.84"), pcie_gen_max=r(4),
            pcie_width_max=r(16), pcie_gen_current=r(4), pcie_width_current=r(16))
    host = Host(cpu_model=r("cpu"), cpus=r(16), memory_total=r(128 * GiB), memory_available=r(100 * GiB),
                memory_limit=r(128 * GiB))
    return HardwareProfile(gpus=(g,), host=host, platform="Linux x86_64")


def _candidate(c):
    return {"backend": c.backend, "setup": c.setup, "lines": [asdict(ln) for ln in c.lines],
            "device_bytes": c.device_bytes, "host_bytes": c.host_bytes, "feasible": c.feasible}


def _snapshot() -> dict:
    out = {}
    for name, cfg in CONFIGS.items():
        topo = dense.describe(tr.Qwen3Config(**cfg))
        for gib in CARDS_GIB:
            for seq, mb in WORKLOADS:
                p = plan(topo, _hw(gib), Workload(seq_len=seq, micro_batch=mb), Constraints())
                body = {"status": p.status, "selected": _candidate(p.selected) if p.selected else None,
                        "alternatives": [_candidate(c) for c in p.alternatives]}
                blob = json.dumps(body, sort_keys=True, default=str).encode()
                out[f"{name}/{gib}GiB/seq{seq}x{mb}"] = {
                    "sha256": hashlib.sha256(blob).hexdigest(),
                    "selected_device_bytes": p.selected.device_bytes if p.selected else None}
    return out


def test_dense_estimates_are_byte_identical_with_and_without_the_moe_terms(monkeypatch):
    from loggetta.backends import experts4bit

    with_terms = _snapshot()
    monkeypatch.setattr(experts4bit, "_gnf4_backward_line", lambda *a, **k: None)
    monkeypatch.setattr(experts4bit, "GNF4_UNMODELLED", ())
    without = _snapshot()
    assert sorted(with_terms) == sorted(without)
    changed = {k: (without[k], with_terms[k]) for k in without if without[k] != with_terms[k]}
    assert not changed, f"dense plans moved: {changed}"
