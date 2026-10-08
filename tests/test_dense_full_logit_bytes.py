"""Full-logit estimate against real autograd storage liveness; no fitted coefficient or GPU timing."""
import importlib.util
from pathlib import Path

import pytest


def test_full_logit_workspace_covers_live_cross_entropy_backward_tensors():
    pytest.importorskip("torch")
    tr = pytest.importorskip("transformers")
    from loggetta import Workload
    from loggetta.backends import dense

    path = Path(__file__).parents[1]/"bench/dq7-loss-diagnosis/dense_loss_census.py"
    spec = importlib.util.spec_from_file_location("loss_census_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    seq, vocab = 16, 1024
    measured = module.run("cpu", seq, vocab)
    # A frozen bf16 head runs the exact causal loss path; three different fp32 tensor storages overlap.
    assert measured["backward_live_tensor_bytes"] > measured["priced_full_logits_bytes"]
    topology = dense.describe(tr.LlamaConfig(hidden_size=64, intermediate_size=160, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, vocab_size=vocab))
    setup = {"base": "nf4", "placement": "device", "loss_chunk": 0, "r": 16, "alpha": 32,
             "adapter_dtype": "fp32", "targets": list(dense.ROLES), "attn_impl": "sdpa"}
    lines, _, refused = dense.estimate(topology, setup, Workload(seq_len=seq))
    assert not refused
    workspace = next(row for row in lines if row[0] == "activations")
    assert workspace[2] >= measured["backward_live_tensor_bytes"]
    assert "NLL grad" in workspace[4]
