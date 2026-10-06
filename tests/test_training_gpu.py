"""Optional full pipeline on CUDA using a tiny randomly initialized local MoE, never a Hub download."""
import json
from pathlib import Path

import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")]


def test_tiny_local_moe_train_save_reload(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen3MoeConfig, Qwen3MoeForCausalLM
    from loggetta import Constraints, Workload, describe_model, execute, load_adapter, plan, probe
    from loggetta.data import TrainingData

    base = tmp_path / "base"
    config = Qwen3MoeConfig(hidden_size=64, intermediate_size=128, moe_intermediate_size=128,
                            num_experts=4, num_experts_per_tok=2, num_hidden_layers=1,
                            num_attention_heads=4, num_key_value_heads=2, head_dim=16, vocab_size=32,
                            max_position_embeddings=256, decoder_sparse_step=1)
    torch.manual_seed(7)
    Qwen3MoeForCausalLM(config).save_pretrained(base)
    raw = Tokenizer(WordLevel({"<eos>": 0, "<unk>": 1, "red": 2, "green": 3, "blue": 4}, unk_token="<unk>"))
    raw.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=raw, eos_token="<eos>", unk_token="<unk>", pad_token="<eos>")
    tok.save_pretrained(base)
    data = tmp_path / "train.jsonl"
    data.write_text(json.dumps({"text": "red green blue " * 64}) + "\n")
    hw = probe()
    p = plan(describe_model(str(base)), hw,
             Workload(seq_len=32, steps=2, data=TrainingData(str(data), format="text").to_dict()),
             Constraints(fixed={"expert_kernel": "reference"}))
    assert p.status == "feasible"
    receipt = execute(p, out_dir=str(tmp_path / "runs"), hardware=hw)
    assert receipt["status"] == "OK"
    adapter = receipt["artifacts"]["adapter"]["path"]
    assert (Path(adapter) / "adapter.safetensors").is_file()
    assert receipt["correctness"]["frozen_expert_bytes_unchanged"]
    first = load_adapter(adapter, device="cuda")
    ids = torch.tensor([[2, 3, 4, 0]], device="cuda")
    with torch.no_grad():
        logits = first(ids).logits.detach().cpu()
    del first
    second = load_adapter(adapter, device="cuda")
    with torch.no_grad():
        torch.testing.assert_close(second(ids).logits.detach().cpu(), logits, rtol=0, atol=0)
