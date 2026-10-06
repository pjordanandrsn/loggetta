"""User data is validated, profiled before planning and fingerprinted before model loading; tests need no network
or GPU."""
import hashlib
import json

import numpy as np
import pytest

from loggetta.data import TrainingData, encode_dataset, encode_example, longer_than, prepare_data
from loggetta.plan import ExecutionPlan, Workload


class Tokenizer:
    eos_token_id = 0
    name_or_path = "test-tokenizer"
    chat_template = "fixture"

    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(c) % 15 + 1 for c in text]}

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": True, "add_generation_prompt": False}
        return [1, 2, self.eos_token_id]

    def save_pretrained(self, path):
        from pathlib import Path
        (Path(path) / "tokenizer_config.json").write_text('{"test_tokenizer": true}')


def jsonl(tmp_path, rows):
    path = tmp_path / "data.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return path


def test_local_jsonl_exact_token_hash_shape_and_cleanup(tmp_path):
    file = jsonl(tmp_path, [{"text": "abcdefg"}, {"text": "hijklmn"}])
    with prepare_data(Tokenizer(), 2, 4, TrainingData(str(file))) as data:
        assert data.blocks.shape == (2, 4)
        assert data.info["format"] == "text"
        assert data.info["tokens"] == 8 and data.info["examples_used"] == 1
        assert data.info["source"]["sha256"] == hashlib.sha256(file.read_bytes()).hexdigest()
        assert data.info["token_stream_sha256"] == hashlib.sha256(data.blocks.tobytes()).hexdigest()
        assert not data.info["repeated"]
        cached = data._folder.name
    from pathlib import Path
    assert not Path(cached).exists()


def test_short_dataset_never_repeats_silently(tmp_path):
    file = jsonl(tmp_path, [{"text": "abc"}])
    with pytest.raises(ValueError, match="explicitly use --repeat-data"):
        prepare_data(Tokenizer(), 3, 4, TrainingData(str(file)))
    with prepare_data(Tokenizer(), 3, 4, TrainingData(str(file), repeat=True)) as data:
        assert data.info["passes"] == 3 and data.info["repeated"]
        assert np.array_equal(data.blocks[0], data.blocks[2])


def test_shuffle_is_seeded_and_reproducible(tmp_path):
    file = jsonl(tmp_path, [{"text": f"row-{i:02d}"} for i in range(30)])
    hashes = []
    for seed in (4, 4, 9):
        with prepare_data(Tokenizer(), 10, 4, TrainingData(str(file), shuffle=True), seed=seed) as data:
            hashes.append(data.info["token_stream_sha256"])
    assert hashes[0] == hashes[1] and hashes[0] != hashes[2]


@pytest.mark.parametrize("rows,match", [
    ([{"wrong": "abc"}], "cannot infer"),
    ([{"text": ""}], "non-empty text"),
    ([{"text": 2}], "non-empty text"),
    ([{"text": "ok", "messages": []}], "unique dataset format"),
])
def test_invalid_rows_fail_cleanly(tmp_path, rows, match):
    file = jsonl(tmp_path, rows)
    with pytest.raises(ValueError, match=match):
        prepare_data(Tokenizer(), 1, 2, TrainingData(str(file)))


def test_missing_local_file_is_not_mistaken_for_hub_dataset(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        prepare_data(Tokenizer(), 1, 2, TrainingData(str(tmp_path / "absent.jsonl")))


def test_alpaca_supports_optional_input_and_field_mapping():
    spec = TrainingData(format="alpaca", instruction_field="question", output_field="answer")
    ids = encode_example(Tokenizer(), {"question": "Hi", "answer": "Hello"}, spec, "alpaca")
    expected = Tokenizer()("### Instruction:\nHi\n\n### Input:\n\n\n### Response:\nHello")["input_ids"] + [0]
    assert ids == expected


def test_chat_uses_training_template_and_does_not_duplicate_eos():
    messages = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]
    assert encode_example(Tokenizer(), {"messages": messages}, TrainingData(format="chat"), "chat") == [1, 2, 0]
    tok = Tokenizer()
    tok.chat_template = None
    with pytest.raises(ValueError, match="chat template"):
        encode_example(tok, {"messages": messages}, TrainingData(format="chat"), "chat")
    with pytest.raises(ValueError, match="role/content"):
        encode_example(Tokenizer(), {"messages": []}, TrainingData(format="chat"), "chat")


def test_chat_accepts_a_tokenizer_that_returns_a_dict():
    # transformers >= 5: apply_chat_template(tokenize=True) returns a BatchEncoding unless return_dict=False
    class DictTokenizer(Tokenizer):
        def apply_chat_template(self, messages, **kwargs):
            return {"input_ids": [1, 2, self.eos_token_id], "attention_mask": [1, 1, 1]}

    messages = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]
    assert encode_example(DictTokenizer(), {"messages": messages}, TrainingData(format="chat"), "chat") == [1, 2, 0]


def test_chat_with_a_real_transformers_tokenizer():
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    vocab = {"[UNK]": 0, "<eos>": 1, "user": 2, "assistant": 3, "hello": 4, "hi": 5}
    core = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token="[UNK]"))
    core.pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()
    tok = transformers.PreTrainedTokenizerFast(tokenizer_object=core, eos_token="<eos>", unk_token="[UNK]")
    tok.chat_template = "{% for m in messages %}{{ m['role'] }} {{ m['content'] }} <eos> {% endfor %}"
    messages = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
    ids = encode_example(tok, {"messages": messages}, TrainingData(format="chat"), "chat")
    assert ids == [2, 4, 1, 3, 5, 1]


def test_text_column_can_be_selected_explicitly(tmp_path):
    file = jsonl(tmp_path, [{"body": "abc"}])
    with prepare_data(Tokenizer(), 1, 4, TrainingData(str(file), format="text", text_field="body")) as data:
        assert data.info["options"]["text_field"] == "body"


def test_hub_config_revision_and_split_are_forwarded(monkeypatch):
    import datasets
    captured = {}
    def load(*args, **kwargs):
        captured.update(args=args, **kwargs)
        return datasets.Dataset.from_list([{"text": "abcd"}])
    monkeypatch.setattr(datasets, "load_dataset", load)
    spec = TrainingData("org/corpus", config="english", revision="abc123", split="train[:10]")
    with prepare_data(Tokenizer(), 1, 4, spec) as data:
        assert data.info["source"]["requested_revision"] == "abc123"
    assert captured == {"args": ("org/corpus",), "name": "english", "revision": "abc123", "split": "train[:10]"}


def test_empty_dataset_rejected(monkeypatch):
    import datasets
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: datasets.Dataset.from_list([]))
    with pytest.raises(ValueError, match="empty"):
        prepare_data(Tokenizer(), 1, 4, TrainingData("org/empty"))


@pytest.mark.parametrize("kw", [{"steps": 0}, {"seq_len": 1}, {"micro_batch": -1},
                                {"learning_rate": float("nan")}, {"learning_rate": 0}])
def test_invalid_training_dimensions(kw):
    with pytest.raises(ValueError):
        Workload(**kw)


def test_workload_extension_round_trip_and_legacy_wire_shape():
    from test_execution import a_plan
    from dataclasses import replace
    old = a_plan().to_dict()
    assert "data" not in old["workload"] and "learning_rate" not in old["workload"]
    assert "epochs" not in old["workload"] and "data_profile" not in old
    assert ExecutionPlan.from_dict(old).to_dict() == old
    plan = replace(a_plan(), workload=Workload(data=TrainingData("org/corpus").to_dict(), learning_rate=1e-4))
    assert ExecutionPlan.from_dict(plan.to_dict()).to_json() == plan.to_json()
    assert "org/corpus" in plan.render()
    with pytest.raises(ValueError, match="serving"):
        Workload(kind="serve", data=TrainingData().to_dict())


def test_profiling_reads_every_row_before_planning(tmp_path):
    file = jsonl(tmp_path, [{"text": "abcdefg"}] * 3 + [{"text": ""}])
    with prepare_data(Tokenizer(), 1, 4, TrainingData(str(file))) as data:     # the budget never reaches the bad row
        assert data.info["examples_used"] == 1
    with pytest.raises(ValueError, match="dataset row 3: .*non-empty text"):
        encode_dataset(Tokenizer(), TrainingData(str(file)))


def test_the_profile_counts_what_the_planner_needs_and_is_deterministic(tmp_path):
    file = jsonl(tmp_path, [{"text": "x" * n} for n in (3, 63, 64, 200, 20000)])     # + EOS each
    with encode_dataset(Tokenizer(), TrainingData(str(file))) as enc:
        p = enc.profile
        assert len(enc) == 5 and list(enc.ids(0)) == [ord("x") % 15 + 1] * 3 + [0]
    assert p["schema"] == "data-profile/1" and p["rows"] == 5 and p["tokens"] == 4 + 64 + 65 + 201 + 20001
    assert p["lengths"]["min"] == 4 and p["lengths"]["max"] == 20001 and p["lengths"]["p50"] == 65
    assert p["histogram"] == [[0, 64, 2], [64, 128, 1], [192, 256, 1], [16384, 32768, 1]]
    assert p["source"]["sha256"] == hashlib.sha256(file.read_bytes()).hexdigest()
    assert "dataset_fingerprint" not in p["source"]
    assert p["tokenizer"]["eos_token_id"] == 0 and p["options"] == TrainingData(str(file)).to_dict()
    with encode_dataset(Tokenizer(), TrainingData(str(file))) as again:
        assert json.dumps(again.profile, sort_keys=True) == json.dumps(p, sort_keys=True)
    assert longer_than(p, 64) == (3, 3) and longer_than(p, 4096) == (1, 1) and longer_than(p, 100) == (2, 3)


@pytest.mark.parametrize("shuffle,seed,blocks,repeat", [(False, 0, 3, False), (True, 4, 10, False),
                                                         (True, 9, 10, False), (False, 0, 40, True),
                                                         (True, 7, 40, True)])
def test_profiled_packing_gives_the_streamed_token_stream(tmp_path, shuffle, seed, blocks, repeat):
    file = jsonl(tmp_path, [{"text": f"row-{i:02d}"} for i in range(30)])
    spec = TrainingData(str(file), shuffle=shuffle, repeat=repeat)
    with encode_dataset(Tokenizer(), spec) as enc:
        profile = enc.profile
    with prepare_data(Tokenizer(), blocks, 4, spec, seed=seed) as streamed, \
            prepare_data(Tokenizer(), blocks, 4, spec, seed=seed, expect=profile) as profiled:
        assert np.array_equal(streamed.blocks, profiled.blocks)
        for key in ("token_stream_sha256", "examples_used", "passes", "discarded_tail_tokens", "tokens"):
            assert streamed.info[key] == profiled.info[key], key
        assert profiled.info["verified_against_plan"] and profiled.info["encoded_sha256"] == profile["encoded_sha256"]


def test_execution_refuses_data_that_changed_since_planning(tmp_path):
    file = jsonl(tmp_path, [{"text": "abcdefg"}, {"text": "hijklmn"}])
    with encode_dataset(Tokenizer(), TrainingData(str(file))) as enc:
        planned = enc.profile
    file.write_text(json.dumps({"text": "abcdefg"}) + "\n" + json.dumps({"text": "hijklmX"}) + "\n")
    with pytest.raises(ValueError, match="changed since the plan was made .*encoded_sha256"):
        prepare_data(Tokenizer(), 2, 4, TrainingData(str(file)), expect=planned)


def test_epochs_allow_exactly_the_passes_they_imply(tmp_path):
    file = jsonl(tmp_path, [{"text": "abc"}])                                       # 4 tokens with EOS
    with encode_dataset(Tokenizer(), TrainingData(str(file))) as enc:
        profile = enc.profile
    with pytest.raises(ValueError, match="in 1 pass"):
        prepare_data(Tokenizer(), 2, 4, TrainingData(str(file)), expect=profile)
    with prepare_data(Tokenizer(), 2, 4, TrainingData(str(file)), expect=profile, max_passes=2) as data:
        assert data.info["passes"] == 2
    with pytest.raises(ValueError, match="in 2 pass"):
        prepare_data(Tokenizer(), 3, 4, TrainingData(str(file)), expect=profile, max_passes=2)


def test_epochs_round_trip_and_need_a_dataset():
    from dataclasses import replace

    from test_execution import a_plan
    w = Workload(data=TrainingData("org/corpus").to_dict(), epochs=1.5)
    plan = replace(a_plan(), workload=w, data_profile={"schema": "data-profile/1", "rows": 1})
    assert ExecutionPlan.from_dict(json.loads(plan.to_json())).to_json() == plan.to_json()
    with pytest.raises(ValueError, match="need a dataset"):
        Workload(epochs=1.0)
    with pytest.raises(ValueError, match="finite and positive"):
        Workload(data=TrainingData("org/corpus").to_dict(), epochs=0)
