"""The built-in demonstration (no --dataset) skips Alpaca's empty-output rows and counts them; a dataset the user
supplies is still refused on an empty output. Rows are the real Alpaca shape (row 284 has an instruction and an empty
input and output); no network or GPU."""
import json

import pytest

import loggetta.data as data_mod
from loggetta.data import DEMO_REVISION, DEMO_SOURCE, TrainingData, check_profile, encode_dataset, prepare_data

#: Alpaca row 284 as the Hub serves it: instruction set, input and output empty
ROW_284 = {"instruction": "Find the area of a circle given its radius.", "input": "", "output": "",
           "text": "Below is an instruction ..."}
ROWS = [{"instruction": f"Say {w}.", "input": "", "output": f"{w}!", "text": "x"} for w in ("one", "two")] \
    + [ROW_284] + [{"instruction": "Say three.", "input": "", "output": "three!", "text": "x"}]


class Tokenizer:
    eos_token_id = 0
    name_or_path = "test-tokenizer"

    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(c) % 15 + 1 for c in text]}


@pytest.fixture
def hub_rows(monkeypatch):
    """Serve ROWS for any Hub dataset id, as load_rows would from the Hub."""
    def load_rows(spec):
        return list(ROWS), {"kind": "hub", "id": spec.source, "config": spec.config,
                            "requested_revision": spec.revision, "revision_note": None}
    monkeypatch.setattr(data_mod, "load_rows", load_rows)


def test_the_demonstration_is_pinned_and_marked():
    spec = TrainingData.from_dict(None)
    assert (spec.source, spec.revision, spec.format) == (DEMO_SOURCE, DEMO_REVISION, "alpaca")
    assert spec.skip_empty_output and spec.repeat and (spec.loss, spec.packing) == ("all", "concat")


def test_the_demonstration_skips_the_empty_output_row_and_counts_it(hub_rows):
    spec = TrainingData.from_dict(None)
    total = sum(len(Tokenizer()(t)["input_ids"]) + 1 for t in [
        "### Instruction:\nSay one.\n\n### Input:\n\n\n### Response:\none!",
        "### Instruction:\nSay two.\n\n### Input:\n\n\n### Response:\ntwo!",
        "### Instruction:\nSay three.\n\n### Input:\n\n\n### Response:\nthree!"])
    # more than one pass of the three good rows: row 284 is met (and skipped) on every pass
    n_blocks, seq = 2, total
    with prepare_data(Tokenizer(), n_blocks, seq, spec) as data:
        assert data.info["passes"] == 2
        assert data.info["rows_skipped_empty_output"] == 2
        assert data.info["examples_used"] == 6


def test_a_pass_with_nothing_usable_refuses_instead_of_looping(monkeypatch):
    """With repeat on, a pass that skips every row must raise, not spin forever."""
    monkeypatch.setattr(data_mod, "load_rows", lambda spec: ([dict(ROW_284), dict(ROW_284)], {"kind": "hub"}))
    with pytest.raises(ValueError, match=r"pass 1 yielded no usable rows \(2 skipped for an empty output\)"):
        prepare_data(Tokenizer(), 1, 16, TrainingData.from_dict(None))


def test_a_user_dataset_with_an_empty_output_is_still_refused(hub_rows):
    spec = TrainingData(source=DEMO_SOURCE, revision=DEMO_REVISION, format="alpaca", repeat=True,
                        loss="all", packing="concat")             # the same rows, supplied by the user
    with pytest.raises(ValueError, match=r"dataset row 2, pass 1: column 'output' must contain non-empty text"):
        prepare_data(Tokenizer(), 2, 64, spec)


def test_a_user_local_file_with_an_empty_output_is_refused(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text("\n".join(json.dumps({k: r[k] for k in ("instruction", "input", "output")}) for r in ROWS) + "\n")
    with pytest.raises(ValueError, match="non-empty text"):
        prepare_data(Tokenizer(), 1, 512, TrainingData(str(path), format="alpaca", loss="all", packing="concat"))


@pytest.mark.parametrize("kw", [
    {"source": "my-data.jsonl"},                                   # a user file
    {"source": DEMO_SOURCE, "revision": None},                     # the demo source, unpinned
    {"source": DEMO_SOURCE, "revision": "main"},                   # another revision
    {"source": DEMO_SOURCE, "revision": DEMO_REVISION, "format": "text"},
])
def test_only_the_pinned_demonstration_may_skip(kw):
    kw = {"format": "alpaca", **kw}
    with pytest.raises(ValueError, match="built-in demonstration's alone"):
        TrainingData(skip_empty_output=True, **kw)


def test_the_profile_counts_skips_and_old_profiles_still_match(hub_rows):
    with encode_dataset(Tokenizer(), TrainingData.from_dict(None)) as enc:
        assert enc.profile["rows_skipped_empty_output"] == 1 and enc.profile["rows"] == 3
    user = TrainingData(source="hub/user-data", format="text")
    with encode_dataset(Tokenizer(), user) as enc:
        assert "rows_skipped_empty_output" not in enc.profile
        old = {**enc.profile, "options": {k: v for k, v in enc.profile["options"].items() if k != "skip_empty_output"}}
        check_profile(enc.profile, old)       # a 0.6.1 profile (no skip_empty_output option) still matches
