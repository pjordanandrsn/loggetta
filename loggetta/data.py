"""Training data preparation. No model weights or device allocations are needed.

A user dataset is validated and tokenized, every row, before the plan is made (:func:`encode_dataset`); its
:func:`profile <encode_dataset>` is an input to the planner, like the hardware inventory. Execution tokenizes it
again, refuses to continue if the result differs from the profile the plan was made from, and packs it into a
temporary, memory-mapped fixed-shape token file before the model loads. The receipt identifies the source and the
exact token stream, not the contents of private examples. Loss covers all tokens, including instruction/chat
prompts. Packing joins examples with EOS; it does not isolate their attention.
"""
from __future__ import annotations

import hashlib
import math
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingData:
    source: str = "tatsu-lab/alpaca"
    format: str = "auto"                 # auto | text | alpaca | chat
    split: str = "train"
    config: str | None = None
    revision: str | None = None
    text_field: str = "text"
    messages_field: str = "messages"
    instruction_field: str = "instruction"
    input_field: str = "input"
    output_field: str = "output"
    shuffle: bool = False
    repeat: bool = False

    def __post_init__(self):
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("dataset source must be a non-empty local file path or Hub dataset ID")
        if self.format not in ("auto", "text", "alpaca", "chat"):
            raise ValueError("dataset format must be auto, text, alpaca, or chat")
        for name in ("split", "text_field", "messages_field", "instruction_field", "input_field", "output_field"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"dataset {name} must be a non-empty string")
        if not isinstance(self.shuffle, bool) or not isinstance(self.repeat, bool):
            raise ValueError("dataset shuffle and repeat must be booleans")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict | None) -> "TrainingData":
        # Old plans retain the original explicitly documented demonstration workload.
        return cls(**value) if value is not None else cls(format="alpaca", repeat=True)


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(spec: TrainingData):
    from datasets import load_dataset

    path = Path(spec.source).expanduser()
    suffix = path.suffix.lower()
    formats = {".json": "json", ".jsonl": "json", ".csv": "csv", ".parquet": "parquet", ".txt": "text"}
    local = path.exists() or path.is_absolute() or spec.source.startswith((".", "~")) or suffix in formats
    if local:
        if not path.is_file():
            raise ValueError(f"dataset file does not exist or is not a regular file: {path}")
        if suffix not in formats:
            raise ValueError("local dataset must be JSON, JSONL, CSV, Parquet, or line-oriented TXT")
        if spec.config is not None or spec.revision is not None:
            raise ValueError("dataset config/revision apply to Hub datasets, not local files")
        before = file_sha256(path)
        ds = load_dataset(formats[suffix], data_files={"train": str(path.resolve())}, split=spec.split)
        if file_sha256(path) != before:
            raise ValueError("dataset file changed while it was being loaded; retry with an unchanged file")
        source = {"kind": "local", "path": str(path.resolve()), "sha256": before}
    else:
        if "://" in spec.source:
            raise ValueError("use a Hub dataset ID or download the dataset to a local file; URLs are not accepted")
        ds = load_dataset(spec.source, name=spec.config, revision=spec.revision, split=spec.split)
        source = {"kind": "hub", "id": spec.source, "config": spec.config,
                  "requested_revision": spec.revision,
                  "revision_note": "pin --dataset-revision for a reproducible Hub revision" if spec.revision is None else None}
    if not len(ds):
        raise ValueError("the selected dataset split is empty")
    source["dataset_fingerprint"] = getattr(ds, "_fingerprint", None)
    return ds, source


def resolve_format(spec: TrainingData, columns) -> str:
    if spec.format != "auto":
        fmt = spec.format
    else:
        possible = []
        if spec.text_field in columns:
            possible.append("text")
        if spec.instruction_field in columns and spec.output_field in columns:
            possible.append("alpaca")
        if spec.messages_field in columns:
            possible.append("chat")
        if len(possible) != 1:
            raise ValueError("cannot infer a unique dataset format; use --format text, alpaca, or chat and field options")
        fmt = possible[0]
    required = {"text": (spec.text_field,), "alpaca": (spec.instruction_field, spec.output_field),
                "chat": (spec.messages_field,)}[fmt]
    missing = set(required) - set(columns)
    if missing:
        raise ValueError(f"dataset is missing required columns: {', '.join(sorted(missing))}")
    return fmt


def encode_example(tokenizer, ex: dict, spec: TrainingData, fmt: str) -> list[int]:
    def text(name, *, optional=False):
        value = ex.get(name, "" if optional else None)
        if optional and value is None:
            value = ""
        if not isinstance(value, str) or (not optional and not value.strip()):
            raise ValueError(f"column {name!r} must contain {'a string' if optional else 'non-empty text'}")
        return value

    if fmt == "chat":
        messages = ex.get(spec.messages_field)
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list of role/content objects")
        for message in messages:
            if (not isinstance(message, dict) or not isinstance(message.get("role"), str)
                    or not isinstance(message.get("content"), str) or not message["content"].strip()):
                raise ValueError("every chat message must have a string role and non-empty text content")
        if not getattr(tokenizer, "chat_template", None):
            raise ValueError("chat data requires a tokenizer chat template; supply preformatted text instead")
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        # transformers >= 5 returns a BatchEncoding here by default (return_dict=True); 4.x returned the ids
        if isinstance(ids, Mapping):
            ids = ids.get("input_ids", ())
    else:
        value = text(spec.text_field) if fmt == "text" else (
            f"### Instruction:\n{text(spec.instruction_field)}\n\n### Input:\n{text(spec.input_field, optional=True)}"
            f"\n\n### Response:\n{text(spec.output_field)}")
        ids = tokenizer(value, add_special_tokens=False)["input_ids"]
    ids = list(ids)
    if not ids or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 or i >= 2**63 for i in ids):
        raise ValueError("tokenizer must produce non-empty, non-negative integer token IDs")
    eos = tokenizer.eos_token_id
    if eos is None or not isinstance(eos, int) or eos < 0:
        raise ValueError("tokenizer must define an EOS token for fixed-shape packing")
    # Chat templates often already end in EOS. Do not duplicate it.
    if ids[-1] != eos:
        ids.append(eos)
    return ids


class PreparedData:
    """Temporary packed tokens; RAM use is bounded by a row and an accessed batch, not training steps."""

    def __init__(self, folder, blocks, info):
        self._folder, self.blocks, self.info = folder, blocks, info

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        mapping = getattr(self.blocks, "_mmap", None)
        if mapping is not None:
            mapping.close()
        self._folder.cleanup()


def prepare_data(tokenizer, n_blocks: int, seq_len: int, spec: TrainingData, *, seed: int = 0,
                 cache_dir: str | None = None, expect: dict | None = None,
                 max_passes: int | None = None) -> PreparedData:
    """Validate and pack exactly the requested training tokens before any model weights load.

    User datasets never wrap silently: exhaustion is an error unless repeat was explicitly enabled, or ``max_passes``
    allows that many passes (a plan's epochs). The recorded hash covers the exact little-endian int64 input stream. The
    cache is temporary and is never included in an adapter artifact.

    With ``expect`` (the data profile the plan was made from), every row is tokenized again and the result must match
    the profile, or this raises before anything is packed: the dataset, its options or the tokenizer changed since
    planning. Without it (plans made before data profiles, and the demonstration data), only the rows the token
    budget needs are tokenized, as before.
    """
    if n_blocks <= 0 or seq_len < 2:
        raise ValueError("training requires positive block count and sequence length >= 2")
    if expect is None:
        return _prepare_streamed(tokenizer, n_blocks, seq_len, spec, seed=seed, cache_dir=cache_dir)
    with encode_dataset(tokenizer, spec, cache_dir=cache_dir) as enc:
        check_profile(enc.profile, expect)
        return _pack(enc, n_blocks, seq_len, spec, seed=seed, cache_dir=cache_dir, max_passes=max_passes,
                     tokenizer=tokenizer)


def _prepare_streamed(tokenizer, n_blocks: int, seq_len: int, spec: TrainingData, *, seed: int = 0,
                      cache_dir: str | None = None) -> PreparedData:
    """The 0.2.0 path: rows tokenized in (seeded) order until the token budget is met; rows after it are not read."""
    import numpy as np

    ds, source = load_rows(spec)
    columns = getattr(ds, "column_names", None) or list(ds[0])
    fmt = resolve_format(spec, columns)
    if spec.shuffle:
        ds = ds.shuffle(seed=seed)
    folder = tempfile.TemporaryDirectory(prefix="training-tokens-", dir=cache_dir)
    token_path = Path(folder.name) / "tokens.bin"
    required = n_blocks * seq_len
    written = encoded = passes = discarded = 0
    digest = hashlib.sha256()
    try:
        with token_path.open("wb") as stream:
            while written < required:
                passes += 1
                for row, ex in enumerate(ds):
                    try:
                        ids = encode_example(tokenizer, ex, spec, fmt)
                    except (ValueError, TypeError, KeyError) as exc:
                        raise ValueError(f"dataset row {row}, pass {passes}: {exc}") from exc
                    encoded += 1
                    used = min(len(ids), required - written)
                    packed = np.asarray(ids[:used], dtype="<i8").tobytes()
                    stream.write(packed)
                    digest.update(packed)
                    written += used
                    discarded += len(ids) - used
                    if written == required:
                        break
                if written < required and not spec.repeat:
                    raise ValueError(f"dataset provides {written} tokens, but the plan needs {required}; reduce --steps "
                                     "or explicitly use --repeat-data")
        info = {"dataset": spec.source, "source": source, "split": spec.split, "format": fmt,
                "options": spec.to_dict(), "examples_used": encoded, "source_rows": len(ds), "passes": passes,
                "repeated": passes > 1, "shuffled": spec.shuffle, "seed": seed,
                "tokens": written, "blocks": n_blocks, "seq_len": seq_len, "discarded_tail_tokens": discarded,
                "token_stream_sha256": digest.hexdigest(), "token_encoding": "little-endian int64",
                "token_cache_bytes": written * 8, "loss": "full-sequence, including prompts",
                "packing": "concatenated examples separated by EOS; attention is not isolated between examples",
                "tokenizer": {"name": getattr(tokenizer, "name_or_path", None),
                              "class": type(tokenizer).__name__, "eos_token_id": tokenizer.eos_token_id}}
        blocks = np.memmap(token_path, dtype="<i8", mode="r", shape=(n_blocks, seq_len))
        return PreparedData(folder, blocks, info)
    except BaseException:
        folder.cleanup()
        raise


#: :func:`encode_dataset`'s profile format
PROFILE_SCHEMA = "data-profile/1"
#: the length histogram's bins ``(low, high]``: 64 tokens wide up to 16,384 tokens, powers of two above, so the number
#: of examples longer than a sequence length that is a multiple of 64 (up to 16,384) is exact
HIST_STEP, HIST_FINE_MAX = 64, 16384


def _bin_low(n: int) -> int:
    m = int(n) - 1
    return (m // HIST_STEP) * HIST_STEP if m < HIST_FINE_MAX else 1 << (m.bit_length() - 1)


def _bin_high(low: int) -> int:
    return low + HIST_STEP if low < HIST_FINE_MAX else 2 * low


def load_tokenizer(model, *, revision=None, trust_remote_code=False):
    """The model's tokenizer (its tokenizer files only; no weights)."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model, revision=revision, trust_remote_code=trust_remote_code)


def tokenizer_identity(tokenizer) -> dict:
    """What a profile records about the tokenizer: enough to say it is not the one the plan was made with."""
    template = getattr(tokenizer, "chat_template", None)
    try:
        size = len(tokenizer)
    except TypeError:
        size = None
    return {"name": getattr(tokenizer, "name_or_path", None), "class": type(tokenizer).__name__, "vocab_size": size,
            "eos_token_id": tokenizer.eos_token_id,
            "chat_template_sha256": hashlib.sha256(template.encode()).hexdigest() if isinstance(template, str) else None}


class EncodedData:
    """Every example of a dataset, validated and tokenized in source order (EOS appended, as packing joins them): a
    temporary memory-mapped token file, the offsets into it, and :attr:`profile`, what the planner reads."""

    def __init__(self, folder, tokens, offsets, profile, source, fmt):
        self._folder, self.tokens, self.offsets = folder, tokens, offsets
        self.profile, self.source, self.format = profile, source, fmt

    def __len__(self):
        return len(self.offsets) - 1

    def ids(self, i: int):
        return self.tokens[self.offsets[i]:self.offsets[i + 1]]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        mapping = getattr(self.tokens, "_mmap", None)
        if mapping is not None:
            mapping.close()
        self._folder.cleanup()


def encode_dataset(tokenizer, spec: TrainingData, *, cache_dir: str | None = None) -> EncodedData:
    """Validate and tokenize every row of ``spec``'s dataset, in source order, and profile the result.

    The profile is deterministic for the same file (or Hub revision), options and tokenizer, so a plan made from it is
    too. ``encoded_sha256`` covers each example's length and ids; ``execute`` compares it with a fresh encoding."""
    import numpy as np

    ds, source = load_rows(spec)
    columns = getattr(ds, "column_names", None) or list(ds[0])
    fmt = resolve_format(spec, columns)
    folder = tempfile.TemporaryDirectory(prefix="training-tokens-", dir=cache_dir)
    path = Path(folder.name) / "encoded.bin"
    lengths, digest = [], hashlib.sha256()
    try:
        with path.open("wb") as stream:
            for row, ex in enumerate(ds):
                try:
                    ids = encode_example(tokenizer, ex, spec, fmt)
                except (ValueError, TypeError, KeyError) as exc:
                    raise ValueError(f"dataset row {row}: {exc}") from exc
                packed = np.asarray(ids, dtype="<i8").tobytes()
                digest.update(np.asarray([len(ids)], dtype="<i8").tobytes())
                digest.update(packed)
                stream.write(packed)
                lengths.append(len(ids))
        lens = np.asarray(lengths, dtype=np.int64)
        offsets = np.concatenate(([0], np.cumsum(lens))).astype(np.int64)
        tokens = np.memmap(path, dtype="<i8", mode="r", shape=(int(offsets[-1]),))
        lows, counts = np.unique(np.asarray([_bin_low(int(n)) for n in lens], dtype=np.int64), return_counts=True)
        q = lambda p: int(np.percentile(lens, p, method="higher"))  # noqa: E731 - an observed length, not interpolated
        profile = {
            "schema": PROFILE_SCHEMA,
            "source": {k: v for k, v in source.items() if k != "dataset_fingerprint"},
            "split": spec.split, "format": fmt, "options": spec.to_dict(),
            "rows": len(lens), "tokens": int(lens.sum()),
            "lengths": {"min": int(lens.min()), "p50": q(50), "p90": q(90), "p99": q(99), "max": int(lens.max()),
                        "mean": round(float(lens.mean()), 2)},
            "histogram": [[int(lo), _bin_high(int(lo)), int(c)] for lo, c in zip(lows, counts)],
            "tokenizer": tokenizer_identity(tokenizer),
            "encoded_sha256": digest.hexdigest(),
            "encoding": "per example in source order: its length, then its ids with EOS, as little-endian int64",
            "loss": "full-sequence, including prompts",
            "packing": "concatenated examples separated by EOS; attention is not isolated between examples",
        }
        return EncodedData(folder, tokens, offsets, profile, source, fmt)
    except BaseException:
        folder.cleanup()
        raise


def check_profile(found: dict, expect: dict) -> None:
    """Raise unless ``found`` (a fresh encoding) is the data the plan was made from."""
    keys = ("encoded_sha256", "rows", "tokens", "format", "options", "tokenizer")
    changed = [k for k in keys if found.get(k) != expect.get(k)]
    if changed:
        raise ValueError(f"the training data changed since the plan was made ({', '.join(changed)} differ); plan "
                         "again with the data as it is now")


def longer_than(profile: dict, seq_len: int) -> tuple:
    """``(at_least, at_most)`` examples longer than ``seq_len`` tokens, from the profile's histogram (equal when
    ``seq_len`` falls on a bin edge)."""
    sure = sum(c for lo, hi, c in profile["histogram"] if lo >= seq_len)
    maybe = sum(c for lo, hi, c in profile["histogram"] if lo < seq_len < hi)
    return sure, sure + maybe


def _pack(enc: EncodedData, n_blocks: int, seq_len: int, spec: TrainingData, *, seed: int, cache_dir, max_passes,
          tokenizer) -> PreparedData:
    """Pack an encoded dataset exactly as :func:`_prepare_streamed` packs the rows it reads (same order, same
    shuffle, same token stream), from the encoding instead of the tokenizer."""
    import numpy as np

    order = range(len(enc))
    if spec.shuffle:                      # the permutation Dataset.shuffle(seed) applies to a dataset of this length
        from datasets import Dataset

        order = Dataset.from_dict({"i": np.arange(len(enc))}).shuffle(seed=seed)["i"]
    allowed = max_passes if max_passes is not None else (math.inf if spec.repeat else 1)
    folder = tempfile.TemporaryDirectory(prefix="training-tokens-", dir=cache_dir)
    token_path = Path(folder.name) / "tokens.bin"
    required = n_blocks * seq_len
    written = used_examples = passes = discarded = 0
    digest = hashlib.sha256()
    try:
        with token_path.open("wb") as stream:
            while written < required:
                if passes >= allowed:
                    raise ValueError(f"dataset provides {written} tokens in {passes} pass(es), but the plan needs "
                                     f"{required}; reduce --steps or explicitly use --repeat-data")
                passes += 1
                for i in order:
                    ids = enc.ids(int(i))
                    used_examples += 1
                    used = min(len(ids), required - written)
                    packed = np.asarray(ids[:used], dtype="<i8").tobytes()
                    stream.write(packed)
                    digest.update(packed)
                    written += used
                    discarded += len(ids) - used
                    if written == required:
                        break
        info = {"dataset": spec.source, "source": enc.source, "split": spec.split, "format": enc.format,
                "options": spec.to_dict(), "examples_used": used_examples, "source_rows": len(enc), "passes": passes,
                "repeated": passes > 1, "shuffled": spec.shuffle, "seed": seed,
                "tokens": written, "blocks": n_blocks, "seq_len": seq_len, "discarded_tail_tokens": discarded,
                "token_stream_sha256": digest.hexdigest(), "token_encoding": "little-endian int64",
                "token_cache_bytes": written * 8, "loss": "full-sequence, including prompts",
                "packing": "concatenated examples separated by EOS; attention is not isolated between examples",
                "tokenizer": {"name": getattr(tokenizer, "name_or_path", None),
                              "class": type(tokenizer).__name__, "eos_token_id": tokenizer.eos_token_id},
                "encoded_sha256": enc.profile["encoded_sha256"], "verified_against_plan": True}
        blocks = np.memmap(token_path, dtype="<i8", mode="r", shape=(n_blocks, seq_len))
        return PreparedData(folder, blocks, info)
    except BaseException:
        folder.cleanup()
        raise
