"""Training data preparation. No model weights or device allocations are needed.

User data is tokenized into a temporary, memory-mapped fixed-shape token file before the model loads. The receipt
identifies the source and the exact token stream, not the contents of private examples. Loss covers all tokens,
including instruction/chat prompts. Packing joins examples with EOS; it does not isolate their attention.
"""
from __future__ import annotations

import hashlib
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
                 cache_dir: str | None = None) -> PreparedData:
    """Validate and pack exactly the requested training tokens before any model weights load.

    User datasets never wrap silently: exhaustion is an error unless repeat was explicitly enabled. Unused rows
    beyond the requested token budget are not tokenized. The recorded hash covers the exact little-endian int64
    input stream. The cache is temporary and is never included in an adapter artifact.
    """
    import numpy as np

    if n_blocks <= 0 or seq_len < 2:
        raise ValueError("training requires positive block count and sequence length >= 2")
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
