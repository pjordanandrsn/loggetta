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


LOSSES = ("auto", "all", "assistant")
PACKINGS = ("auto", "concat", "isolated")
PACKING_TEXT = {
    "concat": "concatenated examples separated by EOS; attention is not isolated between examples",
    "isolated": "whole examples per row (truncated to the row), positions reset per example, padding masked; each "
                "example attends only to itself (transformers' packed-sequence masks)",
}
#: what the receipt and profile say each resolved loss trains on
LOSS_TEXT = {"all": "full-sequence, including prompts",
             "assistant": "assistant tokens only: each assistant turn's text and its end-of-turn marker (chat), or the "
                          "response and EOS (alpaca)"}


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
    #: which tokens train: "auto" (assistant turns for chat, the response for alpaca, everything for text), "all", or
    #: "assistant"
    loss: str = "auto"
    #: how examples share a row: "auto" (isolated for chat and alpaca, concatenated for text), "concat" or "isolated"
    packing: str = "auto"

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
        if self.loss not in LOSSES:
            raise ValueError(f"loss must be one of {', '.join(LOSSES)}")
        if self.packing not in PACKINGS:
            raise ValueError(f"packing must be one of {', '.join(PACKINGS)}")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict | None) -> "TrainingData":
        # Old plans retain the original explicitly documented demonstration workload, and plans written before the
        # loss field trained on every token.
        legacy = {"loss": "all", "packing": "concat"}
        return cls(**{**legacy, **value}) if value is not None else cls(format="alpaca", repeat=True, **legacy)


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


def _text(ex, name, *, optional=False):
    value = ex.get(name, "" if optional else None)
    if optional and value is None:
        value = ""
    if not isinstance(value, str) or (not optional and not value.strip()):
        raise ValueError(f"column {name!r} must contain {'a string' if optional else 'non-empty text'}")
    return value


def _alpaca_parts(ex, spec):
    """``(prompt, response)``: the fixed Alpaca template up to the response, and the response."""
    return (f"### Instruction:\n{_text(ex, spec.instruction_field)}\n\n### Input:\n"
            f"{_text(ex, spec.input_field, optional=True)}\n\n### Response:\n", _text(ex, spec.output_field))


def _chat_messages(tokenizer, ex, spec):
    messages = ex.get(spec.messages_field)
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list of role/content objects")
    for message in messages:
        if (not isinstance(message, dict) or not isinstance(message.get("role"), str)
                or not isinstance(message.get("content"), str) or not message["content"].strip()):
            raise ValueError("every chat message must have a string role and non-empty text content")
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("chat data requires a tokenizer chat template; supply preformatted text instead")
    return messages


def _checked_ids(ids) -> list:
    ids = list(ids)
    if not ids or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 or i >= 2**63 for i in ids):
        raise ValueError("tokenizer must produce non-empty, non-negative integer token IDs")
    return ids


def _eos(tokenizer) -> int:
    eos = tokenizer.eos_token_id
    if eos is None or not isinstance(eos, int) or eos < 0:
        raise ValueError("tokenizer must define an EOS token for fixed-shape packing")
    return eos


def encode_example(tokenizer, ex: dict, spec: TrainingData, fmt: str) -> list[int]:
    if fmt == "chat":
        ids = tokenizer.apply_chat_template(_chat_messages(tokenizer, ex, spec), tokenize=True,
                                            add_generation_prompt=False)
        # transformers >= 5 returns a BatchEncoding here by default (return_dict=True); 4.x returned the ids
        if isinstance(ids, Mapping):
            ids = ids.get("input_ids", ())
    else:
        value = _text(ex, spec.text_field) if fmt == "text" else "".join(_alpaca_parts(ex, spec))
        ids = tokenizer(value, add_special_tokens=False)["input_ids"]
    ids = _checked_ids(ids)
    eos = _eos(tokenizer)
    # Chat templates often already end in EOS. Do not duplicate it.
    if ids[-1] != eos:
        ids.append(eos)
    return ids


def resolve_loss(spec: TrainingData, fmt: str) -> str:
    """The loss a spec means for a resolved format: "all" or "assistant"."""
    if spec.loss == "auto":
        return "assistant" if fmt in ("chat", "alpaca") else "all"
    if spec.loss == "assistant" and fmt == "text":
        raise ValueError("--loss assistant needs chat or alpaca data: plain text has no assistant turns; use --loss all")
    return spec.loss


def resolve_packing(spec: TrainingData, fmt: str) -> str:
    """How examples share a row for a resolved format: "concat" or "isolated"."""
    if spec.packing == "auto":
        return "isolated" if fmt in ("chat", "alpaca") else "concat"
    return spec.packing


#: open rows the packer keeps while placing examples: each example goes to the open row it fills most tightly
PACK_WINDOW = 64


def pack_rows(lengths, seq_len: int, window: int = PACK_WINDOW) -> list:
    """Whole examples into rows of ``seq_len`` tokens, in order: each example (truncated to ``seq_len``) goes into the
    open row it fits most tightly; a new row opens when none fits; with ``window`` rows open, the fullest closes first.
    Deterministic and online. Returns the rows, as lists of example indices, in the order they close."""
    open_rows, closed = [], []                 # open: [remaining, opened_at, indices]
    for i, n in enumerate(lengths):
        need = min(int(n), seq_len)
        fits = [r for r in open_rows if r[0] >= need]
        if fits:
            row = min(fits, key=lambda r: (r[0], r[1]))
        else:
            if len(open_rows) >= window:
                full = min(open_rows, key=lambda r: (r[0], r[1]))
                open_rows.remove(full)
                closed.append(full[2])
            row = [seq_len, i, []]
            open_rows.append(row)
        row[0] -= need
        row[2].append(i)
    closed += [r[2] for r in sorted(open_rows, key=lambda r: r[1])]
    return closed


#: probe texts for reading a chat template's end-of-turn marker (unlikely in any real conversation; mixed case, so a
#: template that changes case does not render them verbatim)
_PROBE_USER, _PROBE_ASSISTANT = "Qz7-probe-User-turn", "Qz7-probe-Assistant-turn"


def _rendered(tokenizer, messages) -> str:
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    if not isinstance(text, str):
        raise ValueError("the chat template did not render text")
    return text


def chat_turn_end(tokenizer) -> str | None:
    """The text a chat template puts after an assistant message's content to close its turn (``<|im_end|>``,
    ``<|eot_id|>``, ``<end_of_turn>``, ``</s>``...), read from the template itself; None if it closes with nothing."""
    text = _rendered(tokenizer, [{"role": "user", "content": _PROBE_USER},
                                 {"role": "assistant", "content": _PROBE_ASSISTANT}])
    if _PROBE_ASSISTANT not in text:
        raise ValueError("the chat template does not render an assistant message's text verbatim, so assistant tokens "
                         "cannot be located; use --loss all")
    return text.split(_PROBE_ASSISTANT, 1)[1].strip() or None


def _offsets(tokenizer, text):
    try:
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        return list(enc["input_ids"]), [tuple(o) for o in enc["offset_mapping"]]
    except (NotImplementedError, TypeError, ValueError, KeyError) as exc:
        raise ValueError("an assistant-only loss needs a fast tokenizer (character offsets); use --loss all") from exc


def _mask_spans(offsets, spans) -> list:
    """1 for each token whose characters overlap a span, else 0."""
    return [int(any((s < b and e > a) or (s == e and a <= s < b) for a, b in spans)) for s, e in offsets]


def _chat_spans(tokenizer, messages, text) -> list:
    """Character spans of every assistant turn in the rendered conversation: its content, then the template's
    end-of-turn marker. Messages are located in order, so text repeated from an earlier message cannot be mistaken
    for a later one; a template that rewrites a message's text is refused rather than guessed."""
    end, spans, cursor = chat_turn_end(tokenizer), [], 0
    for m in messages:
        content = m["content"]
        start = text.find(content, cursor)
        if start < 0:
            content = content.strip()
            start = text.find(content, cursor)
        if start < 0:
            raise ValueError(f"the chat template rewrites a {m['role']!r} message, so its tokens cannot be located for "
                             "an assistant-only loss; use --loss all")
        stop = start + len(content)
        if m["role"] == "assistant":
            if end is not None:
                rest = text[stop:]
                if not rest.lstrip().startswith(end):
                    raise ValueError(f"the chat template does not close an assistant turn with {end!r} where expected; "
                                     "use --loss all")
                stop += len(rest) - len(rest.lstrip()) + len(end)
            spans.append((start, stop))
        cursor = stop
    if not spans:
        raise ValueError("the conversation has no assistant message, so an assistant-only loss trains nothing")
    return spans


def encode_masked(tokenizer, ex: dict, spec: TrainingData, fmt: str, loss: str):
    """``(ids, mask)`` for one example: ``mask`` (1 = trained) is None for a full-sequence loss. The ids are exactly
    :func:`encode_example`'s."""
    if loss == "all":
        return encode_example(tokenizer, ex, spec, fmt), None
    if fmt == "chat":
        messages = _chat_messages(tokenizer, ex, spec)
        text = _rendered(tokenizer, messages)
        ids, offsets = _offsets(tokenizer, text)
        mask = _mask_spans(offsets, _chat_spans(tokenizer, messages, text))
        eos_trains = 0                    # the turn's own end-of-turn marker trains; a separating EOS does not
    elif fmt == "alpaca":
        prompt, response = _alpaca_parts(ex, spec)
        ids, offsets = _offsets(tokenizer, prompt + response)
        mask = _mask_spans(offsets, [(len(prompt), len(prompt) + len(response))])
        eos_trains = 1                    # the response ends with EOS: the model learns to stop
    else:
        raise ValueError(f"no assistant tokens in {fmt} data")
    ids = _checked_ids(ids)
    eos = _eos(tokenizer)
    if ids[-1] != eos:
        ids.append(eos)
        mask.append(eos_trains)
    if not any(mask):
        raise ValueError("no token of this example trains under --loss assistant")
    return ids, mask


class PreparedData:
    """Temporary packed tokens; RAM use is bounded by a row and an accessed batch, not training steps."""

    def __init__(self, folder, blocks, info, mask=None, positions=None):
        #: ``mask`` (uint8, the blocks' shape; 1 = the token is a training target) is None for a full-sequence loss;
        #: ``positions`` (int32, the blocks' shape) restart at 0 for every example under isolated packing, else None
        self._folder, self.blocks, self.info, self.mask, self.positions = folder, blocks, info, mask, positions

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        for array in (self.blocks, self.mask, self.positions):
            mapping = getattr(array, "_mmap", None)
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
        return _prepare_streamed(tokenizer, n_blocks, seq_len, spec, seed=seed, cache_dir=cache_dir,
                                 max_passes=max_passes)
    with encode_dataset(tokenizer, spec, cache_dir=cache_dir, seq_len=seq_len) as enc:
        check_profile(enc.profile, expect)
        if enc.packing == "isolated":
            return _pack_isolated(enc, n_blocks, seq_len, spec, seed=seed, cache_dir=cache_dir, max_passes=max_passes,
                                  tokenizer=tokenizer, verified=True)
        return _pack(enc, n_blocks, seq_len, spec, seed=seed, cache_dir=cache_dir, max_passes=max_passes,
                     tokenizer=tokenizer)


def _prepare_streamed(tokenizer, n_blocks: int, seq_len: int, spec: TrainingData, *, seed: int = 0,
                      cache_dir: str | None = None, max_passes: int | None = None) -> PreparedData:
    """The 0.2.0 path: rows tokenized in (seeded) order until the token budget is met; rows after it are not read.
    Isolated packing places whole examples, so it reads them all first."""
    ds, source = load_rows(spec)
    columns = getattr(ds, "column_names", None) or list(ds[0])
    fmt = resolve_format(spec, columns)
    loss = resolve_loss(spec, fmt)
    if resolve_packing(spec, fmt) == "isolated":
        with encode_dataset(tokenizer, spec, cache_dir=cache_dir, seq_len=seq_len) as enc:
            return _pack_isolated(enc, n_blocks, seq_len, spec, seed=seed, cache_dir=cache_dir,
                                  max_passes=max_passes, tokenizer=tokenizer, verified=False)
    if spec.shuffle:
        ds = ds.shuffle(seed=seed)

    def examples(passes):
        for row, ex in enumerate(ds):
            try:
                yield encode_masked(tokenizer, ex, spec, fmt, loss)
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"dataset row {row}, pass {passes}: {exc}") from exc

    return _write_blocks(examples, n_blocks, seq_len, spec, seed=seed, cache_dir=cache_dir, loss=loss,
                         allowed=max_passes if max_passes is not None else (math.inf if spec.repeat else 1),
                         tokenizer=tokenizer,
                         info={"source": source, "format": fmt, "source_rows": len(ds)})


def _write_blocks(examples, n_blocks, seq_len, spec, *, seed, cache_dir, loss, allowed, tokenizer, info):
    """Pack ``examples(pass_number)`` (an iterable of ``(ids, mask)``, mask None for a full-sequence loss) into
    ``n_blocks`` rows of ``seq_len`` tokens, pass after pass up to ``allowed`` passes, and describe the result."""
    import numpy as np

    folder = tempfile.TemporaryDirectory(prefix="training-tokens-", dir=cache_dir)
    token_path, mask_path = Path(folder.name) / "tokens.bin", Path(folder.name) / "mask.bin"
    masked = loss != "all"
    required = n_blocks * seq_len
    written = used_examples = passes = discarded = trained = 0
    digest, mask_digest = hashlib.sha256(), hashlib.sha256()
    try:
        with token_path.open("wb") as stream, (mask_path.open("wb") if masked else _Null()) as mask_stream:
            while written < required:
                if passes >= allowed:
                    raise ValueError(f"dataset provides {written} tokens in {passes} pass(es), but the plan needs "
                                     f"{required}; reduce --steps or explicitly use --repeat-data")
                passes += 1
                for ids, mask in examples(passes):
                    used_examples += 1
                    used = min(len(ids), required - written)
                    packed = np.asarray(ids[:used], dtype="<i8").tobytes()
                    stream.write(packed)
                    digest.update(packed)
                    if masked:
                        bits = np.asarray(mask[:used], dtype=np.uint8)
                        mask_stream.write(bits.tobytes())
                        mask_digest.update(bits.tobytes())
                        trained += int(bits.sum())
                    written += used
                    discarded += len(ids) - used
                    if written == required:
                        break
        info = {"dataset": spec.source, "source": info["source"], "split": spec.split, "format": info["format"],
                "options": spec.to_dict(), "examples_used": used_examples, "source_rows": info["source_rows"],
                "passes": passes, "repeated": passes > 1, "shuffled": spec.shuffle, "seed": seed,
                "tokens": written, "blocks": n_blocks, "seq_len": seq_len, "discarded_tail_tokens": discarded,
                "token_stream_sha256": digest.hexdigest(), "token_encoding": "little-endian int64",
                "token_cache_bytes": written * (9 if masked else 8), "loss": LOSS_TEXT[loss],
                "packing": "concatenated examples separated by EOS; attention is not isolated between examples",
                "tokenizer": {"name": getattr(tokenizer, "name_or_path", None),
                              "class": type(tokenizer).__name__, "eos_token_id": tokenizer.eos_token_id},
                **{k: v for k, v in info.items() if k not in ("source", "format", "source_rows")}}
        if masked:
            info.update(loss_tokens=trained, loss_mask_sha256=mask_digest.hexdigest())
        blocks = np.memmap(token_path, dtype="<i8", mode="r", shape=(n_blocks, seq_len))
        mask = np.memmap(mask_path, dtype=np.uint8, mode="r", shape=(n_blocks, seq_len)) if masked else None
        return PreparedData(folder, blocks, info, mask)
    except BaseException:
        folder.cleanup()
        raise


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


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

    def __init__(self, folder, tokens, offsets, profile, source, fmt, loss="all", masks=None, packing="concat"):
        self._folder, self.tokens, self.offsets = folder, tokens, offsets
        self.profile, self.source, self.format, self.loss, self.masks = profile, source, fmt, loss, masks
        self.packing = packing

    @property
    def lengths(self):
        import numpy as np

        return np.diff(self.offsets)

    def __len__(self):
        return len(self.offsets) - 1

    def ids(self, i: int):
        return self.tokens[self.offsets[i]:self.offsets[i + 1]]

    def mask(self, i: int):
        return None if self.masks is None else self.masks[self.offsets[i]:self.offsets[i + 1]]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        for array in (self.tokens, self.masks):
            mapping = getattr(array, "_mmap", None)
            if mapping is not None:
                mapping.close()
        self._folder.cleanup()


def encode_dataset(tokenizer, spec: TrainingData, *, cache_dir: str | None = None,
                   seq_len: int | None = None) -> EncodedData:
    """Validate and tokenize every row of ``spec``'s dataset, in source order, and profile the result.

    The profile is deterministic for the same file (or Hub revision), options and tokenizer, so a plan made from it is
    too. ``encoded_sha256`` covers each example's length and ids; ``execute`` compares it with a fresh encoding. Under
    isolated packing, ``seq_len`` (the plan's row length) is needed for the profile's packing summary: rows, fill and
    what truncation to the row drops."""
    import numpy as np

    ds, source = load_rows(spec)
    columns = getattr(ds, "column_names", None) or list(ds[0])
    fmt = resolve_format(spec, columns)
    loss = resolve_loss(spec, fmt)
    packing = resolve_packing(spec, fmt)
    folder = tempfile.TemporaryDirectory(prefix="training-tokens-", dir=cache_dir)
    path, mask_path = Path(folder.name) / "encoded.bin", Path(folder.name) / "encoded-mask.bin"
    lengths, trained, digest = [], 0, hashlib.sha256()
    try:
        with path.open("wb") as stream, (mask_path.open("wb") if loss != "all" else _Null()) as mask_stream:
            for row, ex in enumerate(ds):
                try:
                    ids, mask = encode_masked(tokenizer, ex, spec, fmt, loss)
                except (ValueError, TypeError, KeyError) as exc:
                    raise ValueError(f"dataset row {row}: {exc}") from exc
                packed = np.asarray(ids, dtype="<i8").tobytes()
                digest.update(np.asarray([len(ids)], dtype="<i8").tobytes())
                digest.update(packed)
                stream.write(packed)
                if mask is not None:          # a full-sequence profile hashes exactly as before masks existed
                    bits = np.asarray(mask, dtype=np.uint8).tobytes()
                    digest.update(bits)
                    mask_stream.write(bits)
                    trained += int(sum(mask))
                lengths.append(len(ids))
        lens = np.asarray(lengths, dtype=np.int64)
        offsets = np.concatenate(([0], np.cumsum(lens))).astype(np.int64)
        tokens = np.memmap(path, dtype="<i8", mode="r", shape=(int(offsets[-1]),))
        masks = (np.memmap(mask_path, dtype=np.uint8, mode="r", shape=(int(offsets[-1]),))
                 if loss != "all" else None)
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
            "encoding": "per example in source order: its length, then its ids with EOS, as little-endian int64"
                        + ("; then its loss mask, one byte per token" if loss != "all" else ""),
            "loss": LOSS_TEXT[loss], "loss_mode": loss, "loss_tokens": trained if loss != "all" else int(lens.sum()),
            "packing": PACKING_TEXT[packing], "packing_mode": packing,
        }
        enc = EncodedData(folder, tokens, offsets, profile, source, fmt, loss, masks, packing)
        if packing == "isolated" and seq_len is not None:
            profile["packed"] = packing_summary(enc, seq_len)
        return enc
    except BaseException:
        folder.cleanup()
        raise


def check_profile(found: dict, expect: dict) -> None:
    """Raise unless ``found`` (a fresh encoding) is the data the plan was made from."""
    keys = ("encoded_sha256", "rows", "tokens", "format", "options", "tokenizer", "packed")
    norm = lambda p: {**p, "options": TrainingData.from_dict(p.get("options")).to_dict()}  # noqa: E731 - legacy specs
    found, expect = norm(found), norm(expect)
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
    shuffle, same token and mask streams), from the encoding instead of the tokenizer."""
    import numpy as np

    order = range(len(enc))
    if spec.shuffle:                      # the permutation Dataset.shuffle(seed) applies to a dataset of this length
        from datasets import Dataset

        order = Dataset.from_dict({"i": np.arange(len(enc))}).shuffle(seed=seed)["i"]

    def examples(_passes):
        for i in order:
            yield enc.ids(int(i)), enc.mask(int(i))

    return _write_blocks(examples, n_blocks, seq_len, spec, seed=seed, cache_dir=cache_dir, loss=enc.loss,
                         allowed=max_passes if max_passes is not None else (math.inf if spec.repeat else 1),
                         tokenizer=tokenizer,
                         info={"source": enc.source, "format": enc.format, "source_rows": len(enc),
                               "encoded_sha256": enc.profile["encoded_sha256"], "verified_against_plan": True})


def packing_summary(enc: EncodedData, seq_len: int) -> dict:
    """What isolated packing does to this encoding at ``seq_len``: rows, how full they are, and what truncation drops."""
    lens = [int(n) for n in enc.lengths]
    rows = pack_rows(lens, seq_len)
    kept = sum(min(n, seq_len) for n in lens)
    over = [i for i, n in enumerate(lens) if n > seq_len]
    lost_trained = 0
    if enc.masks is not None:
        lost_trained = sum(int(enc.mask(i)[seq_len:].sum()) for i in over)
    elif enc.loss == "all":
        lost_trained = sum(lens[i] - seq_len for i in over)
    return {"seq_len": seq_len, "rows": len(rows), "efficiency": round(kept / (len(rows) * seq_len), 4),
            "truncated_examples": len(over), "truncated_tokens": sum(lens[i] - seq_len for i in over),
            "truncated_loss_tokens": lost_trained, "window": PACK_WINDOW}


def _pack_isolated(enc: EncodedData, n_blocks: int, seq_len: int, spec: TrainingData, *, seed: int, cache_dir,
                   max_passes, tokenizer, verified: bool) -> PreparedData:
    """Isolated packing: :func:`pack_rows` in source order, then the rows in a seeded order when shuffling. Each row
    holds whole examples (truncated to the row), their positions restart at 0, an example's first token is never a
    target (nothing before it in its own sequence), and the padding at a row's end is EOS, never a target, with its
    own positions. Rows repeat pass after pass only as ``max_passes`` or ``repeat`` allow."""
    import numpy as np

    eos = _eos(tokenizer)
    rows = pack_rows([int(n) for n in enc.lengths], seq_len)
    order = range(len(rows))
    if spec.shuffle:
        from datasets import Dataset

        order = Dataset.from_dict({"i": np.arange(len(rows))}).shuffle(seed=seed)["i"]
    allowed = max_passes if max_passes is not None else (math.inf if spec.repeat else 1)
    folder = tempfile.TemporaryDirectory(prefix="training-tokens-", dir=cache_dir)
    paths = {k: Path(folder.name) / f"{k}.bin" for k in ("tokens", "mask", "positions")}
    written = passes = trained = used_examples = 0
    digest, mask_digest = hashlib.sha256(), hashlib.sha256()
    try:
        with paths["tokens"].open("wb") as ft, paths["mask"].open("wb") as fm, paths["positions"].open("wb") as fp:
            while written < n_blocks:
                if passes >= allowed:
                    raise ValueError(f"dataset packs into {len(rows)} rows of {seq_len} tokens per pass, but the plan "
                                     f"needs {n_blocks}; reduce --steps or explicitly use --repeat-data")
                passes += 1
                for r in order:
                    ids, mask, pos = [], [], []
                    for i in rows[int(r)]:
                        e_ids = enc.ids(i)[:seq_len]
                        e_mask = (np.ones(len(e_ids), dtype=np.uint8) if enc.masks is None
                                  else np.asarray(enc.mask(i)[:seq_len], dtype=np.uint8).copy())
                        e_mask[0] = 0                  # predicted from nothing in its own sequence
                        ids.append(np.asarray(e_ids, dtype="<i8"))
                        mask.append(e_mask)
                        pos.append(np.arange(len(e_ids), dtype="<i4"))
                        used_examples += 1
                    pad = seq_len - sum(len(x) for x in ids)
                    if pad:
                        ids.append(np.full(pad, eos, dtype="<i8"))
                        mask.append(np.zeros(pad, dtype=np.uint8))
                        pos.append(np.arange(pad, dtype="<i4"))
                    row_ids, row_mask = np.concatenate(ids), np.concatenate(mask)
                    ft.write(row_ids.tobytes())
                    fm.write(row_mask.tobytes())
                    fp.write(np.concatenate(pos).tobytes())
                    digest.update(row_ids.tobytes())
                    mask_digest.update(row_mask.tobytes())
                    trained += int(row_mask.sum())
                    written += 1
                    if written == n_blocks:
                        break
        info = {"dataset": spec.source, "source": enc.source, "split": spec.split, "format": enc.format,
                "options": spec.to_dict(), "examples_used": used_examples, "source_rows": len(enc), "passes": passes,
                "repeated": passes > 1, "shuffled": spec.shuffle, "seed": seed,
                "tokens": n_blocks * seq_len, "blocks": n_blocks, "seq_len": seq_len,
                "rows_per_pass": len(rows), "packing_window": PACK_WINDOW,
                "token_stream_sha256": digest.hexdigest(), "token_encoding": "little-endian int64",
                "token_cache_bytes": n_blocks * seq_len * 13, "loss": LOSS_TEXT[enc.loss],
                "packing": PACKING_TEXT["isolated"], "loss_tokens": trained, "loss_mask_sha256": mask_digest.hexdigest(),
                "tokenizer": {"name": getattr(tokenizer, "name_or_path", None),
                              "class": type(tokenizer).__name__, "eos_token_id": tokenizer.eos_token_id},
                "encoded_sha256": enc.profile["encoded_sha256"], "verified_against_plan": verified}
        shape = (n_blocks, seq_len)
        return PreparedData(folder, np.memmap(paths["tokens"], dtype="<i8", mode="r", shape=shape), info,
                            np.memmap(paths["mask"], dtype=np.uint8, mode="r", shape=shape),
                            np.memmap(paths["positions"], dtype="<i4", mode="r", shape=shape))
    except BaseException:
        folder.cleanup()
        raise
