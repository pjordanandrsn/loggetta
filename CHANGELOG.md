# Changelog

## Unreleased

- **Read the dataset before planning.** Every row of `--dataset` is validated and tokenized before the model is
  described. The plan carries the resulting `data-profile/1`:
  - examples and tokens;
  - length quantiles and a histogram;
  - the tokenizer's identity;
  - a hash of the encoded examples.
- **Plans use the profile.**
  - A plan that would read past the data without `--repeat-data` is refused, with the steps that read it once.
  - `--epochs N` derives the steps.
  - Examples longer than `--seq` are counted, because packing splits them.
- **`execute` re-tokenizes and checks.** It refuses data or a tokenizer that changed since planning, before any
  weights load. The packed token stream is identical to 0.2.0's for the same seed.
- Plans without a profile (older plans, the Alpaca demonstration) keep the 0.2.0 behaviour and wire shape.

## 0.2.0

- Accept local JSON/JSONL/CSV/Parquet/TXT files and Hub datasets, with split/config/revision selection.
- Add explicit text, Alpaca, and text-only chat formats, column mapping, seeded shuffle, and opt-in repetition.
- Keep data settings and learning rate in saved plans; preserve old plan defaults and default wire shape.
- Validate and pack data before model loading, with a temporary memory-mapped token cache and token/source hashes.
- Export native runtime adapters and tokenizer files, with exact setup/identity metadata and SHA-256 checksums.
- Add `load_adapter` and strict pre-copy tensor checks; never overwrite existing adapter output.
- Record artifact paths and export failures in run reports, separately from training-step measurements.
- Keep the complete `pip install loggetta` installation, simplify the README, and document the Accelerate comparison.
- Add CPU data, CLI, persistence, and real-LoRA round-trip coverage plus an optional tiny local MoE CUDA test.
- Chat data works with transformers 5, whose `apply_chat_template(tokenize=True)` returns a `BatchEncoding` by
  default; a test now encodes through a real `PreTrainedTokenizerFast`.

Limitations: full-sequence loss only; no automatic held-out evaluation, optimizer resume, PEFT-format export,
first-class dense planning, or server-launch command. CPU tests are not GPU-family validation.

## 0.1.1

- Install the training runtime and low-bit kernel packages by default.

## 0.1.0

- Initial planner, saved execution plans, training orchestration, and measured run reports.
