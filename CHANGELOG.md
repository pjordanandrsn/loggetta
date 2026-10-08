# Changelog

## Unreleased

- **A second backend, `dense`, describes dense decoder models without reading weights** (config + a meta-device
  tree). `inspect` and `plan` now reach a dense model through it instead of stopping at the MoE loader's refusal.
  - Every decoder-layer linear is classified by structural role from its shape: attention in/out, MLP in/out. Names
    only break ties, so fused `qkv_proj` / `gate_up_proj` (Phi-3) classify like separate projections; anything left
    over stays frozen and is reported.
  - Parameter counts, tied heads, and attention implementations are reported, including softcapping that SDPA would
    drop (Gemma 2), along with whether experts4bit-qlora's chunked loss covers the class.
  - MoE, pre-quantized and non-decoder models are refused in words.
  - Training plans for dense models come next. Today such a plan says the backend described the model but does not
    plan the workload yet.

## 0.3.1 — 2026-10-08

**0.3.1.** Documentation only; the package code is identical to 0.3.0. The PyPI page is now shorter than the README and
consistent with it, and its replication link works again: it pointed at an experts4bit-qlora changelog fragment that the
0.49.0 release folded away (#15). The Unsloth comparison is quoted as GPU time, as experts4bit-qlora 0.50.0 quotes it:
Unsloth spends 1.92× e4b's GPU time per step, and 2.80× its wall-clock time on an AMD EPYC 7713 host (#16). It still needs
experts4bit-qlora 0.49.0 or later.

## 0.3.0 — 2026-10-08

**0.3.0.** Train on your own data and keep the adapters. This is the first PyPI release since 0.1.3; 0.2.0 was never
published, and its work ships here. Loggetta now reads your dataset (local files or the Hub), checks and tokenizes it before
any model loads, plans from what it found, trains on assistant tokens with isolated packing and a cosine schedule by default,
and exports a reusable adapter. Upgrade if you want to train on your own data. It needs experts4bit-qlora 0.49.0 or later
(0.48.0 fails under transformers 5.19 and lacks the planner's row and bucket controls).

### Training defaults and planning from the data

- **Isolated packing by default for chat and alpaca data** (`--packing auto|concat|isolated`).
  - Whole examples go into rows (deterministic best fit), positions restart per example, and an example's first token
    and the row padding never train. transformers then keeps each example's attention to itself; the loop passes
    `use_cache=False`, which that isolation requires.
  - Profiles record the packing at the planned seq (rows, fill, truncation), and plans count rows.
  - The per-row attention mask is priced (micro-batch × seq² × 3 B). Its speed cost (no flash attention) is stated,
    not modelled.
  - Models that mix tokens through a recurrent state are refused isolation.
  - Earlier plans keep concatenation.
- **Warmup + cosine learning-rate schedule by default for `train` and `plan`** (`--lr-schedule cosine|constant`,
  `--warmup-steps`).
  - Linear warmup over 3% of the steps, then cosine decay to 10% of the peak.
  - The plan states it, and run reports record the applied rates.
  - Saved plans and direct `Workload` callers keep the constant rate.
- **Assistant-only loss by default for chat and alpaca data (`--loss auto|all|assistant`).**
  - Chat trains each assistant turn's text plus the end-of-turn marker the template closes it with. The marker is
    read from the template.
  - Alpaca trains the response and its EOS.
  - Masks come from character offsets over the rendered conversation. Templates that rewrite text are refused,
    never guessed.
  - Earlier plans and the demonstration keep full-sequence loss.
- **Gradient accumulation averages over trained tokens.** Each micro-batch is weighted by its share of the step's
  trained tokens, so micro-batch × grad-accum splits of the same rows give the same step. Steps with nothing to
  train are skipped and counted.
- **Profiles and receipts record the loss.** Profiles record the loss and its trained-token count, and hash the mask.
  Receipts carry `loss_tokens` and `loss_mask_sha256`.
- **Read the dataset before planning.** Every row of `--dataset` is validated and tokenized before the model is
  described. The plan carries the resulting `data-profile/1`:
  - examples and tokens;
  - length quantiles and a histogram;
  - the tokenizer's identity;
  - a hash of the encoded examples.
- **Plans use the profile.**
  - A plan that would read past the data without `--repeat-data` is refused, with the steps that read it once.
  - `--epochs N` derives the steps.
  - Examples longer than `--seq` are counted: concatenated packing splits them, isolated packing truncates them,
    and the plan states the tokens dropped.
- **`execute` re-tokenizes and checks.** It refuses data or a tokenizer that changed since planning, before any
  weights load. With concatenated packing and full-sequence loss, the packed token stream is identical to 0.2.0's
  for the same seed; chat and Alpaca data now default to isolated packing, whose stream differs.
- Plans without a profile (older plans, the Alpaca demonstration) keep the 0.2.0 behaviour and wire shape.

### Your data and reusable adapters (developed as 0.2.0, never published)

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
