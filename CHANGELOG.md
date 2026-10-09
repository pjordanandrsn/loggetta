# Changelog

## 0.4.0 — 2026-10-09

**0.4.0.** Dense models can now be planned, and run behind a development flag. MoE training estimates now price the
`grouped_nf4` kernel's backward pass. Serve plans for one sequence say which speed-ups the server runs. And resident
`grouped_nf4` training runs again with experts4bit-qlora 0.49.0 or later.

- **Dense models are development-gated.** Loggetta describes and plans dense decoder models. Their plans are
  estimates checked in sample: out of sample (DQ7, VOID) they missed, and the DQ10 reading is pending. Dense training
  runs only with `--allow-development-executor` until DQ8's 24 GB reading passes. Dense training is not supported yet.
- **The DQ10 reserve policy is opt-in.** It is a registered hypothesis for registered dense runs. Shipped defaults do
  not change.
- **The MoE training estimate prices the `grouped_nf4` backward pass.** OLMoE estimates on the RTX A2000 were about
  0.2 GiB under the measured peak; in sample they are now 90–114 MiB over. Near a budget, a plan may now pick the
  reference kernel or host residency where it picked resident `grouped_nf4` before.
- **Single-stream serve plans** name the B=1 levers a default server runs for the model's family, and quote a decode
  figure only from a receipt of the same setup.
- **Fix: resident `grouped_nf4` training no longer stops before its first step** with experts4bit-qlora 0.49.0 or
  later (#47). Loggetta 0.3.x raised `AbsmaxCompressedError` at its frozen-expert digest; the workaround was
  `E4B_ABSMAX_DQ=0`.

**Requirements.** grouped-nf4-gemm 0.42.0 or later (was 0.41.0). PEFT 0.21.2 or later is now a base dependency. The
experts4bit-qlora floor stays 0.49.0.

### MoE training: the `grouped_nf4` backward pass (#43, #44, #45)

- **What changed.** A `grouped_nf4` training estimate carries a new device line, `grouped_nf4 MoE backward (above the
  activation item)`: how much `boundaries + padded LoRA delta + fused workspaces` exceeds experts4bit-qlora's activation
  item (`boundaries + max(logits, 2 x one layer)`), or nothing when it does not.
  - **Padded LoRA delta:** grouped-nf4-gemm's `_lora_delta_padded` holds `rows x (H + first_out + I)` in the adapter
    dtype. Its rows are data-dependent, so they are a bound in shape, not a measured constant: `min(E, T x top_k) x T`
    for the single padded block, capped where the `auto` route would take the loop (2 GiB padded block), on the
    single-block ladder for fp32 adapters (grouped-nf4-gemm 0.44.0's `NF4_QLORA_SINGLE_LADDER=auto`: at most 1.5625x),
    and `2 x T x top_k` once a call has 16,384 routed rows and pads by buckets.
  - **Fused workspaces:** `5 x T x top_k x first_out` bf16.
  - The expert LoRA gradients that are live there are already priced, in the `adapter gradients` line, and are not
    counted again.
  - grouped-nf4-gemm exposes no sizing helper, so its route constants and ladder are mirrored, and a test pins them
    against the installed package. The kernel's run-time overrides (`NF4_QLORA_*`) are listed as unmodelled.
  - The `grouped-nf4-gemm` floor is now 0.42.0, the first release whose bucketed padding is the default.
- **Why.** loggetta#44 replayed the allocator at OLMoE's `grouped_nf4` training peak on the RTX A2000: the peak is in
  an MoE layer's backward, with 500.9 MiB of padded delta and 167.1 MiB of fused workspaces live, where the estimate
  took the loss branch. Estimates were 195–219 MiB under the allocated peak.
- **In-sample only.** Against the committed OLMoE `grouped_nf4` receipts (#29's A2000 and FP1's RTX 5090, #44's
  replays) the estimate now sits 90–114 MiB over the allocated peak, the cost of a bound. The reference kernel, the
  Qwen3-30B receipts (the loss branch is larger) and the granite receipts do not move; the granite runs stay 56–69 MiB
  under, which this term does not explain. `bench/gnf4_terms_in_sample.py` prints the table.
- **Plans near a budget may change setup.** A `grouped_nf4` training plan is now larger, so where resident
  `grouped_nf4` used to fit just barely, a plan may now pick the reference kernel or host residency.
- **Dense plans are unchanged:** byte-identical on a 36-case matrix, before and after.
- **Tests:** `tests/test_gnf4_training_terms.py`, `tests/test_dense_unchanged_by_moe_terms.py`.
- **Plans against the driver's peak, after this change.** `bench/plan_vs_driver.py --replan` on the committed training
  receipts (`evidence/2026-10-09-moe-plan-vs-driver-after-gnf4`, in-sample): with no receipts on file, every OLMoE plan
  is over its driver peak (driver / plan 0.950–0.987, from 1.001–1.105). One plan is still under: granite-3.1-3b-a800m
  replanned with its siblings' receipts on file, at 1.014, because its reserve line borrows OLMoE's (#43, item 2).

### MoE plans against measured driver peaks (report only, #29)

`bench/plan_vs_driver.py` reads every committed receipt with a plan total and a driver peak, and can replan the
training receipts with today's code (`evidence/2026-10-08-moe-plan-vs-driver`).

- **Serving plans hold:** all 22 are over their driver peak, including SV5–SV7's registered RTX 4090 plans.
- **MoE training plans on the RTX A2000 do not:** today's OLMoE plans sit under the driver peak (driver / plan up to 1.105).
  The allocator estimate is 0.213 GiB under the allocated peak on both placements, and no MoE training residual is
  learned.

No code, coefficient or default changed.

### Single-stream serve plans

- **What changed.**
  - **The levers.** A serve plan for one sequence names the B=1 levers that a default `serve_paged` runs for the
    model's family, as the installed experts4bit-qlora resolves them: its family-scoped fused stack (fused q/k/v and
    three glue folds), with the read it rests on or the read the family lacks. An experts4bit-qlora without the
    family-scoped default reads as "off unless set". Nothing here keeps its own table of families.
  - **The figure.** The plan's performance section quotes a decode figure only from a serve receipt of exactly the
    same setup: same model, GPU name and driver, setup, and experts4bit-qlora and grouped-nf4-gemm versions, at one
    sequence. The figure is labelled measured, with the receipt's ID and the prompt and new-token counts it was
    measured at (a decode step grows with the KV position). Anything less shows nothing: loggetta still predicts no
    throughput, and no figure is interpolated or carried between hosts.
  - **The receipt.** `bench/serve_validate.py --concurrency 1` traces the engine's steps (`E4B_PAGED_STEP_TRACE`) and
    records `measured.decode_step_ms_b1`: the median whole-step time of the steps that decoded one row and ran no
    prefill. It also records the decode's device time and the step count, from `loggetta.measure.decode_step_b1`.
- **Why.** On experts4bit-qlora, one sequence's decode speed now depends on the model family as well as the setup:
  the B=1 fused stack is on by default only where it was read (lane P115, experts4bit-qlora #1361 and #1379). A plan
  should say which levers apply, and quote a speed only where one was measured on the same setup.
- **Tests:** `tests/test_single_stream.py`:
  - the trace reader;
  - the levers line with and without the family-scoped default;
  - the exact-match rule, each mismatch showing nothing;
  - a planner round trip.

### Dense models (development-gated)

- **Dense execution prototype.** Dense training plans dispatch through transformers, PEFT linear LoRA and the
  existing e4b chunked-loss and dense-offload engines. Checkpoint loading reads one safetensors tensor at a time;
  NF4 quantization handles one decoder linear at a time, and streamed codes stay on the host. The selected setup
  is checked against the loaded structure and echoed in a dense ExecutionReceipt. Sampled frozen-layer integrity,
  adapter movement, memory and timing share the measured loop with MoE. Dense adapters use PEFT with checksummed
  export/reload; existing MoE adapters retain their native format. PEFT >=0.21.2 is a base dependency for all installs;
  0.21.2 is the tested floor. Dense execution requires `--allow-development-executor` until a registered capacity
  reading passes (DQ8, 24 GB).
  CPU tiny-model checks cover exact checkpoint reconstruction, resident/streamed loss and gradient equality,
  actual chunked loss and adapter precision on reload. The tiny CUDA/NF4 proof passed; DQ7's capacity reading is
  VOID and reserve calibration remains unlicensed. The e4b floor remains 0.49.0.
- **Dense adapter-training plans.** The dense backend now plans `train` for dense models.
  - **Candidates:** bf16 base resident, NF4 base resident, NF4 base streamed from pinned host memory. Speed order:
    bf16 before NF4, resident before streamed, per experts4bit-qlora's measured DQ lanes.
  - **Estimate:** itemized.
    - *derived:* weights, embeddings/head, LoRA (fp32 r16 α32 on every classified projection by default),
      gradients, AdamW, and experts4bit-qlora's `offload_plan` for streaming.
    - *heuristic:* activations scaled to DQ4's measured slope; the chunked-loss workspace from `chunked_loss_bytes`.
    - Linear biases are counted once, with norms and other parameters.
  - **Checked in-sample:** the activation slope is fitted to DQ4, and against DQ4's measured peaks at Qwen3-32B the
    estimate is +1.9 to +3.1% resident and +3.0 to +8.7% streamed, never under.
  - **Out of sample, it misses** (experts4bit-qlora DQ7, RTX 5090, 2026-10-08; VOID as a lane because one anchor arm
    could not run). Qwen3-14B's allocator estimates were above their peaks (+1.5 to +4.9%); Llama-3.1-8B's fell below
    at 2048 and 4096 tokens (-2.0 to -6.4%). The plan's device total, reserve included, was below the measured driver
    peak on every streamed arm, by up to 2.4 GB, and on resident Llama 4096 by 0.82 GB. None of this is a capacity or
    calibration pass, and dense execution stays behind `--allow-development-executor`.
  - **Refused, in words:** streaming that would free nothing, attention that would change semantics, a sequence past
    the model's positions, int8.
  - `bench/dense_plan_sweep.py` plans 14 pinned configs on stated 16–48 GB cards
    (`evidence/2026-10-08-dense-plan-sweep`; estimator-only, with enforced streamed headroom).
    The unpinned 2026-10-07 table predates #26/#28. Comparing by model name, four feasible cells now refuse:
    Qwen2.5-72B at 32 GB, Qwen3-32B at 16 GB, Gemma-2-9B and Gemma-3-12B at 24 GB. Neither table proves capacity.
- **Reserve slack is learned per backend.** A plan's reserve, including the fallback, comes only from its own backend's
  receipts. The GPU's CUDA context and host baseline stay shared. Receipts that do not name their backend count as
  experts4bit's, so MoE plans are unchanged.
- **A second backend, `dense`, describes dense decoder models without reading weights** (config + a meta-device
  tree). `inspect` and `plan` now reach a dense model through it instead of stopping at the MoE loader's refusal.
  - Every decoder-layer linear is classified by structural role from its shape: attention in/out, MLP in/out. Names
    only break ties, so fused `qkv_proj` / `gate_up_proj` (Phi-3) classify like separate projections; anything left
    over stays frozen and is reported.
  - Parameter counts, tied heads, and attention implementations are reported, including softcapping that SDPA would
    drop (Gemma 2), along with whether experts4bit-qlora's chunked loss covers the class.
  - MoE, pre-quantized and non-decoder models are refused in words.
  - Config, model-build and chunked-loss errors become refusals naming the exception type; refused descriptions
    can be copied or pickled without recursion.
  - Dense training plans and their development executor are described above; other dense workloads remain planned only.
- **Dense pricing corrections** (structural; execution, reserve/context hypotheses and observation licensing do not
  change):
  - **Full-logit loss** is priced at twelve bytes per logit, not ten: the pinned CPU/CUDA census sees three distinct
    fp32 logits-sized tensors live together in the log-softmax backward.
  - **Small frozen weights** that the offload engine keeps on the device below its streaming threshold are counted,
    with their quantization statistics and the trainable LoRA kept separate (#39).
  - **NF4 quantization state** is counted in full: the two private codebooks and the offset, with packed codes and
    nested statistic blocks rounded up for partial blocks (#41).
- **Streamed admission headroom after DQ7.** The VOID DQ7 reading understates device use on every completed streamed
  arm, by up to 2.40 GB, and resident Llama 4096 driver use by 0.82 GB. Dense plans warn, and streamed admission keeps
  at least max(2.4 GB, 20% of the full device estimate) of headroom even when the caller asks for less. Receipts still
  compare the original estimate, without the headroom.
- **The DQ10 memory hypothesis (opt-in).** A scoped reserve policy prices a training reserve and driver overhead
  separately for registered dense runs. Plans, admission and execution carry the frozen hypothesis and the policy's
  SHA-256; an invalid selection or changed pricing is refused before loading. Shipped defaults, allocator pricing,
  streamed headroom and the executor opt-in do not change. Raw diagnostic and holdout observations cannot teach the
  planner an overhead or a capacity calibration.
- **D7 reserve calibration, registered and read.** The proposed D7 dense allocator-reserve calibration is registered
  with six sequence holdouts, model/placement cohorts, byte-level held-out gates and strict observation scopes. Its
  reader has fixed derivation and holdout cohorts, exact integer fractions, both byte-level holdout gates and the
  original 20% reporting; it recomputes DQ7 through the merged reducer and keeps source checksums. Neither licenses a
  planner observation or changes a reserve or execution gate.
- **The docs state the current gates.** The tiny CUDA correctness proof passed; DQ7 stays VOID. The capacity,
  calibration and 24 GB boundary requirements are explicit, the margin is described as enforced policy, and the older
  config-only sweep is labelled historical estimator output.

### Fixes

- **Frozen integrity fails closed.** An empty sampled expert digest now marks a MoE run `ALARM`; checking no frozen
  bytes cannot pass integrity. Normal non-empty frozen expert digests behave as before.
- **Legacy observation licensing with nullable plan metadata.** Receipts with a null plan or null constraints keep
  their existing `licensed_for` scope instead of raising. Raw DQ7/DQ9/DQ10 records and explicit DQ10 policy receipts
  stay excluded from observation import (#37).
- **Resident `grouped_nf4` training no longer stops at the frozen-expert digest (#47, #48).**
  - **What changed.** `_expert_digest` hashes the frozen expert absmax as it is stored. That is the double-quantized
    payload (`<which>_absmax_q`, `_s`, `_off`, `_code`) when experts4bit-qlora compressed it, and the fp32 buffer
    otherwise. Nothing is decompressed. With the fp32 absmax the digest is byte-for-byte the earlier one.
  - **Why.** Since experts4bit-qlora 0.49.0, `enable_fast_train` double-quantizes the absmax by default for resident
    training, and a guard under the old `<which>_absmax` name raises on any use. With loggetta 0.3.x and
    experts4bit-qlora 0.49.0 or later, resident `grouped_nf4` training raised `AbsmaxCompressedError` before step 1.
    The workaround was `E4B_ABSMAX_DQ=0`. Host residency and the reference kernel were not affected.
  - **Evidence.** Reproduced on the released pair (loggetta 0.3.1 + experts4bit-qlora 0.50.0) on an RTX A2000. The
    fixed digest trains the same setup to an OK receipt (`evidence/2026-10-09-a2000-absmax-digest-repro`).
  - **Tests.** `tests/test_absmax_digest.py` builds the model through `prepare_qlora_training`, which calls
    `enable_fast_train`, on a tiny local MoE on the CPU, with experts4bit-qlora's default. It fails before the fix and
    passes after, with experts4bit-qlora 0.49.0 and 0.50.0. The released-backend CI job fails if this test skips.

### Docs

- **The memory headline says what it covers.** README and PYPI's planner memory check (24.54 GiB estimated, 24.34
  measured) is one in-sample case, and they say so. They point to the MoE plan-vs-driver audit, now with the replan
  after the `grouped_nf4` term.

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
