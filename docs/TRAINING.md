# Train on your data and keep the adapter

Starting with 0.2.0, Loggetta accepts your dataset, includes it in a saved training plan, and exports a reusable
adapter plus the tokenizer and a measured run report. The installation still includes the runtime and kernels:

```bash
python -m pip install --upgrade loggetta
```

This is single-GPU MoE QLoRA through the included experts4bit-qlora runtime. A compatible CUDA device and driver
are required for execution. Planning does not load model weights. Successful planning is an estimate, not a
guarantee that execution cannot run out of memory.

## One command

Prepare a dataset file containing enough text for the requested steps, then run:

```bash
loggetta train Qwen/Qwen3-30B-A3B \
  --dataset ./data/train.jsonl --format text --text-field text \
  --seq 512 --micro-batch 1 --steps 20 \
  --learning-rate 0.0002 --seed 42 \
  --out runs/my-training --adapter-out adapters/my-adapter
```

No checkpoint weights are published or uploaded. Data stays local except when you explicitly select a Hub dataset
and it is downloaded. Logs and run reports identify the source and token hash; they do not embed your example text.
Treat dataset paths, Hub dataset names, and tokenizer/model identifiers in reports as potentially private metadata
before sharing them.

## Inspect, save, execute

```bash
loggetta plan Qwen/Qwen3-30B-A3B \
  --dataset ./data/train.jsonl --format text \
  --seq 512 --micro-batch 1 --steps 20 --out plan.json

loggetta execute plan.json --seed 42 \
  --out runs/my-training --adapter-out adapters/my-adapter
```

Data settings and learning rate are part of the plan. `execute` does not silently replace them. Make a new plan
to change the dataset or token dimensions.

**Your dataset is read before the plan is made.** Every row is validated and tokenized first, before the model's
config is even fetched. The plan carries the result as a data profile:
- examples and tokens;
- tokens per example (p50/p90/p99/max) and a length histogram;
- the tokenizer's identity;
- a hash of the encoded examples.

The planner uses the profile to:
- refuse a plan that would read past the data without permission, saying how many steps read it once;
- turn `--epochs N` into steps;
- warn about examples longer than `--seq`, which packing always splits.

`execute` tokenizes the data again and stops before loading any weights if the result differs from the plan's
profile: the file changed, or the tokenizer did. Plans made before 0.3 have no profile; they, and the Alpaca
demonstration, read only the rows the token budget needs, as before.

Without `--adapter-out`, each execution chooses a fresh `OUT/RUN_ID/adapter` directory. The run report is
`OUT/RUN_ID.json`, preserving the existing report-discovery layout. An existing adapter output path is rejected
before training; it is never overwritten. `execute(plan, out_dir=None)` returns the report to the Python caller
and defaults the adapter directory to `runs/RUN_ID/adapter`.

## Supported data

Local files: JSON arrays, JSONL, CSV, Parquet, or TXT with one text record per line. A local filename is not a Hub
ID. Local files expose a `train` split, and split slices such as `train[:100]` are supported.

Hub datasets:

```bash
loggetta train Qwen/Qwen3-30B-A3B \
  --dataset your-org/your-dataset --dataset-config english --split train \
  --dataset-revision COMMIT_SHA --format text --text-field body \
  --seq 512 --steps 20 --out runs/hub-training
```

`--dataset-config` and `--dataset-revision` are optional Hub-only arguments. Pin a dataset commit and a model
`--revision` for reproducibility. The run report records the requested dataset revision, the dataset fingerprint,
local source-file SHA-256 where applicable, shuffle seed, and the SHA-256 of the exact token stream consumed.
It does not claim an unpinned Hub ID is an immutable revision.

### Text

```json
{"text": "A complete training document or a conversation you have already formatted."}
```

Use `--format text --text-field FIELD` for a differently named column. Text is tokenized without automatically
adding a BOS token; an EOS token separates records.

### Instruction/input/output (Alpaca)

```json
{"instruction": "Summarize the maintenance note.", "input": "Replaced the worn seal and checked for leaks.", "output": "Seal replaced; leak check completed."}
```

Use `--format alpaca`. `input` is optional. Alternative field names are supported through `--instruction-field`,
`--input-field`, and `--output-field`. Formatting uses the explicit `### Instruction`, `### Input`, and
`### Response` template, not an inferred model-specific chat template.

### Text-only chat messages

```json
{"messages": [{"role": "user", "content": "What was repaired?"}, {"role": "assistant", "content": "The worn seal was replaced."}]}
```

Use `--format chat`, optionally `--messages-field FIELD`. The tokenizer must have a chat template. It is applied
with `tokenize=True` and `add_generation_prompt=False`. There is no fallback that guesses a chat format.

`--format auto` infers a format only when the columns indicate exactly one supported choice. Ambiguity is a
request to choose a format explicitly, not permission to guess.

## Token and loss semantics

`--loss` decides which tokens are training targets:

| `--loss` | chat | alpaca | text |
|---|---|---|---|
| `auto` (default) | assistant turns | the response | every token |
| `assistant` | assistant turns | the response | refused: plain text has no assistant turns |
| `all` | every token | every token | every token |

- **Chat:** each assistant message's text trains, together with the end-of-turn marker the template closes it with
  (`<|im_end|>`, `<|eot_id|>`, `<end_of_turn>`, ...). That marker is read from the template itself by rendering a
  probe conversation, so the model learns where to stop. System and user turns, the template's headers and any block
  the template inserts before a reply (Qwen3's empty `<think>` block) are context only.
- **Alpaca:** the response trains, followed by its EOS. The `### Instruction` / `### Input` / `### Response` prompt is
  context only.
- **How assistant tokens are found:** every message is located, in order, in the conversation the template renders,
  and character offsets map them onto tokens. This needs a fast tokenizer. A template that rewrites message text,
  or that does not close an assistant turn where expected, is refused with "use --loss all"; it is never guessed.
- **Plans made before `--loss` existed** keep full-sequence loss, as does the Alpaca demonstration.
- **Gradient accumulation averages over trained tokens.** Each micro-batch's mean loss is weighted by its share of
  the optimizer step's trained tokens, so one step does not depend on how its rows are split into micro-batches. A
  micro-batch with nothing to train is skipped, and so is a step with nothing to train; the receipt counts such
  steps. Full-sequence loss keeps the earlier arithmetic exactly.

Not implemented: tool-call and multimodal chat formats, automatic train/evaluation splitting, or a
validation-loss early-stopping policy.

Examples are concatenated with EOS and cut into fixed-length blocks. Attention can cross example boundaries
within a block. This is continuous text packing, not isolated per-example attention.

The plan consumes exactly:

```text
steps * gradient_accumulation * micro_batch * sequence_length
```

input tokens. This counts input positions, not the number of shifted loss targets. A final example may be cut at
the token budget; the report records its unused tail. By default user data is read in order. `--shuffle-data` makes
the order deterministic for the execution `--seed`.

`--epochs N` (instead of `--steps`) reads the dataset N times. The plan records the derived steps:

```text
steps = floor(N * dataset_tokens / (sequence_length * micro_batch * gradient_accumulation)), at least 1
```

A dataset too short for the requested steps is refused by the plan, which suggests the steps that read it once.
Pass `--epochs`, or explicitly add `--repeat-data`, to read it more than once. Repetition and passes are recorded,
not disguised as more unique examples. Repeated passes use the same selected order. Omitting `--dataset` preserves the historical Alpaca demonstration, which repeats
as needed and is announced in the log.

Tokens are prepared in a temporary memory-mapped file, normally beside the adapter output's parent, and cleaned
up afterward. This avoids retaining every training step's tokens as Python objects. Required cache space is eight
bytes per input token. Dataset-reader buffers, the largest encoded row, and temporary disk capacity are outside
the training memory estimator; choose an output filesystem with enough space.

## What is saved

```text
adapters/my-adapter/
    adapter.safetensors
    adapter_manifest.json
    tokenizer.json / tokenizer_config.json / other tokenizer files
runs/my-training/
    RUN_ID.json
```

The native adapter artifact contains only the trainable LoRA tensors identified by the runtime's own structural
adapter discovery. It does not serialize the full model state, frozen experts, optimizer, scheduler, or RNG state.
The manifest records the base model and resolved revision when available, exact backend setup (including rank,
alpha and adapter precision), dataset record, learning rate, seed, runtime version, tensor shapes/dtypes, and file
checksums. Model checkpoints addressed by local paths must still be available when the adapter is reloaded.

This is **native experts4bit-qlora adapter data, not a PEFT `save_pretrained` artifact** and not an exact training
resume checkpoint. Save time is reported separately from measured training-step time. A failed export marks the
run `SAVE_FAILED`; failed integrity checks do not produce an adapter presented as a successful result.

## Reload it

```python
import torch
from transformers import AutoTokenizer
from loggetta import load_adapter

adapter = "adapters/my-adapter"
model = load_adapter(adapter, device="cuda")
tokenizer = AutoTokenizer.from_pretrained(adapter)
```

The loader verifies artifact checksums, rebuilds the recorded setup through the runtime, validates every saved
tensor name/shape/dtype before copying any adapters, and returns the model in evaluation mode. Format prompts
consistently with the training data. The loader does not merge adapters into base weights or enable remote code
trust. The saved base revision is used when known; a missing immutable base revision remains a reproducibility
limitation, explicitly recorded in the manifest.

For an already prepared matching runtime model, use
`loggetta.backends.experts4bit_adapters.load_adapter_weights(model, adapter)`. Its trainable adapter layout and
dtypes must match the artifact exactly. This is useful for validating a save/reload round trip without reloading
the frozen model weights.

## Validation scope

The test suite covers local/Hub loader wiring, format validation, deterministic packing, shortage/repetition,
saved-plan compatibility, output collision protection, failed-save reporting, and an actual CPU LoRA
train/save/reload round trip using runtime LoRA modules. A separate CUDA test builds a tiny local MoE checkpoint
and exercises the complete pipeline when a GPU is available. CPU tests are not a GPU performance result and do
not establish numerical parity for every model family or runtime configuration.
