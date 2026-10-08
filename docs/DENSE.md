# Dense adapter training

This is the development executor for dense training plans. The released MoE training path is unchanged. CPU tests
exercise real tiny transformers models, PEFT adapters and e4b dense-offload handles; the CUDA/NF4 path and capacity
calibration require the registered GPU proof and reading before a release claim.
Execution requires the explicit opt-in `--allow-development-executor` until the registered DQ7 CUDA proof passes.

## What executes

The executor follows the plan's base precision, placement, attention implementation, loss chunk, adapter rank,
alpha, dtype and target roles. Decoder projections are selected from the same structural description the planner
used. A changed structure, missing checkpoint tensor, unmatched adapter count or mechanism that fails to engage
stops execution. It does not switch to another setup.

The loader starts with transformers' meta model and reads safetensors tensors individually. NF4 quantization
materializes one bf16 decoder linear at a time. For streamed placement, packed codes stay on CPU while small
quantization states stay on the GPU; e4b then creates pinned host homes and its training-prefetch schedule.
Embeddings, the head, norms and biases retain the bf16 storage the planner priced. Loading still needs temporary
space for one linear and its quantization, and a mapped checkpoint may use page cache. The receipt records the
load peak separately from the training peak.

Models need a safetensors checkpoint with unambiguous tensor names. Legacy pickle checkpoints and remote modeling
code are refused. A multimodal checkpoint's text tower is loaded using unique exact suffix matches; this is text
training, not multimodal training. Tied embeddings share one loaded parameter.

## Plan, train and reload

Use the same data validation, assistant loss, packing and schedule controls as [MoE training](TRAINING.md):

```sh
loggetta plan Qwen/Qwen3-8B --dataset ./train.jsonl --format text --seq 512 --steps 20
loggetta train --allow-development-executor Qwen/Qwen3-8B --dataset ./train.jsonl --format text --seq 512 --steps 20 \
  --out runs/dense --adapter-out adapters/dense
```

Inspect the selected setup before execution. Fix a priced field when needed, for example `--fix base=nf4` or
`--fix placement=stream`. Estimates are not an out-of-memory guarantee; calibration has not licensed replacing
the dense backend's inferred 20% reserve yet. `expandable_segments` remains a caller-controlled setting.

DQ7's [actual executor reading](https://github.com/pjordanandrsn/experts4bit-qlora/blob/main/bench/dq7/RESULTS-dq7.md)
is **VOID**: Qwen3-32B resident 4096 was refused before loading, and two registered anchor arms are missing.
Completed Llama allocator estimates fall below their measured peaks at 2048/4096 in both placements. Every completed
streamed full-device plan also understates sampled driver use, by as much as 2,395,904,403 bytes, despite its 20% reserve.
The bitwise tiny CUDA proof passes; no successful out-of-sample or reserve calibration follows.

Every dense plan therefore warns about this measurement. Streamed candidates require at least **2,400,000,000 bytes
(2.4 GB decimal)** admission headroom, **20% of the full device estimate rounded up**, and the normal policy or caller-requested
headroom, whichever is largest. A smaller `--headroom` cannot bypass it. This floor is rounded above the largest observed device-total
shortfall; the proportional floor rounds above DQ7's maximum driver/plan ratio 1.196. A streamed plan outside the
measured subjects, sequences or recipe carries another visible warning. Both floors are conservative admission policy. They do not change the allocator or reserve coefficients and cannot
establish a general fit guarantee. The required margin is recorded in the plan budget and candidate bounds separately from all itemized estimates,
so receipts and DQ4 allocator comparisons keep their original meanings. The development execution opt-in remains.

The receipt identifies backend `dense`, echoes the setup and keeps allocator/reserved/driver peaks and step time
alongside the estimate. Frozen integrity hashes the first and last decoder layers, including their offloaded homes
and quantization states. It is explicitly sampled, not a hash of the whole checkpoint. A frozen-weight mutation or
adapters that do not move marks the run `ALARM` and prevents a successful adapter export.

Dense adapter directories contain PEFT's `adapter_model.safetensors`, `adapter_config.json`, tokenizer files and
Loggetta's manifest. Every file is checksummed. The loader rejects corruption and verifies that the reloaded
adapter keys, dtypes and values equal the exported tensors. Local base snapshots must remain available; an
unpinned Hub revision cannot reproduce a base checkpoint. Optimizer and scheduler state are not saved.

```python
from loggetta import load_adapter

model = load_adapter("adapters/dense", device="cuda")
```

## Validation boundary

The CPU tests reconstruct checkpoint tensors exactly, compare streamed/resident deterministic losses and all LoRA
gradients, execute chunked Qwen3 loss, detect frozen-weight mutation, and train/export/reload real PEFT adapters
without changing logits. CPU streaming tests lower e4b's tensor-size threshold so tiny weights actually stream;
production execution retains e4b's threshold. These tests are correctness evidence, not GPU timing or capacity
evidence. No new dense kernel is proposed: the DQ1 speed primitive was a negative result.

The registered CUDA lane must prove the harness before its reading and test resident/streamed deterministic agreement.
Qwen3-32B is an in-sample reproducibility anchor. A different Qwen size and a non-Qwen family must test whether the
allocator estimate ever falls below the measured peak. The DQ4 overestimate brackets (4% resident and 10% streamed)
are predictions for those new subjects, rather than a pass condition tuned on the fitting subject. Calibration will use
only the dense backend's own receipts under a separate registered replacement rule.

**DQ7's result (2026-10-08).** The estimate held on Qwen3-14B but fell below the measured peak on Llama-3.1-8B at 2048
and 4096 tokens, by the same bytes resident and streamed. And the plan's device total was below the measured driver
peak on every streamed arm, by up to 2.4 GB. Until a re-read clears it, leave that much headroom on a streamed plan.


## Full-logit workspace correction

The DQ7 diagnosis identifies a concrete missing buffer in the full-logit loss path. At the pinned torch 2.8 and
transformers 5.18 versions, CPU and CUDA operator census both observe three distinct fp32 `[tokens, vocabulary]`
tensors simultaneously in log-softmax backward: saved log-softmax output, incoming NLL gradient, and outgoing
logits gradient. Dense full-logit loss now prices **12 bytes per logit**, replacing the old 10-byte allowance.
The activation coefficient, chunked-loss calculation and inferred 20% reserve are unchanged.

The [census and receipts](../bench/dq7-loss-diagnosis/loss-cuda.json) preserve the measured tensor bytes and runtime
source hashes. This is a derived workspace correction, not a new whole-model reading or a refitted coefficient.
At Llama 4096 the extra `2 × tokens × vocab` term is 1,050,673,152 bytes, exceeding DQ7's observed allocator miss by
about 0.39 GB; that surplus remains visible. DQ7 stays VOID, and no capacity pass, reserve calibration, gate removal
or release follows from recomputing historical estimates. Load-cache attribution and a full-model reread require
reviewed telemetry and registration. The developmental execution opt-in remains.
