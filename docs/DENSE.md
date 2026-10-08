# Dense adapter training

This is the development executor for dense training plans. The released MoE training path is unchanged. CPU tests
exercise real tiny transformers models, PEFT adapters and e4b dense-offload handles. The tiny CUDA/NF4 correctness
proof passed, but DQ7's capacity reading is VOID. Execution still requires `--allow-development-executor`: removing
it needs separately reviewed prospective capacity and calibration evidence, the registered 24 GB boundary reading,
and an evidence-linked gate-removal change. DQ8 has not been drawn. See [D7's release boundary](../bench/D7-PREREG.md).

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
The resident Llama 4096 plan understates driver use by 820,943,251 bytes as well. The mandatory margin below applies
to streamed candidates; it does not establish resident capacity.
Individual Qwen3-14B allocator estimates were above their observed peaks; those partial observations do not license
a lane pass. Qwen3-32B's anchor residuals remain unattributed. The tiny CUDA proof passed, but no successful
out-of-sample or reserve calibration follows.

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

After the workspace correction below, prospective capacity and reserve validation require a new reviewed
registration. Known-subject diagnostic readings can investigate load-cache attribution, but cannot substitute
for new out-of-sample subjects or the separate 24 GB boundary. No 5090 reserve transfers to a 4090.

The [current pinned config-only sweep](../evidence/2026-10-08-dense-plan-sweep/README.md) includes the streamed floor
and full-logit correction. The unpinned 2026-10-07 table predates both; comparison is by model name only. Four
model/card choices change from feasible to refused. Neither sweep establishes capacity.

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
