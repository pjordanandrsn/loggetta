# Dense adapter training

This is the development executor for dense training plans. The released MoE training path is unchanged. CPU tests
exercise real tiny transformers models, PEFT adapters and e4b dense-offload handles; the CUDA/NF4 path and capacity
calibration require the registered GPU proof and reading before a release claim.

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
loggetta train Qwen/Qwen3-8B --dataset ./train.jsonl --format text --seq 512 --steps 20 \
  --out runs/dense --adapter-out adapters/dense
```

Inspect the selected setup before execution. Fix a priced field when needed, for example `--fix base=nf4` or
`--fix placement=stream`. Estimates are not an out-of-memory guarantee; calibration has not licensed replacing
the dense backend's inferred 20% reserve yet. `expandable_segments` remains a caller-controlled setting.

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
