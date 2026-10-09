# What is live at the MoE training peak the estimate misses (#43)

**Question** (#43): on the RTX A2000, OLMoE-1B-7B training with the `grouped_nf4` expert kernel peaked about 0.2 GiB
above the plan's allocator estimate, on both placements. With the `reference` kernel it did not. What is live at that
peak?

**Method.** `bench/train_residual.py` re-runs a committed receipt's model, setup and workload through the planner
and the executor, with the caching allocator's history recorded (Python stacks). It then replays the trace from the
first training step and groups the allocations live at the training peak by the innermost frame that names a
mechanism. Three runs, one process each, replicating the 2026-10-04 receipts (micro-batch 2 × 512, 12 steps, NF4
experts, r8 bf16 adapters):

| run | receipt replicated |
|---|---|
| B | `2026-10-04-rtx-a2000/OLMoE-1B-7B-0924-device-reference-t1024-20261004T175624Z` (reference kernel, resident) |
| A2 | `…-device-grouped_nf4-t1024-20261004T175201Z` (grouped_nf4, resident), with `E4B_ABSMAX_DQ=0` |
| C | `…-host-grouped_nf4-t1024-20261004T182135Z` (grouped_nf4, experts on host) |

A2 sets `E4B_ABSMAX_DQ=0` because the 2026-10-04 receipts predate experts4bit-qlora #1292 and ran with an fp32 expert
absmax. With today's default the resident grouped run stops at Loggetta's frozen-integrity digest before step 1. That
is a separate bug, reported on the bus.

```sh
python bench/train_residual.py evidence/2026-10-04-rtx-a2000/<receipt>.json --out runs/train-residual/<run>
python bench/train_residual.py --regroup runs/train-residual/B runs/train-residual/A2 runs/train-residual/C
python bench/train_residual.py --summarize B A2 C > summary.md    # each <run>/residual.json + its receipt, as here
```

The runs used loggetta `fa1ed17` plus this script, experts4bit-qlora `5905a0ce`, grouped-nf4-gemm `b4f93f1`, torch
2.8.0+cu128, transformers 5.18.0, and one RTX A2000 12GB. Each run directory holds its `residual.json` (the groups at the
peak, re-keyed by `--regroup`, which re-checked the replayed peak and event) and the run's own ExecutionReceipt.
The snapshots (~350 MB each) are not committed.

## What the script printed

| live at the training peak, MiB | B | A2 | C |
|---|---|---|---|
| frozen expert stacks (resident) | 3456.0 | 3456.0 | 0.0 |
| dense weights (bf16) | 909.3 | 909.3 | 909.3 |
| other (incl. unattributed frames) | 421.5 | 96.9 | 96.9 |
| logits and loss | 196.5 | 0.0 | 0.0 |
| optimizer state (adamw) | 232.0 | 232.0 | 232.0 |
| expert LoRA adapters | 112.0 | 112.0 | 112.0 |
| checkpointed layer inputs | 60.0 | 0.0 | 0.0 |
| attention LoRA adapters | 4.0 | 4.0 | 4.0 |
| padded LoRA delta (gnf4 _lora_delta_padded: G x widest rows) | 0.0 | 500.9 | 496.5 |
| expert LoRA gradients | 0.0 | 105.0 | 105.0 |
| fused grouped kernel workspaces | 0.0 | 167.1 | 167.1 |
| frozen expert stacks (one layer staged) | 0.0 | 0.0 | 216.0 |
| **replayed peak** | **5391.3** | **5583.2** | **2338.9** |
| measured allocated peak | 5394.2 | 5583.8 | 2339.1 |
| plan's allocator estimate | 5384.5 | 5384.5 | 2144.5 |
| residual (measured − estimate) | +9.7 | +199.3 | +194.6 |

- B: kernel reference, residency device, E4B_ABSMAX_DQ=None; plan's activations 555.2 MiB: 16 saved layer inputs (T x H bf16) + max(logits and loss in bf16+fp32+fp32 = 0.52 GB, 2 x one layer's recompute = 0.20 GB) at T=1024
- A2: kernel grouped_nf4, residency device, E4B_ABSMAX_DQ='0'; plan's activations 555.2 MiB: 16 saved layer inputs (T x H bf16) + max(logits and loss in bf16+fp32+fp32 = 0.52 GB, 2 x one layer's recompute = 0.20 GB) at T=1024
- C: kernel grouped_nf4, residency host, E4B_ABSMAX_DQ=None; plan's activations 555.2 MiB: 16 saved layer inputs (T x H bf16) + max(logits and loss in bf16+fp32+fp32 = 0.52 GB, 2 x one layer's recompute = 0.20 GB) at T=1024

## Reading

- **The reference kernel's peak is the loss, and the estimate holds there (+9.7 MiB).** Every static item is live at
  exactly its planned size: frozen stacks, dense weights, adapters, optimizer state. The logits buffers sit in "other",
  allocated from C++ with no Python frame (2 × 196.5 MiB).
- **With `grouped_nf4`, the peak moves into an MoE layer's backward pass,** on both placements.
  - Beyond the static items, 869.9 MiB is live there (A2). The estimate's dynamic items price 671.2 MiB (activations
    555.2 + adapter gradients 116), a difference of +198.7 MiB.
  - **The padded LoRA delta, 500.9 MiB** (A2; 496.5 in C). This is grouped-nf4-gemm `kernel/nf4_qlora.py`
    `_lora_delta_padded`:
    - line 540 is `x = zeros(G·widest, K)`;
    - line 542 is `d = bmm(bmm(x, Aᵀ), Bᵀ)`, shaped `[G·widest, N]`;
    - both are in the adapter dtype, bf16 here. G counts the experts with routed rows, and widest is the busiest
      expert's routed rows.
    - The 199.75 MiB buffers are 51,136 rows × 2,048 × 2 B. With G ≤ 64 and widest ≤ T = 1,024, only G = 64 and
      widest = 799 fit. The routed rows were T·top_k = 8,192, a mean of 128 per expert.
    - Live at the peak: x and d for gate_up (`G·widest × H` and `× 2I`), and x for down (`G·widest × I`).
  - **Fused kernel workspaces, 167.1 MiB:** four 32 MiB buffers, T·top_k × 2I × 2 B (8,192 × 2,048 × 2), from
    `fused_experts_train_forward`, `fused_grouped_lora`, `gemm_4bit_grouped` and `_scaled`, plus 39 MiB in
    `nf4_qlora` forward.
  - **Expert LoRA gradients, 105 MiB,** and 96.9 MiB of other allocations.
- **Why the estimate misses it.** The activation item is `boundaries + max(logits, 2 × layer)`. Its layer term counts
  T × top_k routed rows, has no G·widest padding, and does not depend on the kernel. For OLMoE it takes the loss branch
  (0.52 GB against 0.20 GB), while the grouped kernel's backward phase is larger.
- **Context:** grouped-nf4-gemm's `_lora_delta_padded` has a compact path, off unless `NF4_QLORA_COMPACT_DELTA=1`. It
  was not engaged in these runs.

## What this does not say

- **How large the term is on another card, model or shape.** widest depends on the data, through routing
  imbalance, and is at most T per expert. So the padded term is bounded by `E·T·(H + 2I + I)·adapter bytes`: 640 MiB at
  this shape. That is a formula and a bound, not a proposed constant.
- **Whether today's code matches 2026-10-04's exactly.** Today's peaks are a little lower: A2 5.453 GiB against
  5.471, C 2.284 against 2.308, and B 5.268 identical. The residuals were +199.3 / +194.6 MiB, against 2026-10-04's +218.1 / +219.0 MiB (0.213 / 0.214 GiB in #43).

Report only: no estimate, coefficient or default changed.
