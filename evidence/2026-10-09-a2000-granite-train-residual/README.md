# What is live at granite-3.1-3b-a800m's training peak (#43)

**Question.** granite-3.1-3b-a800m (`grouped_nf4`, resident) peaked 56 MiB above the plan's allocator estimate on the
RTX A2000 on 2026-10-04 (3,164 against 3,108 MiB). Its plan, replanned with its siblings' receipts on file, sat under
the driver peak at 1.014. What is live at that peak?

**Method.** `bench/train_residual.py` as in `../2026-10-09-a2000-moe-train-residual/`: one re-run of the committed
receipt `2026-10-04-rtx-a2000/granite-3.1-3b-a800m-instruct-device-grouped_nf4-t1024-20261004T180058Z` (micro-batch
2 × 512, 12 steps). The allocator history is recorded and replayed from the first training step. `E4B_ABSMAX_DQ=0`,
since the receipt predates the double-quantized absmax default. The run used loggetta 0.4.0 (`ac2d898`),
experts4bit-qlora 0.51.0, grouped-nf4-gemm 0.44.0, torch 2.11.0+cu128, transformers 5.19.0 (`freeze.txt`) and one RTX
A2000 12GB, about 2 minutes of GPU. `replay.log` is the whole output, `runs/G/residual.json` the groups at the peak,
`runs/G/receipts/` the run's receipt. The trained adapter and the ~145 MB snapshot are not committed.

| live at the training peak, MiB | G |
|---|---|
| frozen expert stacks (resident) | 1620.0 |
| dense weights (bf16) | 531.9 |
| fp32 logits-sized tensors: 2 unattributed (`?`) + 1 `fixed_cross_entropy` | 3 × 192.0 = 576.0 |
| saved layer inputs (`modeling_granitemoe.py:361`, 32 × 3 MiB) | 96.0 |
| optimizer state (adamw) | 190.0 |
| expert LoRA adapters | 90.0 |
| attention LoRA adapters | 5.0 |
| other (LoRA forward 8.1, two layer buffers 3.0 each, small) | 23.4 |
| **replayed peak** | **3131.4** |
| measured allocated peak | 3131.5 |
| plan's allocator estimate | 3108.0 |
| residual (measured − estimate) | **+23.5** |

**Reading.**

- **The peak is the loss, not an MoE layer's backward.** Unlike OLMoE's `grouped_nf4` runs (#44), granite's fused-kernel
  workspaces are not live there (0.1 MiB).
- **Three fp32 logits-sized tensors are live together: 3 × T × V × 4 B = 576 MiB.** The estimate's activation item
  prices logits and loss at 10 B per logit (bf16 + fp32 + fp32 = 480 MiB at T = 1024, V = 49,155). That is 96 MiB short.
  It is the same miss loggetta's dense estimate corrected in #27 (12 B per logit). The MoE estimate is
  experts4bit-qlora's (`estimate_qlora_footprint`), and it still prices 10.
- **The adapter gradients are not live at a loss peak.** The estimate prices them (95 MiB), so they offset most of
  the logits miss. Net: +96 − 95 + the small items = +23.5 MiB.
- **The 2026-10-04 receipt showed +56 MiB, today's re-run +23.5 MiB.** The stacks differ: experts4bit-qlora 0.44.0,
  torch 2.8.0 and transformers 5.18.0 then; 0.51.0, 2.11.0 and 5.19.0 now. The 32 MiB difference is not attributed
  here.

**Not shown.** Another card, shape or vocabulary. With a larger vocabulary the logits miss grows as 2 × T × V bytes, and
with the loss as the peak nothing offsets it but the adapter gradients.
