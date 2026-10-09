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
