| receipt | card | kernel, residency | allocated peak | est. before | before − alloc | new line | est. after | after − alloc |
|---|---|---|---|---|---|---|---|---|
| 2026-10-04-fp1-rtx5090/fp1-5090-1-olmoe_device | GeForce RTX 5090 | grouped_nf4, device | 5600.8 | 5384.5 | -216.2 | 308.8 | 5693.3 | +92.5 |
| 2026-10-04-fp1-rtx5090/fp1-5090-1-qwen3_device | GeForce RTX 5090 | grouped_nf4, device | 22433.1 | 22618.2 | +185.0 | 0.0 | 22618.2 | +185.0 |
| 2026-10-04-fp1-rtx5090/fp1-5090-1-qwen3_host | GeForce RTX 5090 | grouped_nf4, host | 6866.7 | 7390.2 | +523.5 | 0.0 | 7390.2 | +523.5 |
| 2026-10-04-rtx-a2000/OLMoE-1B-7B-0924-device-grouped_nf4-t1024-20261004T173932Z | RTX A2000 12GB | grouped_nf4, device | 5602.3 | 5384.5 | -217.8 | 308.8 | 5693.3 | +91.0 |
| 2026-10-04-rtx-a2000/OLMoE-1B-7B-0924-device-grouped_nf4-t1024-20261004T175201Z | RTX A2000 12GB | grouped_nf4, device | 5602.6 | 5384.5 | -218.1 | 308.8 | 5693.3 | +90.6 |
| 2026-10-04-rtx-a2000/OLMoE-1B-7B-0924-device-reference-t1024-20261004T175624Z | RTX A2000 12GB | reference, device | 5394.2 | 5384.5 | -9.7 | 0.0 | 5384.5 | -9.7 |
| 2026-10-04-rtx-a2000/OLMoE-1B-7B-0924-host-grouped_nf4-t1024-20261004T175357Z | RTX A2000 12GB | grouped_nf4, host | 2362.1 | 2144.5 | -217.6 | 308.8 | 2453.3 | +91.1 |
| 2026-10-04-rtx-a2000/OLMoE-1B-7B-0924-host-grouped_nf4-t1024-20261004T182135Z | RTX A2000 12GB | grouped_nf4, host | 2363.5 | 2144.5 | -219.0 | 308.8 | 2453.3 | +89.7 |
| 2026-10-04-rtx-a2000/granite-3.1-3b-a800m-instruct-device-grouped_nf4-t1024-20261004T180058Z | RTX A2000 12GB | grouped_nf4, device | 3163.9 | 3108.0 | -56.0 | 0.0 | 3108.0 | -56.0 |
| 2026-10-04-rtx-a2000/granite-4.0-h-tiny-device-grouped_nf4-t1024-20261004T183236Z | RTX A2000 12GB | grouped_nf4, device | 6846.6 | 6777.7 | -69.0 | 0.0 | 6777.7 | -69.0 |
| 2026-10-09-a2000-moe-train-residual/A2/OLMoE-1B-7B-0924-device-grouped_nf4-t1024-20261009T014815Z-a52e0454 | RTX A2000 12GB | grouped_nf4, device | 5583.8 | 5384.5 | -199.3 | 308.8 | 5693.3 | +109.5 |
| 2026-10-09-a2000-moe-train-residual/B/OLMoE-1B-7B-0924-device-reference-t1024-20261009T014359Z-28877bcf | RTX A2000 12GB | reference, device | 5394.2 | 5384.5 | -9.7 | 0.0 | 5384.5 | -9.7 |
| 2026-10-09-a2000-moe-train-residual/C/OLMoE-1B-7B-0924-host-grouped_nf4-t1024-20261009T014637Z-7ea1afbe | RTX A2000 12GB | grouped_nf4, host | 2339.1 | 2144.5 | -194.6 | 308.8 | 2453.3 | +114.1 |

MiB. A negative difference is an estimate under the allocated peak.

- 2026-10-04-fp1-rtx5090/fp1-5090-1-olmoe_device: boundaries 64.0 MiB + padded LoRA delta 640.0 MiB (65536 rows x (H + first_out + I = 2048 + 2048 + 1024) x 2 B; single block: min(E, T x top_k) x T = 64 x 1024) + fused workspaces 160.0 MiB (5 x T x top_k x first_out bf16) - the activation item 555.2 MiB; a bound in shape (#43, #44)

- 2026-10-04-rtx-a2000/OLMoE-1B-7B-0924-device-grouped_nf4-t1024-20261004T173932Z: boundaries 64.0 MiB + padded LoRA delta 640.0 MiB (65536 rows x (H + first_out + I = 2048 + 2048 + 1024) x 2 B; single block: min(E, T x top_k) x T = 64 x 1024) + fused workspaces 160.0 MiB (5 x T x top_k x first_out bf16) - the activation item 555.2 MiB; a bound in shape (#43, #44)

- 2026-10-04-rtx-a2000/OLMoE-1B-7B-0924-device-grouped_nf4-t1024-20261004T175201Z: boundaries 64.0 MiB + padded LoRA delta 640.0 MiB (65536 rows x (H + first_out + I = 2048 + 2048 + 1024) x 2 B; single block: min(E, T x top_k) x T = 64 x 1024) + fused workspaces 160.0 MiB (5 x T x top_k x first_out bf16) - the activation item 555.2 MiB; a bound in shape (#43, #44)

- 2026-10-04-rtx-a2000/OLMoE-1B-7B-0924-host-grouped_nf4-t1024-20261004T175357Z: boundaries 64.0 MiB + padded LoRA delta 640.0 MiB (65536 rows x (H + first_out + I = 2048 + 2048 + 1024) x 2 B; single block: min(E, T x top_k) x T = 64 x 1024) + fused workspaces 160.0 MiB (5 x T x top_k x first_out bf16) - the activation item 555.2 MiB; a bound in shape (#43, #44)

- 2026-10-04-rtx-a2000/OLMoE-1B-7B-0924-host-grouped_nf4-t1024-20261004T182135Z: boundaries 64.0 MiB + padded LoRA delta 640.0 MiB (65536 rows x (H + first_out + I = 2048 + 2048 + 1024) x 2 B; single block: min(E, T x top_k) x T = 64 x 1024) + fused workspaces 160.0 MiB (5 x T x top_k x first_out bf16) - the activation item 555.2 MiB; a bound in shape (#43, #44)

- 2026-10-09-a2000-moe-train-residual/A2/OLMoE-1B-7B-0924-device-grouped_nf4-t1024-20261009T014815Z-a52e0454: boundaries 64.0 MiB + padded LoRA delta 640.0 MiB (65536 rows x (H + first_out + I = 2048 + 2048 + 1024) x 2 B; single block: min(E, T x top_k) x T = 64 x 1024) + fused workspaces 160.0 MiB (5 x T x top_k x first_out bf16) - the activation item 555.2 MiB; a bound in shape (#43, #44)

- 2026-10-09-a2000-moe-train-residual/C/OLMoE-1B-7B-0924-host-grouped_nf4-t1024-20261009T014637Z-7ea1afbe: boundaries 64.0 MiB + padded LoRA delta 640.0 MiB (65536 rows x (H + first_out + I = 2048 + 2048 + 1024) x 2 B; single block: min(E, T x top_k) x T = 64 x 1024) + fused workspaces 160.0 MiB (5 x T x top_k x first_out bf16) - the activation item 555.2 MiB; a bound in shape (#43, #44)
