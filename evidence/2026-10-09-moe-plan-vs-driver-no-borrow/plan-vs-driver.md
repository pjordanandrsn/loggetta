## Summary (receipts with a plan total and a driver peak)

| card | kind | placement | receipts with plan + driver | plan under driver | driver / plan, min–max |
|---|---|---|---|---|---|
| GeForce RTX 4090 | serve | offload (solver tiers) | 4 | 0 | 0.902–0.981 |
| GeForce RTX 4090 | serve | resident (all-VRAM) | 1 | 0 | 0.988–0.988 |
| RTX A2000 12GB | serve | offload (solver tiers) | 2 | 0 | 0.879–0.939 |
| RTX A2000 12GB | serve | resident (all-VRAM) | 15 | 0 | 0.749–0.870 |
| RTX A2000 12GB | train | offload (experts on host) | 3 | 3 | 1.097–1.237 |
| RTX A2000 12GB | train | resident (experts on device) | 8 | 3 | 0.829–1.184 |

## Where the plans under their driver peak went short (GiB)

| run | driver − plan | allocated − estimate | (reserved − allocated) − reserve line | (driver − reserved) − context line | the plan's reserve line | the plan's context line |
|---|---|---|---|---|---|---|
| 2026-10-04-rtx-a2000/OLMoE-1B-7B-0924-device-grouped_nf4-t1024-20261004T173932Z | +1.058 | +0.213 | +1.213 | -0.367 | None | inferred: default; no receipt on file measured this GPU + driver |
| 2026-10-04-rtx-a2000/OLMoE-1B-7B-0924-device-grouped_nf4-t1024-20261004T175201Z | +0.260 | +0.213 | +0.047 | +0.000 | measured: receipt OLMoE-1B-7B-0924-device-grouped_nf4-t1024-20261004T173932Z: reserved peak / allocated peak - 1 = 0.222 | measured: receipt OLMoE-1B-7B-0924-device-grouped_nf4-t1024-20261004T173932Z: driver-reported process peak minus allocator reserved peak |
| 2026-10-04-rtx-a2000/OLMoE-1B-7B-0924-host-grouped_nf4-t1024-20261004T175357Z | +0.639 | +0.213 | +0.426 | +0.000 | measured: receipt OLMoE-1B-7B-0924-device-grouped_nf4-t1024-20261004T175201Z: reserved peak / allocated peak - 1 = 0.222 | measured: receipt OLMoE-1B-7B-0924-device-grouped_nf4-t1024-20261004T175201Z: driver-reported process peak minus allocator reserved peak |
| 2026-10-04-rtx-a2000/OLMoE-1B-7B-0924-host-grouped_nf4-t1024-20261004T182135Z | +0.295 | +0.214 | +0.081 | +0.000 | measured: receipt OLMoE-1B-7B-0924-host-grouped_nf4-t1024-20261004T175357Z (same residency and kernel): reserved / allocated peak - 1 = 0.386 | measured: receipt granite-3.1-3b-a800m-instruct-device-grouped_nf4-t1024-20261004T180058Z: driver-reported process peak minus allocator reserved peak |
| 2026-10-09-a2000-moe-train-residual/A2/OLMoE-1B-7B-0924-device-grouped_nf4-t1024-20261009T014815Z-a52e0454 | +0.006 | +0.195 | +0.179 | -0.367 | inferred: default 20% of the allocator estimate; no receipt on file measured this GPU + driver | inferred: default; no receipt on file measured this GPU + driver |
| 2026-10-09-a2000-moe-train-residual/C/OLMoE-1B-7B-0924-host-grouped_nf4-t1024-20261009T014637Z-7ea1afbe | +0.315 | +0.190 | +0.492 | -0.367 | inferred: default 20% of the allocator estimate; no receipt on file measured this GPU + driver | inferred: default; no receipt on file measured this GPU + driver |

## Every receipt

| source | run | card | kind | placement | status | plan total | driver peak | driver − plan | driver / plan | allocator est. | allocated peak | allocated / est. | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| loggetta plan | 2026-10-04-rtx-a2000/OLMoE-1B-7B-0924-device-grouped_nf4-t1024-20261004T173932Z | RTX A2000 12GB | train | resident (experts on device) | OK | 5.758 | 6.816 | +1.058 | 1.184 | 5.258 | 5.471 | 1.040 | **PLAN UNDER DRIVER**; **EST. UNDER ALLOCATED** |
| loggetta plan | 2026-10-04-rtx-a2000/OLMoE-1B-7B-0924-device-grouped_nf4-t1024-20261004T175201Z | RTX A2000 12GB | train | resident (experts on device) | OK | 6.557 | 6.816 | +0.260 | 1.040 | 5.258 | 5.471 | 1.041 | **PLAN UNDER DRIVER**; **EST. UNDER ALLOCATED** |
| loggetta plan | 2026-10-04-rtx-a2000/OLMoE-1B-7B-0924-device-reference-t1024-20261004T175624Z | RTX A2000 12GB | train | resident (experts on device) | OK | 7.421 | 6.277 | -1.144 | 0.846 | 5.258 | 5.268 | 1.002 | **EST. UNDER ALLOCATED** |
| loggetta plan | 2026-10-04-rtx-a2000/OLMoE-1B-7B-0924-host-grouped_nf4-t1024-20261004T175357Z | RTX A2000 12GB | train | offload (experts on host) | OK | 2.691 | 3.330 | +0.639 | 1.237 | 2.094 | 2.307 | 1.101 | **PLAN UNDER DRIVER**; **EST. UNDER ALLOCATED** |
| loggetta plan | 2026-10-04-rtx-a2000/OLMoE-1B-7B-0924-host-grouped_nf4-t1024-20261004T182135Z | RTX A2000 12GB | train | offload (experts on host) | OK | 3.036 | 3.330 | +0.295 | 1.097 | 2.094 | 2.308 | 1.102 | **PLAN UNDER DRIVER**; **EST. UNDER ALLOCATED** |
| loggetta plan | 2026-10-04-rtx-a2000/granite-3.1-3b-a800m-instruct-device-grouped_nf4-t1024-20261004T180058Z | RTX A2000 12GB | train | resident (experts on device) | OK | 4.340 | 3.928 | -0.412 | 0.905 | 3.035 | 3.090 | 1.018 | **EST. UNDER ALLOCATED** |
| loggetta plan | 2026-10-04-rtx-a2000/granite-4.0-h-tiny-device-grouped_nf4-t1024-20261004T183236Z | RTX A2000 12GB | train | resident (experts on device) | OK | 8.262 | 7.406 | -0.856 | 0.896 | 6.619 | 6.686 | 1.010 | **EST. UNDER ALLOCATED** |
| loggetta plan | 2026-10-05-a2000-int4-serve/a2000-olmoe-4x4096-attnint4 | RTX A2000 12GB | serve | resident (all-VRAM) | OK | 7.867 | 5.920 | -1.947 | 0.753 | 6.139 | 5.751 | 0.937 |  |
| loggetta plan | 2026-10-05-a2000-int4-serve/a2000-olmoe-4x4096-both | RTX A2000 12GB | serve | resident (all-VRAM) | OK | 7.895 | 5.916 | -1.979 | 0.749 | 6.162 | 5.758 | 0.934 |  |
| loggetta plan | 2026-10-05-a2000-int4-serve/a2000-olmoe-4x4096-expint4 | RTX A2000 12GB | serve | resident (all-VRAM) | OK | 7.679 | 5.783 | -1.896 | 0.753 | 5.983 | 5.579 | 0.932 |  |
| loggetta plan | 2026-10-05-a2000-int4-serve/a2000-olmoe-4x4096-nf4 | RTX A2000 12GB | serve | resident (all-VRAM) | OK | 7.651 | 5.775 | -1.875 | 0.755 | 5.959 | 5.571 | 0.935 |  |
| loggetta plan | 2026-10-05-a2000-int4-serve/before-heap-fix/int4-olmoe-4x4096-attnint4 | RTX A2000 12GB | serve | resident (all-VRAM) | OK | 7.867 | 5.904 | -1.962 | 0.751 | 6.139 | 5.751 | 0.937 |  |
| loggetta plan | 2026-10-05-a2000-int4-serve/before-heap-fix/int4-olmoe-4x4096-both | RTX A2000 12GB | serve | resident (all-VRAM) | OK | 7.895 | 5.916 | -1.979 | 0.749 | 6.162 | 5.758 | 0.934 |  |
| loggetta plan | 2026-10-05-a2000-int4-serve/before-heap-fix/int4-olmoe-4x4096-expint4 | RTX A2000 12GB | serve | resident (all-VRAM) | OK | 7.679 | 5.783 | -1.896 | 0.753 | 5.983 | 5.579 | 0.932 |  |
| loggetta plan | 2026-10-05-a2000-int4-serve/before-heap-fix/int4-olmoe-4x4096-nf4 | RTX A2000 12GB | serve | resident (all-VRAM) | OK | 7.651 | 5.775 | -1.875 | 0.755 | 5.959 | 5.571 | 0.935 |  |
| loggetta plan | 2026-10-05-a2000-int4-serve/before-heap-fix/int4fix-olmoe-4x4096-attnint4 | RTX A2000 12GB | serve | resident (all-VRAM) | OK | 7.867 | 5.904 | -1.962 | 0.751 | 6.139 | 5.751 | 0.937 |  |
| loggetta plan | 2026-10-05-a2000-int4-serve/before-heap-fix/int4fix-olmoe-4x4096-both | RTX A2000 12GB | serve | resident (all-VRAM) | OK | 7.895 | 5.916 | -1.979 | 0.749 | 6.162 | 5.758 | 0.934 |  |
| loggetta plan | 2026-10-05-a2000-int4-serve/before-heap-fix/int4fix-olmoe-4x4096-expint4 | RTX A2000 12GB | serve | resident (all-VRAM) | OK | 7.679 | 5.783 | -1.896 | 0.753 | 5.983 | 5.579 | 0.932 |  |
| loggetta plan | 2026-10-05-a2000-int4-serve/before-heap-fix/int4fix2-olmoe-4x4096-attnint4 | RTX A2000 12GB | serve | resident (all-VRAM) | OK | 7.867 | 5.920 | -1.947 | 0.753 | 6.139 | 5.751 | 0.937 |  |
| loggetta plan | 2026-10-05-rtx-a2000-serve/longprompt-olmoe-2x4096-p4000 | RTX A2000 12GB | serve | resident (all-VRAM) | OK | 6.670 | 5.652 | -1.017 | 0.847 | 5.420 | 5.414 | 0.999 |  |
| loggetta plan | 2026-10-05-rtx-a2000-serve/planned-tiers-olmoe-v3.0-r4.5 | RTX A2000 12GB | serve | offload (solver tiers) | OK | 2.498 | 2.346 | -0.152 | 0.939 | 2.052 | 2.051 | 1.000 |  |
| loggetta plan | 2026-10-05-rtx-a2000-serve/serve-OLMoE-1B-7B-0924-c4096x4-20261005T000049 | RTX A2000 12GB | serve | resident (all-VRAM) | OK | 6.642 | 5.775 | -0.867 | 0.870 | 5.397 | 5.571 | 1.032 | **EST. UNDER ALLOCATED** |
| loggetta plan | 2026-10-06-a2000-family-serve/a2000-ernie-4.5-21b-4x4096 | RTX A2000 12GB | serve | offload (solver tiers) | OK | 10.115 | 8.893 | -1.223 | 0.879 | 8.013 | 8.345 | 1.041 | **EST. UNDER ALLOCATED** |
| loggetta plan | 2026-10-06-a2000-family-serve/a2000-granite-3.1-3b-4x4096 | RTX A2000 12GB | serve | resident (all-VRAM) | OK | 4.096 | 3.176 | -0.921 | 0.775 | 2.997 | 2.959 | 0.987 |  |
| loggetta plan | 2026-10-09-a2000-granite-train-residual/runs/G/receipts/granite-3.1-3b-a800m-instruct-device-grouped_nf4-t1024-20261009T145928Z-ce9e5abb | RTX A2000 12GB | train | resident (experts on device) | OK | 4.142 | 3.434 | -0.709 | 0.829 | 3.035 | 3.058 | 1.008 | **EST. UNDER ALLOCATED** |
| loggetta plan | 2026-10-09-a2000-moe-train-residual/A2/OLMoE-1B-7B-0924-device-grouped_nf4-t1024-20261009T014815Z-a52e0454 | RTX A2000 12GB | train | resident (experts on device) | OK | 6.810 | 6.816 | +0.006 | 1.001 | 5.258 | 5.453 | 1.037 | **PLAN UNDER DRIVER**; **EST. UNDER ALLOCATED** |
| loggetta plan | 2026-10-09-a2000-moe-train-residual/B/OLMoE-1B-7B-0924-device-reference-t1024-20261009T014359Z-28877bcf | RTX A2000 12GB | train | resident (experts on device) | OK | 6.810 | 6.275 | -0.535 | 0.922 | 5.258 | 5.268 | 1.002 | **EST. UNDER ALLOCATED** |
| loggetta plan | 2026-10-09-a2000-moe-train-residual/C/OLMoE-1B-7B-0924-host-grouped_nf4-t1024-20261009T014637Z-7ea1afbe | RTX A2000 12GB | train | offload (experts on host) | OK | 3.013 | 3.328 | +0.315 | 1.105 | 2.094 | 2.284 | 1.091 | **PLAN UNDER DRIVER**; **EST. UNDER ALLOCATED** |
| SV5 registered plan | sv5-4090-1/a5_long | GeForce RTX 4090 | serve | resident (all-VRAM) | OOM | 22.740 | — | — | — | 22.150 | 22.535 | 1.017 | **EST. UNDER ALLOCATED**; the lane's registered reading; no driver reading (0 samples) |
| SV5 registered plan | sv5-4090-1/a5_short | GeForce RTX 4090 | serve | resident (all-VRAM) | OK | 22.740 | 22.477 | -0.263 | 0.988 | 22.150 | 21.734 | 0.981 | same setup, shorter prompts |
| SV6 registered plan | sv6-4090-3/b6_long | GeForce RTX 4090 | serve | offload (solver tiers) | OK | 22.343 | 21.357 | -0.985 | 0.956 | 20.476 | 20.146 | 0.984 | the lane's registered reading |
| SV6 registered plan | sv6-4090-3/b6_short | GeForce RTX 4090 | serve | offload (solver tiers) | OK | 22.343 | 20.154 | -2.188 | 0.902 | 20.476 | 19.168 | 0.936 | same setup, shorter prompts |
| SV7 registered plan | sv7-4090-1/b7_long | GeForce RTX 4090 | serve | offload (solver tiers) | OK | 22.344 | — | — | — | 20.987 | 20.657 | 0.984 | the lane's registered reading; no driver reading (0 samples) |
| SV7 registered plan | sv7-4090-1/b7_short | GeForce RTX 4090 | serve | offload (solver tiers) | OK | 22.344 | — | — | — | 20.987 | 19.681 | 0.938 | same setup, shorter prompts; no driver reading (0 samples) |
| SV7 registered plan | sv7-4090-2/b7_long | GeForce RTX 4090 | serve | offload (solver tiers) | OK | 22.344 | 21.916 | -0.428 | 0.981 | 20.987 | 20.657 | 0.984 | the lane's registered reading |
| SV7 registered plan | sv7-4090-2/b7_short | GeForce RTX 4090 | serve | offload (solver tiers) | OK | 22.344 | 20.527 | -1.816 | 0.919 | 20.987 | 19.681 | 0.938 | same setup, shorter prompts |
| receipt's own estimate (no plan) | 2026-10-04-fp1-rtx5090/fp1-5090-1-olmoe_device | GeForce RTX 5090 | train | resident (experts on device) | OK | — | 6.904 | — | — | 5.258 | 5.469 | 1.040 | **EST. UNDER ALLOCATED** |
| receipt's own estimate (no plan) | 2026-10-04-fp1-rtx5090/fp1-5090-1-qwen3_device | GeForce RTX 5090 | train | resident (experts on device) | OK | — | 24.340 | — | — | 22.088 | 21.907 | 0.992 |  |
| receipt's own estimate (no plan) | 2026-10-04-fp1-rtx5090/fp1-5090-1-qwen3_host | GeForce RTX 5090 | train | offload (experts on host) | OK | — | 9.414 | — | — | 7.217 | 6.706 | 0.929 |  |
| receipt's own estimate (no plan) | 2026-10-04-rtx-a2000/direct-OLMoE-1B-7B-0924-device-grouped_nf4-t1024-20261004T173932Z | ? | train | resident (experts on device) | OK | — | 6.816 | — | — | — | 5.471 | — |  |
| receipt's own estimate (no plan) | 2026-10-05-rtx-a2000-serve/solver-olmoe-v1.2-d1.5-h64 | RTX A2000 12GB | serve | offload (solver tiers) | OK | — | 2.830 | — | — | — | 2.355 | — |  |
| receipt's own estimate (no plan) | 2026-10-05-rtx-a2000-serve/solver-olmoe-v2.0-d0.8-h128 | RTX A2000 12GB | serve | offload (solver tiers) | OK | — | 3.701 | — | — | — | 3.144 | — |  |
| receipt's own estimate (no plan) | 2026-10-05-sv1-rtx5090/sv1-5090-1-olmoe_eager | GeForce RTX 5090 | serve | resident (all-VRAM) | OK | — | 9.506 | — | — | 9.196 | 8.812 | 0.958 |  |
| receipt's own estimate (no plan) | 2026-10-05-sv1-rtx5090/sv1-5090-1-olmoe_graphs | GeForce RTX 5090 | serve | resident (all-VRAM) | OK | — | 9.811 | — | — | 9.213 | 8.870 | 0.963 |  |
| receipt's own estimate (no plan) | 2026-10-05-sv1-rtx5090/sv1-5090-1-olmoe_prefill | GeForce RTX 5090 | serve | resident (all-VRAM) | OK | — | 10.207 | — | — | 9.213 | 9.104 | 0.988 |  |
| receipt's own estimate (no plan) | 2026-10-05-sv1-rtx5090/sv1-5090-1-qwen3_graphs | GeForce RTX 5090 | serve | resident (all-VRAM) | OK | — | 22.619 | — | — | 21.786 | 21.612 | 0.992 |  |
| receipt's own estimate (no plan) | 2026-10-05-sv1-rtx5090/sv1-5090-1-qwen3_prefill | GeForce RTX 5090 | serve | resident (all-VRAM) | OK | — | 23.375 | — | — | 21.786 | 22.169 | 1.018 | **EST. UNDER ALLOCATED** |
| receipt's own estimate (no plan) | 2026-10-05-sv2-rtx5090/sv2-5090-1-q_both | GeForce RTX 5090 | serve | resident (all-VRAM) | OK | — | 23.252 | — | — | 22.410 | 22.236 | 0.992 |  |
| receipt's own estimate (no plan) | 2026-10-05-sv2-rtx5090/sv2-5090-1-q_exp | GeForce RTX 5090 | serve | resident (all-VRAM) | OK | — | 22.656 | — | — | 21.839 | 21.664 | 0.992 |  |
| receipt's own estimate (no plan) | 2026-10-05-sv2-rtx5090/sv2-5090-1-q_exp_prefill | GeForce RTX 5090 | serve | resident (all-VRAM) | OK | — | 23.398 | — | — | 21.839 | 22.222 | 1.018 | **EST. UNDER ALLOCATED** |
| receipt's own estimate (no plan) | 2026-10-05-sv2-rtx5090/sv2-5090-1-q_nf4 | GeForce RTX 5090 | serve | resident (all-VRAM) | OK | — | 22.619 | — | — | 21.786 | 21.612 | 0.992 |  |
| receipt's own estimate (no plan) | 2026-10-06-sv3-rtx5090/sv3-5090-1-gptoss_g16 | GeForce RTX 5090 | serve | resident (all-VRAM) | OK | — | 17.109 | — | — | 15.391 | 15.673 | 1.018 | **EST. UNDER ALLOCATED** |
| receipt's own estimate (no plan) | 2026-10-06-sv3-rtx5090/sv3-5090-1-q36_e16 | GeForce RTX 5090 | serve | resident (all-VRAM) | OK | — | 24.082 | — | — | 23.217 | 23.408 | 1.008 | **EST. UNDER ALLOCATED** |
| receipt's own estimate (no plan) | 2026-10-06-sv3-rtx5090/sv3-5090-1-q36_g16 | GeForce RTX 5090 | serve | resident (all-VRAM) | OK | — | 25.568 | — | — | 24.187 | 24.423 | 1.010 | **EST. UNDER ALLOCATED** |
| receipt's own estimate (no plan) | 2026-10-06-sv3-rtx5090/sv3-5090-1-q36_g1_capped | GeForce RTX 5090 | serve | resident (all-VRAM) | OK | — | 22.732 | — | — | 21.720 | 21.948 | 1.010 | **EST. UNDER ALLOCATED** |
| receipt's own estimate (no plan) | 2026-10-06-sv3-rtx5090/sv3-5090-1-q36_g1_default | GeForce RTX 5090 | serve | resident (all-VRAM) | ALARM | — | 24.219 | — | — | 22.629 | 22.893 | 1.012 | **EST. UNDER ALLOCATED** |
| receipt's own estimate (no plan) | 2026-10-06-sv4-rtx4090/sv4-4090-1-t4_all1 | GeForce RTX 4090 | serve | resident (all-VRAM) | OK | — | 19.242 | — | — | 18.685 | 18.668 | 0.999 |  |
| receipt's own estimate (no plan) | 2026-10-06-sv4-rtx4090/sv4-4090-1-t4_deep4 | GeForce RTX 4090 | serve | offload (solver tiers) | OK | — | 13.396 | — | — | 12.528 | 12.131 | 0.968 |  |
| receipt's own estimate (no plan) | 2026-10-06-sv4-rtx4090/sv4-4090-1-t4_plan8 | GeForce RTX 4090 | serve | offload (solver tiers) | OK | — | 18.639 | — | — | 18.264 | 17.471 | 0.957 |  |

## Today's planner on the same training setups

| run | allocated peak | today's allocator est. | driver peak | today, no receipts: plan total (driver / plan) | today, siblings on file: plan total (driver / plan; reserve; context) |
|---|---|---|---|---|---|
| 2026-10-04-rtx-a2000/OLMoE-1B-7B-0924-device-grouped_nf4-t1024-20261004T173932Z | 5.471 | 5.560 | 6.816 | 7.172 (0.950) | 6.958 (0.980); reserve measured 1.232; context measured 0.166 |
| 2026-10-04-rtx-a2000/OLMoE-1B-7B-0924-device-grouped_nf4-t1024-20261004T175201Z | 5.471 | 5.560 | 6.816 | 7.172 (0.950) | 6.958 (0.980); reserve measured 1.232; context measured 0.166 |
| 2026-10-04-rtx-a2000/OLMoE-1B-7B-0924-device-reference-t1024-20261004T175624Z | 5.268 | 5.258 | 6.277 | 6.810 (0.922) | 7.454 (0.842); reserve policy 2.030; context measured 0.166 |
| 2026-10-04-rtx-a2000/OLMoE-1B-7B-0924-host-grouped_nf4-t1024-20261004T175357Z | 2.307 | 2.396 | 3.330 | 3.375 (0.987) | 3.485 (0.956); reserve measured 0.923; context measured 0.166 |
| 2026-10-04-rtx-a2000/OLMoE-1B-7B-0924-host-grouped_nf4-t1024-20261004T182135Z | 2.308 | 2.396 | 3.330 | 3.375 (0.987) | 3.487 (0.955); reserve measured 0.925; context measured 0.166 |
| 2026-10-04-rtx-a2000/granite-3.1-3b-a800m-instruct-device-grouped_nf4-t1024-20261004T180058Z | 3.090 | 3.035 | 3.928 | 4.142 (0.948) | 4.373 (0.898); reserve policy 1.172; context measured 0.166 |
| 2026-10-04-rtx-a2000/granite-4.0-h-tiny-device-grouped_nf4-t1024-20261004T183236Z | 6.686 | 6.619 | 7.406 | 8.443 (0.877) | 9.307 (0.796); reserve policy 2.555; context measured 0.133 |
| 2026-10-09-a2000-granite-train-residual/runs/G/receipts/granite-3.1-3b-a800m-instruct-device-grouped_nf4-t1024-20261009T145928Z-ce9e5abb | 3.058 | 3.035 | 3.434 | 4.142 (0.829) | 4.142 (0.829); reserve inferred 0.607; context inferred 0.500 |
| 2026-10-09-a2000-moe-train-residual/A2/OLMoE-1B-7B-0924-device-grouped_nf4-t1024-20261009T014815Z-a52e0454 | 5.453 | 5.560 | 6.816 | 7.172 (0.950) | 7.172 (0.950); reserve inferred 1.112; context inferred 0.500 |
| 2026-10-09-a2000-moe-train-residual/B/OLMoE-1B-7B-0924-device-reference-t1024-20261009T014359Z-28877bcf | 5.268 | 5.258 | 6.275 | 6.810 (0.922) | 6.810 (0.922); reserve inferred 1.052; context inferred 0.500 |
| 2026-10-09-a2000-moe-train-residual/C/OLMoE-1B-7B-0924-host-grouped_nf4-t1024-20261009T014637Z-7ea1afbe | 2.284 | 2.396 | 3.328 | 3.375 (0.986) | 3.375 (0.986); reserve inferred 0.479; context inferred 0.500 |
