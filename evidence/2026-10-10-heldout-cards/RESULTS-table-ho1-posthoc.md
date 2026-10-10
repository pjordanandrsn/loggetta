| card | model | seq × mb | kernel | adapter | arms | driver peak (sampled) | plan | driver / plan | allocated peak | estimate | allocated / estimate | plan status | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RTX 5090 | Qwen3-30B-A3B | 2048 × 2 | grouped_nf4 | bf16 | 4 | 24.32 | 27.10 | 0.898 | 22.98 | 22.16 | 1.037 **short** | feasible | not shown under |
| RTX 5090 | Qwen3-30B-A3B | 2048 × 2 | grouped_nf4 | fp32 | 4 | 26.72 | 30.61 | 0.873 | 25.32 | 25.09 | 1.009 **short** | refused | not shown under |
| RTX 5090 | Qwen3-30B-A3B | 2048 × 2 | reference | fp32 | 3 | 26.46 | 37.12 | 0.713 | 25.28 | 30.51 | 0.829 | refused | not shown under |
| RTX 5090 | Qwen3-30B-A3B | 4096 × 1 | grouped_nf4 | bf16 | 4 | 27.07 | 27.10 | 0.999 | 25.12 | 22.16 | 1.133 **short** | feasible | not shown under |
| RTX 5090 | Qwen3-30B-A3B | 4096 × 1 | grouped_nf4 | fp32 | 18 | 29.56 | 30.61 | 0.966 | 26.29 | 25.09 | 1.048 **short** | refused | not shown under |
| RTX 5090 | OLMoE-1B-7B-0924-Instruct | 2048 × 2 | reference | fp32 | 1 | 7.01 | 9.60 | 0.731 | 6.04 | 7.58 | 0.797 | feasible | not shown under |
| RTX 5090 | granite-3.1-3b-a800m-instruct | 2048 × 2 | reference | fp32 | 1 | 4.88 | 6.95 | 0.702 | 3.87 | 5.38 | 0.720 | feasible | not shown under |
| RTX 5090 | Mixtral-8x7B-Instruct-v0.1 | 2048 × 2 | grouped_nf4 | fp32 | 12 | 31.00 | 43.44 | 0.714 | 29.11 | 35.78 | 0.813 | refused | not shown under |
| RTX 5090 | Mixtral-8x7B-Instruct-v0.1 | 2048 × 2 | reference | fp32 | 2 | 31.00 | 35.83 | 0.865 | 28.23 | 29.44 | 0.959 | refused | not shown under |
| H100 NVL | Qwen3-30B-A3B | 2048 × 2 | reference | fp32 | 1 | 26.55 | 37.12 | 0.715 | 25.28 | 30.51 | 0.829 | feasible | not shown under |

GiB. Driver peak, allocated peak: the maximum over the setup's primary arms.

- 5090 setups: 9
- UNDER: 0
- estimate short (exact): 4
- false refusals: 5
- A2000-on-file variant differs: 0
