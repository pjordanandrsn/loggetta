| card | model | seq × mb | kernel | adapter | arms | driver peak (sampled) | plan | driver / plan | allocated peak | estimate | allocated / estimate | plan status | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RTX 5090 | Qwen3-30B-A3B | 2048 × 2 | grouped_nf4 | bf16 | 4 | 24.32 | 27.47 | 0.885 | 22.98 | 22.48 | 1.022 **short** | feasible | not shown under |
| RTX 5090 | Qwen3-30B-A3B | 2048 × 2 | grouped_nf4 | fp32 | 4 | 26.72 | 30.98 | 0.863 | 25.32 | 25.40 | 0.997 | refused | not shown under |
| RTX 5090 | Qwen3-30B-A3B | 2048 × 2 | reference | fp32 | 3 | 26.46 | 38.51 | 0.687 | 25.28 | 31.68 | 0.798 | refused | not shown under |
| RTX 5090 | Qwen3-30B-A3B | 4096 × 1 | grouped_nf4 | bf16 | 4 | 27.07 | 27.47 | 0.985 | 25.12 | 22.48 | 1.118 **short** | feasible | not shown under |
| RTX 5090 | Qwen3-30B-A3B | 4096 × 1 | grouped_nf4 | fp32 | 18 | 29.56 | 30.98 | 0.954 | 26.29 | 25.40 | 1.035 **short** | refused | not shown under |
| RTX 5090 | OLMoE-1B-7B-0924-Instruct | 2048 × 2 | reference | fp32 | 1 | 7.01 | 10.06 | 0.697 | 6.04 | 7.97 | 0.758 | feasible | not shown under |
| RTX 5090 | granite-3.1-3b-a800m-instruct | 2048 × 2 | reference | fp32 | 1 | 4.88 | 7.41 | 0.659 | 3.87 | 5.76 | 0.673 | feasible | not shown under |
| RTX 5090 | Mixtral-8x7B-Instruct-v0.1 | 2048 × 2 | grouped_nf4 | fp32 | 12 | 31.00 | 44.70 | 0.694 | 29.11 | 36.83 | 0.790 | refused | not shown under |
| RTX 5090 | Mixtral-8x7B-Instruct-v0.1 | 2048 × 2 | reference | fp32 | 2 | 31.00 | 36.13 | 0.858 | 28.23 | 29.69 | 0.951 | refused | not shown under |
| H100 NVL | Qwen3-30B-A3B | 2048 × 2 | reference | fp32 | 1 | 26.55 | 38.51 | 0.689 | 25.28 | 31.68 | 0.798 | feasible | not shown under |

GiB. Driver peak, allocated peak: the maximum over the setup's primary arms.

- 5090 setups: 9
- UNDER: 0
- estimate short (exact): 3
- false refusals: 5
- A2000-on-file variant differs: 10
