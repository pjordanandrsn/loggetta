| card | model | seq × mb | kernel | adapter | arms | driver peak (sampled) | plan | driver / plan | allocated peak | estimate | allocated / estimate | plan status | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RTX 5090 | Qwen3-30B-A3B | 2048 × 2 | grouped_nf4 | bf16 | 4 | 24.32 | 25.99 | 0.936 | 22.98 | 21.24 | 1.082 **short** | feasible | not shown under |
| RTX 5090 | Qwen3-30B-A3B | 2048 × 2 | grouped_nf4 | fp32 | 4 | 26.72 | 29.50 | 0.906 | 25.32 | 24.16 | 1.048 **short** | feasible | not shown under |
| RTX 5090 | Qwen3-30B-A3B | 2048 × 2 | reference | fp32 | 3 | 26.46 | 38.51 | 0.687 | 25.28 | 31.68 | 0.798 | refused | not shown under |
| RTX 5090 | Qwen3-30B-A3B | 4096 × 1 | grouped_nf4 | bf16 | 4 | 27.07 | 25.99 | 1.042 | 25.12 | 21.24 | 1.183 **short** | feasible | **UNDER** (4 of 4 arms) |
| RTX 5090 | Qwen3-30B-A3B | 4096 × 1 | grouped_nf4 | fp32 | 18 | 29.56 | 29.50 | 1.002 | 26.29 | 24.16 | 1.088 **short** | feasible | **UNDER** (12 of 18 arms) |
| RTX 5090 | OLMoE-1B-7B-0924-Instruct | 2048 × 2 | reference | fp32 | 1 | 7.01 | 10.06 | 0.697 | 6.04 | 7.97 | 0.758 | feasible | not shown under |
| RTX 5090 | granite-3.1-3b-a800m-instruct | 2048 × 2 | reference | fp32 | 1 | 4.88 | 7.41 | 0.659 | 3.87 | 5.76 | 0.673 | feasible | not shown under |
| RTX 5090 | Mixtral-8x7B-Instruct-v0.1 | 2048 × 2 | grouped_nf4 | fp32 | 12 | 31.00 | 42.35 | 0.732 | 29.11 | 34.87 | 0.835 | refused | not shown under |
| RTX 5090 | Mixtral-8x7B-Instruct-v0.1 | 2048 × 2 | reference | fp32 | 2 | 31.00 | 36.13 | 0.858 | 28.23 | 29.69 | 0.951 | refused | not shown under |
| H100 NVL | Qwen3-30B-A3B | 2048 × 2 | reference | fp32 | 1 | 26.55 | 38.51 | 0.689 | 25.28 | 31.68 | 0.798 | feasible | not shown under |

GiB. Driver peak, allocated peak: the maximum over the setup's primary arms.

- 5090 setups: 9
- UNDER: 2
- estimate short (exact): 4
- false refusals: 3
- A2000-on-file variant differs: 10
