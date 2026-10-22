| card | model | seq × mb | kernel | adapter | arms | driver peak (sampled) | plan | driver / plan | allocated peak | estimate | allocated / estimate | plan status | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RTX 5090 | Qwen3-30B-A3B | 2048 × 2 | grouped_nf4 | bf16 | 58 | 25.05 | 25.99 | 0.964 | 23.76 | 21.24 | 1.119 **short** | feasible | not shown under |
| RTX 5090 | Qwen3-30B-A3B | 2048 × 1 | grouped_nf4 | fp32 | 2 | 25.16 | 27.94 | 0.901 | 24.27 | 22.87 | 1.061 **short** | feasible | not shown under |
| RTX 5090 | Qwen3-30B-A3B | 2048 × 2 | grouped_nf4 | fp32 | 91 | 28.44 | 29.50 | 0.964 | 27.15 | 24.16 | 1.124 **short** | feasible | not shown under |
| RTX 5090 | Qwen3-30B-A3B | 2048 × 2 | reference | fp32 | 3 | 26.46 | 38.51 | 0.687 | 25.28 | 31.68 | 0.798 | refused | not shown under |
| RTX 5090 | Qwen3-30B-A3B | 4096 × 1 | grouped_nf4 | bf16 | 10 | 27.07 | 25.99 | 1.042 | 25.12 | 21.24 | 1.183 **short** | feasible | **UNDER** (10 of 10 arms) |
| RTX 5090 | Qwen3-30B-A3B | 4096 × 1 | grouped_nf4 | fp32 | 27 | 31.33 | 29.50 | 1.062 | 30.33 | 24.16 | 1.255 **short** | feasible | **UNDER** (21 of 27 arms) |
| RTX 5090 | OLMoE-1B-7B-0924-Instruct | 2048 × 2 | grouped_nf4 | bf16 | 1 | 6.83 | 9.18 | 0.744 | 5.84 | 7.24 | 0.808 | feasible | not shown under |
| RTX 5090 | OLMoE-1B-7B-0924-Instruct | 2048 × 2 | grouped_nf4 | fp32 | 4 | 7.93 | 9.73 | 0.815 | 7.15 | 7.69 | 0.929 | feasible | not shown under |
| RTX 5090 | OLMoE-1B-7B-0924-Instruct | 2048 × 2 | reference | fp32 | 1 | 7.01 | 10.06 | 0.697 | 6.04 | 7.97 | 0.758 | feasible | not shown under |
| RTX 5090 | granite-3.1-3b-a800m-instruct | 2048 × 2 | grouped_nf4 | bf16 | 1 | 4.47 | 6.81 | 0.656 | 3.52 | 5.26 | 0.669 | feasible | not shown under |
| RTX 5090 | granite-3.1-3b-a800m-instruct | 2048 × 2 | grouped_nf4 | fp32 | 2 | 4.94 | 7.25 | 0.681 | 4.02 | 5.63 | 0.714 | feasible | not shown under |
| RTX 5090 | granite-3.1-3b-a800m-instruct | 2048 × 2 | reference | fp32 | 1 | 4.88 | 7.41 | 0.659 | 3.87 | 5.76 | 0.673 | feasible | not shown under |
| RTX 5090 | Mixtral-8x7B-Instruct-v0.1 | 2048 × 2 | grouped_nf4 | fp32 | 12 | 31.00 | 42.35 | 0.732 | 29.11 | 34.87 | 0.835 | refused | not shown under |
| RTX 5090 | Mixtral-8x7B-Instruct-v0.1 | 2048 × 2 | reference | fp32 | 2 | 31.00 | 36.13 | 0.858 | 28.23 | 29.69 | 0.951 | refused | not shown under |
| H100 NVL | Qwen3-30B-A3B | 2048 × 2 | grouped_nf4 | fp32 | 3 | 26.75 | 29.50 | 0.907 | 25.39 | 24.16 | 1.051 **short** | feasible | not shown under |
| H100 NVL | Qwen3-30B-A3B | 2048 × 2 | reference | fp32 | 1 | 26.55 | 38.51 | 0.689 | 25.28 | 31.68 | 0.798 | feasible | not shown under |

GiB. Driver peak, allocated peak: the maximum over the setup's primary arms.

- 5090 setups: 14
- UNDER: 2
- estimate short (exact): 5
- false refusals: 3
- A2000-on-file variant differs: 16
