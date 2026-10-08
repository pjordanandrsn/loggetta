Dense plans at seq 4096, micro-batch 1 (bench/dense_plan_sweep.py; device totals include the inferred 20% reserve and 0.5 GiB context; no receipts).

| model | 16 GB | 24 GB (4090) | 32 GB (5090) | 48 GB |
|---|---|---|---|---|
| Qwen_Qwen2.5-72B-Instruct | refused (nf4 base, streamed): device 26.63 + headroom 0.77 GiB > budget 15.50 GiB | refused (nf4 base, streamed): device 26.63 + headroom 1.18 GiB > budget 23.52 GiB | nf4 streamed 26.6 GiB | nf4 streamed 26.6 GiB |
| Qwen_Qwen3-1.7B | bf16 resident 6.2 GiB | bf16 resident 6.2 GiB | bf16 resident 6.2 GiB | bf16 resident 6.2 GiB |
| Qwen_Qwen3-14B | nf4 streamed 10.0 GiB | nf4 resident 17.0 GiB | nf4 resident 17.0 GiB | bf16 resident 38.7 GiB |
| Qwen_Qwen3-32B | nf4 streamed 13.8 GiB | nf4 streamed 13.8 GiB | nf4 streamed 13.8 GiB | nf4 resident 30.7 GiB |
| Qwen_Qwen3-4B | bf16 resident 12.1 GiB | bf16 resident 12.1 GiB | bf16 resident 12.1 GiB | bf16 resident 12.1 GiB |
| Qwen_Qwen3-8B | nf4 resident 11.1 GiB | nf4 resident 11.1 GiB | bf16 resident 22.5 GiB | bf16 resident 22.5 GiB |
| microsoft_phi-4 | nf4 streamed 11.6 GiB | nf4 resident 18.9 GiB | nf4 resident 18.9 GiB | bf16 resident 41.1 GiB |
| mistralai_Mistral-7B-v0.3 | nf4 resident 8.9 GiB | bf16 resident 20.4 GiB | bf16 resident 20.4 GiB | bf16 resident 20.4 GiB |
| mistralai_Mistral-Small-24B-Base-2501 | refused (nf4 base, streamed): device 14.75 + headroom 0.77 GiB > budget 15.50 GiB | nf4 streamed 14.8 GiB | nf4 resident 26.6 GiB | nf4 resident 26.6 GiB |
| unsloth_Llama-3.1-8B | nf4 streamed 11.4 GiB | nf4 resident 15.0 GiB | bf16 resident 26.5 GiB | bf16 resident 26.5 GiB |
| unsloth_Llama-3.3-70B-Instruct | refused (nf4 base, streamed): device 24.53 + headroom 0.77 GiB > budget 15.50 GiB | refused (nf4 base, streamed): device 24.53 + headroom 1.18 GiB > budget 23.52 GiB | nf4 streamed 24.5 GiB | nf4 streamed 24.5 GiB |
| unsloth_gemma-2-9b | refused (nf4 base, streamed): device 17.34 + headroom 0.77 GiB > budget 15.50 GiB | nf4 resident 21.8 GiB | nf4 resident 21.8 GiB | bf16 resident 35.5 GiB |
| unsloth_gemma-3-12b-it | refused (nf4 base, streamed): device 18.49 + headroom 0.77 GiB > budget 15.50 GiB | nf4 streamed 18.5 GiB | nf4 resident 24.2 GiB | bf16 resident 42.0 GiB |
| unsloth_gemma-3-27b-it | refused (nf4 base, streamed): device 22.45 + headroom 0.77 GiB > budget 15.50 GiB | refused (nf4 base, streamed): device 22.45 + headroom 1.18 GiB > budget 23.52 GiB | nf4 streamed 22.5 GiB | nf4 resident 36.3 GiB |
