### Check dense adapter artifacts across streamed placement and storage dtypes

Exercise real Llama and Qwen3 adapter training, export and public inference reload
with resident or streamed bases and fp32 or bf16 adapters. Require bitwise tensors
and logits, and reject a missing adapter tensor even when its file checksum passes.
