### Check cached generation after dense adapter reload

Require bitwise resident/streamed logits, KV cache tensors and greedy generated
tokens after saved Llama and Qwen3 adapter reload. A corrupted cached-prefix control
must fail the same comparison. Older e4b CPU inference retains the explicit #1539
expected-failure marker; fixed runtimes exercise the streamed generation path.
