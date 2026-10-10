### Dense accumulated training parity

Add CPU fp32 checks that resident and streamed dense training preserve bitwise-equal losses, adapter updates and complete AdamW state after each accumulated update, including uneven supervised-token masks and skipped empty steps. A corrupted streamed host-home mutant fails the same equality check. These tests establish recipe semantics on tiny local Llama and Qwen3 fixtures, with no capacity or timing claim.
