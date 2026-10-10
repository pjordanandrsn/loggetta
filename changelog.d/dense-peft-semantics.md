### Dense adapter arithmetic contracts

Add CPU integration checks between the actual PEFT linear adapter and e4b's native LoRA wrapper: bitwise fp32 outputs, gradients and accumulated AdamW state across projection shapes, ranks, scaling and checkpoint modes. Pin the distinct delta-add rounding of plain PEFT under a bf16 base with fp32 adapters, and the PEFT bitsandbytes wrapper's pre-add rounding with autocast off. No production arithmetic or default changes.
