### Adapters: the MoE manifest records the quantized base it was trained on (`backend-adapter/2`)

A `backend-adapter/1` manifest records the setup and the base revision, but not what the base was quantized to. A
server rebuilding the base therefore cannot tell an fp32 expert absmax from the double-quantized one training uses by
default. With an experts4bit-qlora that defines `backend-adapter/2`, the MoE executor now writes it. It records:
- the expert absmax form, the attention recipe, the fused q/k/v count and the versions, copied from
  `prepare_qlora_training`'s report;
- the LoRA scale the modules held;
- a `qfp-v1` fingerprint of the quantized base (experts always; attention when the adapter trains it), read where
  training kept the stacks (`expert_residency`: device or host);
- the run id and the time written.

A fact that cannot be recorded refuses the save before the directory exists. A run recorded as device whose stacks
turn out to be evicted is refused, never fingerprinted from a host copy nobody recorded. `load_adapter` fingerprints
the rebuilt base and refuses a mismatch, and reading a `/2` directory goes through experts4bit-qlora's own verifier.
`loggetta inspect --adapter DIR` prints it all. With an older experts4bit-qlora the manifest stays `/1`, as before.
The dense executor writes `/1`.
