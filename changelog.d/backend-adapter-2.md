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

Everything `/2` records is checked right after the model is prepared, BEFORE the first training step, so a run that
could not record it is refused as not run instead of failing after training. The fingerprint is taken then too: the
base is frozen for the run. A run recorded as device whose stacks are evicted is refused, never fingerprinted from a
host copy nobody recorded. Trained tensors are never discarded over manifest metadata: if `/2` can only fail at save,
the adapter is written as `/1`, its manifest and the receipt say why (`v2_refused`), and the run reads `SAVED_V1`
(its measurements still count as planning evidence). `load_adapter` fingerprints the rebuilt base and refuses a
mismatch, and reading a `/2` directory goes through experts4bit-qlora's own verifier.
`loggetta inspect --adapter DIR` prints it all. With an older experts4bit-qlora the manifest stays `/1`, as before.
The dense executor writes `/1`.
