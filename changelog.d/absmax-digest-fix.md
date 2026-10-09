### Fix: resident `grouped_nf4` training no longer stops at the frozen-expert digest (#47)

- **What changed.** `_expert_digest` hashes the frozen expert absmax as it is stored. That is the double-quantized
  payload (`<which>_absmax_q`, `_s`, `_off`, `_code`) when experts4bit-qlora compressed it, and the fp32 buffer
  otherwise. Nothing is decompressed. With the fp32 absmax the digest is byte-for-byte the earlier one.
- **Why.** Since experts4bit-qlora 0.49.0, `enable_fast_train` double-quantizes the absmax by default for resident
  training, and a guard under the old `<which>_absmax` name raises on any use. With loggetta 0.3.x and
  experts4bit-qlora 0.49.0 or later, resident `grouped_nf4` training raised `AbsmaxCompressedError` before step 1.
  The workaround was `E4B_ABSMAX_DQ=0`. Host residency and the reference kernel were not affected.
- **Evidence.** Reproduced on the released pair (loggetta 0.3.1 + experts4bit-qlora 0.50.0) on an RTX A2000. The
  fixed digest trains the same setup to an OK receipt (`evidence/2026-10-09-a2000-absmax-digest-repro`).
- **Tests.** `tests/test_absmax_digest.py` builds the model through `prepare_qlora_training`, which calls
  `enable_fast_train`, on a tiny local MoE on the CPU, with experts4bit-qlora's default. It fails before the fix and
  passes after, with experts4bit-qlora 0.49.0 and 0.50.0. The released-backend CI job fails if this test skips.
