### MoE training with `grouped_nf4`: the MoE layer backward's working set is priced (#43)

- **What changed.** A `grouped_nf4` training estimate carries a new device line, `grouped_nf4 MoE backward (above the
  activation item)`: how much `boundaries + padded LoRA delta + fused workspaces` exceeds experts4bit-qlora's activation
  item (`boundaries + max(logits, 2 x one layer)`), or nothing when it does not.
  - **Padded LoRA delta:** grouped-nf4-gemm's `_lora_delta_padded` holds `rows x (H + first_out + I)` in the adapter
    dtype. Its rows are data-dependent, so they are a bound in shape, not a measured constant: `min(E, T x top_k) x T`
    for the single padded block, capped where the `auto` route would take the loop (2 GiB padded block), on the
    single-block ladder for fp32 adapters (grouped-nf4-gemm 0.44.0's `NF4_QLORA_SINGLE_LADDER=auto`: at most 1.5625x),
    and `2 x T x top_k` once a call has 16,384 routed rows and pads by buckets.
  - **Fused workspaces:** `5 x T x top_k x first_out` bf16.
  - The expert LoRA gradients that are live there are already priced, in the `adapter gradients` line, and are not
    counted again.
  - grouped-nf4-gemm exposes no sizing helper, so its route constants and ladder are mirrored, and a test pins them
    against the installed package. The kernel's run-time overrides (`NF4_QLORA_*`) are listed as unmodelled.
  - The `grouped-nf4-gemm` floor is now 0.42.0, the first release whose bucketed padding is the default.
- **Why.** loggetta#44 replayed the allocator at OLMoE's `grouped_nf4` training peak on the RTX A2000: the peak is in
  an MoE layer's backward, with 500.9 MiB of padded delta and 167.1 MiB of fused workspaces live, where the estimate
  took the loss branch. Estimates were 195–219 MiB under the allocated peak.
- **In-sample only.** Against the committed OLMoE `grouped_nf4` receipts (#29's A2000 and FP1's RTX 5090, #44's
  replays) the estimate now sits 90–114 MiB over the allocated peak, the cost of a bound. The reference kernel, the
  Qwen3-30B receipts (the loss branch is larger) and the granite receipts do not move; the granite runs stay 56–69 MiB
  under, which this term does not explain. `bench/gnf4_terms_in_sample.py` prints the table.
- **Dense plans are unchanged:** byte-identical on a 36-case matrix, before and after.
- **Tests:** `tests/test_gnf4_training_terms.py`, `tests/test_dense_unchanged_by_moe_terms.py`.
