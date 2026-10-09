### Dense plans: the activation item is the larger of its two branches as returned

- **What changed.** The dense activation item chose its branch with `T × layer_work > loss_bytes`, then returned the
  layer branch scaled by `ACTIVATION_COEFFICIENT` (1.178). The comparison left the coefficient out, so a larger loss
  workspace could switch it to the smaller branch: an estimate that fell as one of its terms rose. It now computes
  both branches as returned and takes the larger.
- **The fallback.** For an experts4bit-qlora without `chunked_loss_bytes`, the fallback's bytes per chunk logit are now
  `CHUNK_LOSS_BYTES_PER_LOGIT` (12, measured in experts4bit-qlora #1504), up from a literal 10.
- **Effect, in sample on the 36-case dense matrix.**
  - With today's experts4bit-qlora, four Qwen3-14B plans at 2,048 tokens rise 24.0 MiB: the cases the old comparison
    under-priced. No plan falls.
  - With experts4bit-qlora #1505 (12 B per chunk logit) as well, every changed plan rises (+178.1 MiB). Qwen3-32B at
    2,048 tokens, which #1505 alone would have lowered by 185.1 MiB, is unchanged.
- **Tests.**
  - `test_a_larger_loss_term_never_lowers_the_activation_estimate`, at Qwen3-32B and 2,048 tokens: it fails before the
    fix and passes after.
  - The DQ9 prior-estimates snapshot is unchanged. Its two affected rows get an explicit activation correction in
    `tests/fixtures/dq9/activation-max-correction.json`, and the test checks that they now take the layer branch.
