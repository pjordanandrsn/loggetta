### HO1 addendum (post hoc): the over-plan arms ran without pad buckets

`evidence/2026-10-10-heldout-cards` gains a post-hoc addendum. HO1_UNDER stays the registered verdict.
- **The over-plan arms:** all 9 padded the LoRA delta without buckets, by an `NF4_QLORA_PAD_BUCKETS=0` override or on
  grouped-nf4-gemm 0.41.0. The planner does not model that path.
- **The detector:** `bench/ho1_rows.py --posthoc` now flags such arms from what they recorded, pinned by
  `tests/test_ho1_unmodelled.py`.
- **The post-hoc read:** no setup is under.
- **Still open:** the estimate falls short as sequence length grows at fixed tokens (+0.97 GiB fp32, +2.14 GiB bf16),
  tracked in experts4bit-qlora#1526.
- **The 0.5.0 line:** README and PYPI now say every over-plan run used an unmodelled setting.
