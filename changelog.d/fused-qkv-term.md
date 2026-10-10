### Tests: the fused q/k/v training projection's term reaches the plan, and its knob turns it off

experts4bit-qlora#1560 turns the fused q/k/v training projection on by default. Its estimate prices the fused
projections' fp32 absmax: 3 B per 64 q/k/v values, 23.6 MB on Qwen3-30B-A3B. Loggetta defers to that estimate.
`tests/test_fused_qkv_term.py` pins four things: the line is in the plan under the default, `E4B_TRAIN_FUSE_QKV=0`
drops it, it appears only where the run fuses (NF4 attention, trained attention, the grouped kernel), and the plan's
`estimate_env` records the knob. The tests skip against an experts4bit-qlora that predates the term. No code changes.
