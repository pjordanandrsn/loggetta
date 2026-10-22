- `evidence/planner-split/SPLIT-v2.json`: the planner split's version 2 (`SPLIT.json` stays frozen and unchanged). It adds HO2's eight
  held-out rows: LFM2-8B-A1B and ERNIE-4.5-21B-A3B, each at 1,024 and 2,048 tokens, on an RTX 5090 and an RTX 4090 (experts4bit-qlora
  `bench/ho2/`, #1969 and #1987), all UNSEEN and all COVERED. Four family × card cohorts now hold the 2 rows a band needs; the band is
  L2's. Every row cites its public receipt copy by sha256 beside the private original. These are device-placement rows only, and
  each row's loss branch is inferred, not counted.
