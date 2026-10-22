- `evidence/planner-split/`: the planner's frozen derivation / held-out split (T2). Every plan/measured pair on file is tagged SEEN (a
  planner constant was set or tuned after it) or UNSEEN. The held-out side has one row, DQ7's resident Qwen3-14B at 4096 tokens, so every
  family × card is unmeasured and no band is printed yet. The band reads only this file.
