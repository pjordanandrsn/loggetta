### HO1 read: on the RTX 5090, one of 14 setups is under its plan

`evidence/2026-10-10-heldout-cards` re-plans committed experts4bit-qlora training runs, as registered in #59. The
plans use Loggetta 0.5.0 with no receipt on file.
- One RTX 5090 setup is under its plan: Qwen3-30B-A3B at 4096 × 1, fused kernel, fp32 adapters, with driver/plan
  1.023.
- The exact estimate is short on 5 setups, all fused Qwen3-30B.
- Five setups that ran are refused.

Scripts: `bench/ho1_replan.py` and `bench/ho1_report.py`. No code changes.

README and PYPI now say, beside the 0.5.0 in-sample line, that the held-out check found one RTX 5090 setup over its plan.
