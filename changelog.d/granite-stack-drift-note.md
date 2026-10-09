### Docs: granite-3.1's 32 MiB peak drift between two stacks is recorded, not attributed

`evidence/2026-10-09-a2000-granite-train-residual/STACK-DRIFT.md`. The same setup on the same RTX A2000 peaked
3,163.9 MiB on the 2026-10-04 stack and 3,131.5 MiB today. 27.6 of the 32.4 MiB is already present after load, the
direction is safe, and the estimate covers both runs. Four components moved at once (experts4bit-qlora 0.44 → 0.51,
grouped-nf4-gemm 0.37 → 0.44, torch 2.8 → 2.11, transformers 5.18 → 5.19), so it is not attributed. A drift the other
way, or past the estimate, would reopen it. No code changes.
