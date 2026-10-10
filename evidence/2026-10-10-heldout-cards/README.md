# HO1: held-out cards

Registration: [`bench/HO1-PREREG.md`](../../bench/HO1-PREREG.md), merged in #59 at `a8f5e89f`. This is the read, against
that commit.

**Verdict: HO1_UNDER.** One of the 14 primary RTX 5090 setups is under its plan: Qwen3-30B-A3B at 4096 × 1, fused
kernel, fp32 adapters. Its sampled driver peak is 31.33 GiB against a 30.61 GiB plan (1.023), and 9 of its 27 arms
are over.

The driver peak is a 1 Hz sample, so it can only read low. An UNDER is conclusive; "not shown under" is not proof of
cover.

## What was run

- **Plans:** `python bench/ho1_replan.py --e4b E4B --rows evidence/2026-10-10-heldout-cards/rows.json --out
  evidence/2026-10-10-heldout-cards --a2000-evidence evidence`.
- **Versions:** Loggetta 0.5.0 and experts4bit-qlora 0.52.0 from PyPI, transformers 5.18.0, torch 2.14.1, on CPU,
  macOS. Configs came from the Hub.
- **Receipts:** none on file.
- **Variant:** with the 15 committed A2000 training receipts on file. It changed no plan, as the lookup rules said
  it would.
- **Outputs:**
  - `ho1.json` holds every setup: 45 in all, primary and excluded, with per-arm rows.
  - `RESULTS-table.md` is rendered from it by `bench/ho1_report.py`.

## Primary setups

See [`RESULTS-table.md`](RESULTS-table.md). Counts on the 14 RTX 5090 setups:

| | setups |
|---|---|
| UNDER (sampled driver peak above the plan) | 1 |
| estimate short (exact: allocated peak above the allocator estimate) | 5 |
| false refusals (the plan refuses a setup that ran) | 5 |

The 2 H100 setups are reported, not graded. The fused Qwen3-30B H100 setup's estimate is also short (1.012).

## Against the prediction

- **"No primary setup UNDER": wrong.** One is.
- **"Driver/plan 0.80–0.95": wrong.** The range is 0.686 to 1.023:
  - granite and OLMoE sit at 0.69–0.83, over-planned;
  - Qwen3-30B fused sits at 0.86–1.02.
- **False refusals:** the prediction named six setups.
  - Expected: three of the four were refused (Qwen3-30B 4096 × 1 fp32, both Mixtral setups). Qwen3-30B 4096 × 1
    bf16 was not (plan 27.10 GiB, feasible).
  - Borderline: both were refused (Qwen3-30B 2048 × 2 fp32, fused and reference).
  - That makes five false refusals.
- **"Estimate shortfalls, if any, on older e4b releases": wrong.** Every fused Qwen3-30B arm is over the estimate, in
  every e4b release the runs used (0.40.0 to 0.48.0).
  - At 2048 × 2 the shortfall shrinks on newer releases but never closes: fp32 goes from 1.082 (0.40.0) to 1.021
    (0.48.0), bf16 from 1.072 to 1.037.
  - The 4096 × 1 setups ran only on 0.48.0.

## What the table shows

These are observations, not tested causes:

- **The estimate misses on Qwen3-30B with the fused kernel, and only there.** The other models, and the reference
  kernel on every model, have estimates above their allocated peaks. Fused Qwen3-30B runs 0.9–21% over.
- **Sequence length.** The estimate is the same at 4096 × 1 as at 2048 × 2: 22.16 GiB bf16 and 25.09 GiB fp32, the
  same tokens per micro-batch. The measured peak is higher at 4096 × 1: 25.12 vs 23.76 GiB bf16, and 30.33 vs
  27.15 GiB fp32. The excess grows with sequence length at fixed tokens. That points at a term the estimate does not
  scale with `seq`, which is a hypothesis for experts4bit-qlora to test, not a finding here.
- **The 20% reserve covered the estimate's shortfall everywhere except at 4096 × 1 fp32.** At 4096 × 1 bf16 it held by
  0.03 GiB (driver/plan 0.999).
- **Refused and under at once.** The UNDER setup's plan was also refused. The planner said the run would not fit, the
  run fitted, and its peak was above the plan anyway. On a 32 GB card, the plan's total and the card's limit sit within
  a gigabyte of each other.

## Not held out

These are as registered: the estimator's development saw these runs, the defaults date from FP1, the runs' code is
older than the plan's, and Qwen3-30B dominates. The read covers one card, in practice: the RTX 5090.
