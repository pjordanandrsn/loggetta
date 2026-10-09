# MoE plans against the driver's peak, with training reserve borrowing removed

`bench/plan_vs_driver.py --e4b E4B --replan`, unchanged, on this branch: training plans no longer borrow another
model's reserve slack (#43, item 2). It was run on CPU with experts4bit-qlora 0.50.0 installed, experts4bit-qlora
`058f98eb` (v0.51.0) as `E4B`, and transformers 5.18.0. Configs came from the Hugging Face Hub, without weights.
[`plan-vs-driver.md`](plan-vs-driver.md) is the whole output.

**In-sample.** These are the receipts the borrowing was found on.

- **Today's planner, no receipts on file:** unchanged from `../2026-10-09-moe-plan-vs-driver-after-gnf4/`.
- **Siblings on file:** no plan is under its driver peak.
  - granite-3.1-3b-a800m goes from 1.014 (UNDER, reserve borrowed from OLMoE, 0.222) to 0.898. Its reserve is now the
    conservative bound: the largest training slack measured on the A2000, OLMoE with experts on host at 0.386.
  - granite-4.0-h-tiny goes from 0.896 to 0.796, by the same bound. That is the headroom cost of a bound where a
    model's own slack is small (its own: 0.083).
  - OLMoE's rows are unchanged: they have their own receipts.
- **A new row:** `2026-10-09-a2000-granite-train-residual`'s re-run (driver / plan 0.829).

**Correction (#49 follow-up).** As merged in #49, that row cited run G's receipt
(`…/runs/G/receipts/…-20261009T145928Z-ce9e5abb`), which was never committed and is lost (see that directory's README).
`plan-vs-driver.md` and `.json` are regenerated with the same command. The row now cites the re-run G2's committed
receipt (`…/runs/G2/receipts/…-20261009T160437Z-a178340f`). Every number in the row is unchanged: allocated 3.058,
estimate 3.035, driver 3.434, plan 4.142, driver / plan 0.829. The cited run id is the only change.
