# MoE plans against the driver's measured peak, after the grouped_nf4 backward term

The 2026-10-08 audit ([`../2026-10-08-moe-plan-vs-driver/`](../2026-10-08-moe-plan-vs-driver/README.md)) re-run
unchanged after #45 priced the `grouped_nf4` MoE layer backward (#43).

```sh
PYTHONPATH=. python bench/plan_vs_driver.py --e4b E4B --replan \
  --json evidence/2026-10-09-moe-plan-vs-driver-after-gnf4/plan-vs-driver.json \
  > evidence/2026-10-09-moe-plan-vs-driver-after-gnf4/plan-vs-driver.md
```

Run on CPU with loggetta `8e2f951` (main), experts4bit-qlora `02fc0165` (main) as `E4B`, the installed
experts4bit-qlora 0.50.0, grouped-nf4-gemm 0.43.0 and transformers 5.18.0. Configs came from the Hugging Face Hub (no
weights). [`plan-vs-driver.md`](plan-vs-driver.md) is the whole output, unedited.

**In-sample.** These receipts are where the gap was found (#29) and attributed (#44).

- **The plans attached to the receipts are unchanged.** Those are the plans as they ran, so the summary and
  every-receipt tables match 2026-10-08's.
- **Today's planner on the same training setups, no receipts on file.** Every OLMoE plan is now over its driver
  peak: driver / plan 0.950 resident and 0.986–0.987 with experts on host, against 1.001–1.105 on 2026-10-08. The
  reference-kernel plans are unchanged at 0.922. Both granite plans are over (0.948 and 0.877).
- **Siblings on file.** Every OLMoE plan is over, at 0.950–0.987. One plan is still under: granite-3.1-3b-a800m
  at 1.014. Its reserve line borrows OLMoE's measured reserve (#43, item 2). The `grouped_nf4` term does not address that.
