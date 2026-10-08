# MoE plans against the driver's measured peak

**Question** (the maintainer, on the bus, after DQ7): DQ7 found every streamed dense plan's device total below the
measured driver peak, by up to 2.40 GB. Do Loggetta's MoE (`experts4bit`) plans hold?

**Method.** One script reads every committed receipt that carries a Loggetta plan total and a measured driver peak.
No number below was typed by hand.

```sh
PYTHONPATH=.:E4B:GNF4/kernel HF_HUB_OFFLINE=1 python bench/plan_vs_driver.py --e4b E4B --replan \
  --json evidence/2026-10-08-moe-plan-vs-driver/plan-vs-driver.json > evidence/2026-10-08-moe-plan-vs-driver/plan-vs-driver.md
```

It was run with loggetta `efe1c93` (main), experts4bit-qlora `6357ecc3` (main), transformers 5.18, on CPU.
[`plan-vs-driver.md`](plan-vs-driver.md) is its whole output, unedited, including the every-receipt table, and
`plan-vs-driver.json` holds the rows in bytes.

- **Plan total vs driver peak.** The plan's device total (allocator estimate + reserve + CUDA context) against the
  process's driver-reported peak.
- **Allocator estimate vs allocated peak.** The plan without its reserve and context lines, against PyTorch's
  allocated peak.

**Sources:**

1. Loggetta ExecutionReceipts with the plan attached: 23, all on the RTX A2000.
2. experts4bit-qlora SV5, SV6 and SV7, each with its registered plan total (`PLAN_BYTES`, `EST_BYTES` in the merged
   reducer), beside its arm receipts on an RTX 4090. That gives five arms with a driver reading. SV5's long arm ran out
   of memory, and SV7's first run sampled no driver.
3. Receipts with a driver peak but no plan: FP1 and SV1–SV4. They give the allocator comparison only.

Dense receipts are DQ7's (experts4bit-qlora#1359).

`--replan` plans each training receipt's exact setup, workload and stated hardware again with today's code, twice:
with no receipts, and with its sibling receipts on file. Its own receipt is left out, so a plan is never taught by
the run it is compared with.

## What the script printed

### Summary (receipts with a plan total and a driver peak)

| card | kind | placement | receipts with plan + driver | plan under driver | driver / plan, min–max |
|---|---|---|---|---|---|
| GeForce RTX 4090 | serve | offload (solver tiers) | 4 | 0 | 0.902–0.981 |
| GeForce RTX 4090 | serve | resident (all-VRAM) | 1 | 0 | 0.988–0.988 |
| RTX A2000 12GB | serve | offload (solver tiers) | 2 | 0 | 0.879–0.939 |
| RTX A2000 12GB | serve | resident (all-VRAM) | 15 | 0 | 0.749–0.870 |
| RTX A2000 12GB | train | offload (experts on host) | 2 | 2 | 1.097–1.237 |
| RTX A2000 12GB | train | resident (experts on device) | 5 | 2 | 0.846–1.184 |

### Where the plans under their driver peak went short (GiB)

| run | driver − plan | allocated − estimate | (reserved − allocated) − reserve line | (driver − reserved) − context line | the plan's reserve line | the plan's context line |
|---|---|---|---|---|---|---|
| 2026-10-04-rtx-a2000/OLMoE-1B-7B-0924-device-grouped_nf4-t1024-20261004T173932Z | +1.058 | +0.213 | +1.213 | -0.367 | None | inferred: default; no receipt on file measured this GPU + driver |
| 2026-10-04-rtx-a2000/OLMoE-1B-7B-0924-device-grouped_nf4-t1024-20261004T175201Z | +0.260 | +0.213 | +0.047 | +0.000 | measured: receipt OLMoE-1B-7B-0924-device-grouped_nf4-t1024-20261004T173932Z: reserved peak / allocated peak - 1 = 0.222 | measured: receipt OLMoE-1B-7B-0924-device-grouped_nf4-t1024-20261004T173932Z: driver-reported process peak minus allocator reserved peak |
| 2026-10-04-rtx-a2000/OLMoE-1B-7B-0924-host-grouped_nf4-t1024-20261004T175357Z | +0.639 | +0.213 | +0.426 | +0.000 | measured: receipt OLMoE-1B-7B-0924-device-grouped_nf4-t1024-20261004T175201Z: reserved peak / allocated peak - 1 = 0.222 | measured: receipt OLMoE-1B-7B-0924-device-grouped_nf4-t1024-20261004T175201Z: driver-reported process peak minus allocator reserved peak |
| 2026-10-04-rtx-a2000/OLMoE-1B-7B-0924-host-grouped_nf4-t1024-20261004T182135Z | +0.295 | +0.214 | +0.081 | +0.000 | measured: receipt OLMoE-1B-7B-0924-host-grouped_nf4-t1024-20261004T175357Z (same residency and kernel): reserved / allocated peak - 1 = 0.386 | measured: receipt granite-3.1-3b-a800m-instruct-device-grouped_nf4-t1024-20261004T180058Z: driver-reported process peak minus allocator reserved peak |

### Today's planner on the same training setups

| run | allocated peak | today's allocator est. | driver peak | today, no receipts: plan total (driver / plan) | today, siblings on file: plan total (driver / plan; reserve; context) |
|---|---|---|---|---|---|
| 2026-10-04-rtx-a2000/OLMoE-1B-7B-0924-device-grouped_nf4-t1024-20261004T173932Z | 5.471 | 5.258 | 6.816 | 6.810 (1.001) **UNDER** | 6.589 (1.034) **UNDER**; reserve measured 1.165; context measured 0.166 |
| 2026-10-04-rtx-a2000/OLMoE-1B-7B-0924-device-grouped_nf4-t1024-20261004T175201Z | 5.471 | 5.258 | 6.816 | 6.810 (1.001) **UNDER** | 6.590 (1.034) **UNDER**; reserve measured 1.166; context measured 0.166 |
| 2026-10-04-rtx-a2000/OLMoE-1B-7B-0924-device-reference-t1024-20261004T175624Z | 5.268 | 5.258 | 6.277 | 6.810 (0.922) | 7.454 (0.842); reserve measured 2.030; context measured 0.166 |
| 2026-10-04-rtx-a2000/OLMoE-1B-7B-0924-host-grouped_nf4-t1024-20261004T175357Z | 2.307 | 2.094 | 3.330 | 3.013 (1.105) **UNDER** | 3.067 (1.086) **UNDER**; reserve measured 0.807; context measured 0.166 |
| 2026-10-04-rtx-a2000/OLMoE-1B-7B-0924-host-grouped_nf4-t1024-20261004T182135Z | 2.308 | 2.094 | 3.330 | 3.013 (1.105) **UNDER** | 3.069 (1.085) **UNDER**; reserve measured 0.808; context measured 0.166 |
| 2026-10-04-rtx-a2000/granite-3.1-3b-a800m-instruct-device-grouped_nf4-t1024-20261004T180058Z | 3.090 | 3.035 | 3.928 | 4.142 (0.948) | 3.874 (1.014) **UNDER**; reserve measured 0.673; context measured 0.166 |
| 2026-10-04-rtx-a2000/granite-4.0-h-tiny-device-grouped_nf4-t1024-20261004T183236Z | 6.686 | 6.619 | 7.406 | 8.443 (0.877) | 8.262 (0.896); reserve measured 1.510; context measured 0.133 |

## Reading

- **Serving holds.** All 22 serve plans with a driver reading are over their driver peak:
  - the A2000, resident and solver tiers: driver / plan 0.749–0.939;
  - SV5–SV7's registered 4090 plans: 0.902–0.988.
- **Training does not.**
  - **As run, 4 of the 7 A2000 training plans were under.** These were OLMoE, resident and with experts on host.
    - The first plan of 2026-10-04 had no reserve line: its 1.06 GiB shortfall is mostly unpriced reserve.
    - Once the receipts taught the reserve, the shortfall that remains is the allocator estimate: allocated −
      estimate is +0.213 GiB in every OLMoE run, plus that amount times the slack fraction.
    - A host plan that borrowed the resident slack (22.2% for the host arm's 38.6%) also missed 0.43 GiB of reserve.
  - **Today's code reproduces it.** The allocator estimates for these setups are byte-identical to 2026-10-04's.
    - OLMoE resident plans 1.001 with no receipts, and 1.034 with its sibling receipts.
    - OLMoE with experts on host plans 1.105 and 1.086.
    - granite-3.1-3b plans 1.014 with its siblings, because it then borrows OLMoE's reserve.
  - **The same +0.21 GiB appears elsewhere.** It is the same on both placements of OLMoE. FP1's OLMoE run on the RTX
    5090 measured +0.211 GiB over its own estimate. granite-3.1 shows +0.055 and granite-4.0-h +0.067.
  - **No training residual is learned for MoE plans,** so a learned reserve cannot cover an allocator underestimate.
- **Not shown:**
  - rented-card MoE *training* against a plan total: FP1 carried e4b's estimate, not a Loggetta plan;
  - any training micro-batch or sequence other than the A2000 runs' 2 × 512;
  - why the allocator term is +0.213 GiB.

No code, coefficient or default changed. No GPU was used and nothing was rented. As the maintainer asked, the gap was
reported on the bus before any fix was proposed; this report proposes none.
