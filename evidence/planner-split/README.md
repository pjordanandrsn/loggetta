# The planner's frozen split: which receipts may test a band

`SPLIT.json` lists every plan/measured pair on file when it was frozen, and puts each on one side:
- **derivation:** every row the planner was built or tuned on;
- **held-out:** the rows a band may be tested against.

The band itself (the planner printing a per-family × card range beside each estimate) reads only this file.

## The rule

A row is **SEEN** if any planner constant was set or tuned after looking at it. Planner constants are the allocator reserve, the
context allowance, the overheads, and the estimate's own terms. Otherwise the row is **UNSEEN**. Only UNSEEN rows sit on the held-out side.

Reading a row is not tuning: a row that was looked at but never fed a constant stays UNSEEN. Where the record was unclear, the row was
tagged SEEN. A family × card cohort with fewer than 2 held-out rows is **unmeasured**, and the planner prints
`unmeasured family/card: no band` for it.

## What was found

The held-out side has **one row**:

| row | cohort | why UNSEEN |
|---|---|---|
| `dq7-5090-1`, `read-qwen3_14b-device-4096.json` (sha256 `dcf1c95d…`) | Qwen3-14B / resident / RTX 5090 | D7's held-out rung. #26 covers streamed arms only, #28 covers Llama only, and DQ10 was fitted on DQ9's own reads. |

No cohort has 2 held-out rows. So **every family × card is unmeasured today**, and no band is printed anywhere. That is the correct output
until the held-out side fills.

Everything else is derivation. `SPLIT.json` gives each set's reason and citation:
- **loggetta's own evidence** (26 directories). Each was added by the change that read it, and HO1's README says it is not held out.
- **The register rows** in `bench/validate_register.py`. They were read on the day the defaults were set, so they are tagged SEEN
  conservatively.
- **In experts4bit-qlora:**
  - DQ4's brackets, against which the dense estimate was checked (#11);
  - DQ7's other 13 completed arms: streamed (#26), Llama (#28), and the rungs below 4096 (D7's derivation);
  - DQ9's 16 fitting reads (the DQ10 policy);
  - the TC1 runs that issue #1526's estimate terms were fitted or checked on.

## What fills the held-out side next

- loggetta#21's registered RTX 4090 reading.
- DQ10's new-family holdouts, when they read.
- A held-out read registered for this split: 2 MoE families untouched by any constant, at their defaults, on an RTX 5090 and an
  RTX 4090. It names its families before any measurement.

## Frozen

This file is frozen at merge. Any change is a new versioned file (`SPLIT-v2.json`) with its own review, never an in-place edit. Every
set is pinned by git object id at the commits in `pinned_at`, and every DQ7 row by sha256. The held-out row's bytes match the private
receipt store's copy.
