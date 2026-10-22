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

## SPLIT-v2: HO2's eight held-out rows

`SPLIT-v2.json` is version 2. `SPLIT.json` stays frozen; every v1 row and reason is carried over unchanged. v2 adds the eight rows of
HO2, the held-out read this README asked for above ("2 MoE families untouched by any constant, at their defaults, on an RTX 5090 and an
RTX 4090"), registered in experts4bit-qlora before any measurement (`bench/ho2/PREREG-ho2.md`, #1969) and read in #1987.

| cohort | held-out rows | card |
|---|---|---|
| LFM2-8B-A1B / resident / RTX 5090 | 2 (1,024 and 2,048 tokens) | `ho2-5090-1` |
| ERNIE-4.5-21B-A3B-PT / resident / RTX 5090 | 2 | `ho2-5090-1` |
| LFM2-8B-A1B / resident / RTX 4090 | 2 | `ho2-4090-2` |
| ERNIE-4.5-21B-A3B-PT / resident / RTX 4090 | 2 | `ho2-4090-2` |

- **UNSEEN:** each family was named before any measurement, and no planner constant was set or tuned after any of these rows. Every
  row used the planner's own plan with no receipts on file, and all eight were COVERED.
- **Cited by sha256:** each row cites the derived public copy of its Loggetta receipt in experts4bit-qlora at `512614e6`
  (`bench/ho2/receipts/…/public/…` in experts4bit-qlora) by `sha256`. It also cites the private store's original by `store_sha256`. The two differ
  only in the GPU-UUID values, which the copy sets to `null`.
- **Device placement only:** every row planned and ran experts on device. Host offload has no held-out row yet.
- **Loss branch:** each row records it, inferred and not counted: the 1,024-token rows ran the stock loss, the 2,048-token rows the
  chunked loss, both under experts4bit-qlora's auto gate. The basis is each receipt's own plan: at 1,024 tokens it prices the logits
  held by the caller (0.25 GB LFM2, 0.395 GB ERNIE), and at 2,048 none. The measured peaks fall with it. LFM2's 2,048-token rows sit
  exactly at the gate (4,096 × 65,536 × 4 = 1 GiB). Receipts written from now on count the branch (#94).
- **Bands:** four cohorts now have 2 held-out rows, the minimum, so they are eligible for a band. The band itself is L2's; v2 computes
  none. Qwen3-14B / resident / RTX 5090 still has 1 row and stays unmeasured.
- **One box per cohort:** each cohort's 2 rows come from ONE box and one host draw, at two seq lengths. A band built from them
  carries no host-to-host variation, so L2 must say so, or require rows from 2 boxes.
- **Pins:** `pinned_at` moves to experts4bit-qlora `512614e6` and loggetta `9e005d68`. `fills_next` drops the HO2 item it now holds.

v2 is frozen at merge like v1: any change is a new versioned file.
