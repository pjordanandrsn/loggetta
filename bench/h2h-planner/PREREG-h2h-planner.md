# The planner head-to-head at the release: Loggetta's plan against the measured held-out rows, beside Accelerate

This registers one descriptive row of the release-day head-to-head: what Loggetta's planner estimates for the held-out
training rows, against what those rows measured, beside what a common published estimator gives for the same models.
CPU only: no GPU, no rental, USD 0. It changes no estimate, constant, band or split.

## The rows

Every held-out row of `evidence/planner-split/SPLIT-v3.json` (frozen; merged in #120), 21 rows:

| cohort | rows | boxes |
|---|---|---|
| LFM2-8B-A1B / resident / RTX 5090 | 4 | 2 |
| ERNIE-4.5-21B-A3B-PT / resident / RTX 5090 | 4 | 2 |
| LFM2-8B-A1B / resident / RTX A4000 | 4 | 2 |
| ERNIE-4.5-21B-A3B-PT / host offload / RTX A4000 | 4 | 2 |
| LFM2-8B-A1B / resident / RTX 4090 | 2 | 1 (one box) |
| ERNIE-4.5-21B-A3B-PT / resident / RTX 4090 | 2 | 1 (one box) |
| Qwen3-14B / resident / RTX 5090 (dense) | 1 | 1 (one box) |

Each row's receipt is read from experts4bit-qlora at the split's `pinned_at` (`a4b2b98f`) with `git show`, and refused
unless its sha256 equals the split's. No row is added, dropped or re-tagged.

## The subject and the harness

- **The planner:** Loggetta 0.7.0, installed from PyPI, with the backends' estimators from experts4bit-qlora 0.58.0
  and grouped-nf4-gemm 0.51.0, the releases tagged today. The installed versions are recorded. If either tag does not
  exist when the read runs, the row reads "not read today" and no older number stands in.
- **The harness:** `bench/insample_replan.py`'s own planning (`_plan`: the receipt's recorded card, workload and setup,
  no receipts on file; the dense row through `dense.describe` from DQ7's committed config, as its `dense` mode does).
  `bench/h2h_planner.py` feeds it the split's rows and grades each new plan against the receipt's own measured peaks with
  `execution.compare`, the function every receipt's `comparison` comes from.
- **Beside it, the previous release:** each receipt's recorded `comparison`, the estimates made when it ran
  (Loggetta 0.6.2 for HO2 and HO3), quoted as recorded and not re-measured.

## The comparator

Hugging Face Accelerate 1.15.0 (its latest release), through its own CLI exactly as a user runs it, once per model:

```
accelerate estimate-memory MODEL --library_name transformers --dtypes float16 int4
```

Its command, version and full stdout are kept in the output. It sizes a model's weights from its config on the meta
device, at the Hub's current revision (the CLI takes no revision), and prints a "Training using Adam" figure for full
precision. **It does not model QLoRA, LoRA, activations, the allocator or offload.** Its column is labelled "a
full-precision Adam estimate, not like-for-like" and is a reference: no row is ruled on it, and no "win" is claimed
against it. Its sizes are read as printed, in its own 1024-based units.

## What is read (descriptive; no bar)

For each row:
- **Loggetta now and then:** plan total / sampled driver peak and allocator estimate / allocated peak, with host estimate
  / host peak on the host-offload rows. The row's outcome is the planner's own rule from HO2 and HO3: COVERED when every
  estimate is at least its peak; otherwise UNDER, SHORT or HOST_UNDER, each named.
- **Accelerate:** its int4 total, its float16 total and its float16 "Training using Adam" figure, each beside the row's
  sampled driver peak.

The results quote the held-out band's wording ("observed on N held-out rows from M machines"; one-box cohorts say so),
and the negatives beside the wins. A row the planner refuses reads as refused.

## Launch and budget

The read runs on a development host's CPU, from this registration's merge commit, after the tags exist and after an
Approved that names the merge commit, as `python bench/h2h_planner.py evidence/planner-split/SPLIT-v3.json --e4b
E4B_CHECKOUT --out DIR --expect loggetta=0.7.0 --expect experts4bit-qlora=0.58.0 --expect grouped-nf4-gemm=0.51.0`:
any other installed version refuses ("not read today"). It writes `evidence/<date>-h2h-planner/h2h-planner.json` and `accelerate.json`.
The maintainer re-derives it by running the same command. USD 0.

## What was seen before this page (stated, not hidden)

- **One row re-planned** as a smoke of the harness (`ho3-a4000-r2/ernie/host/2048`, at Loggetta main after 0.7.0 with
  an experts4bit-qlora development checkout): plan device 8,777,066,624 and host 15,451,815,936 bytes, the same as that
  receipt's recorded plan.
- **Accelerate on ERNIE-4.5-21B-A3B-PT**, as a smoke of the CLI: float16 total 40.65 GB, Adam training 162.61 GB, int4
  total 10.16 GB (no int4 training figure).
- Every row's measured peaks and recorded estimates were seen when SPLIT-v3 and the L2 band were built.

## What this row cannot say

- Nothing about any card, model or recipe outside the 21 rows.
- Nothing about a further machine: the rows are 2 boxes per cohort at most.
- Nothing like-for-like about Accelerate, which estimates something else.
