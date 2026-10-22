# T1 step 3: the first-use matrix (preregistered before any data)

**Question.** Does a new user's first install of a Loggetta release candidate end in a trained run or a named refusal,
never a traceback, across NVIDIA drivers below, at, and above the default PyPI torch's CUDA minimum? How long does it take
to reach a trained run?

**What runs.** In each cell, `scripts/first_run.sh` from the release candidate's commit runs in a new directory, with
`PACKAGE_SPEC` = the RC wheel built at that commit. It trains `allenai/OLMoE-1B-7B-0924-Instruct` at revision
`7f1c97f440f06ce36705e4f2b843edb5925f4498` for 10 steps at seq 512 (the script's defaults). The controller records the
SHA-256 of the wheel and of `first_run.sh`, and the box refuses either one if it differs. This runs on the next release
candidate, the one after experts4bit-qlora ships `adapter_schema`, and before the tag.

## Cells

Each cell is graded by the driver **the box reads** (`first_run.json`'s `gpu` field, from nvidia-smi), never by the offer
listing. A box whose read driver falls outside its cell's band reads **CELL_MISMATCH**. That is neither a pass nor a fail,
and the verdict keeps both the listed and the read value. The bands and the expectations are constants in
`fu_reduce.py`, which is the single place they are defined.

| cell | box driver band (major) | TORCH_INDEX | expected exit | expected torch index |
|---|---|---|---|---|
| below-default | 525 ≤ d < 580 | `pypi` | 4 (named torch/driver refusal at plan) | pypi |
| below-matched | 525 ≤ d < 580 | unset (`inspect --torch-index`) | 0 or 7 | cu126, or cu129 on Blackwell |
| at-default | 580 ≤ d < 590 | `pypi` | 0 or 7 | pypi |
| at-matched | 580 ≤ d < 590 | unset | 0 or 7 | cu130 |
| current-default | d ≥ 590 | `pypi` | 0 or 7 | pypi |
| current-matched | d ≥ 590 | unset | 0 or 7 | cu130 |
| cpu-torch | any band (GPU visible) | `cpu` | 4 (torch cannot use the GPU) | cpu |
| no-gpu | no NVIDIA GPU, Linux | unset | 3 (no torch build fits) | none |

`first_run.sh` exit codes:
- 0 = TRAINED, device check passed.
- **7 = TRAINED, device check not run (not a pass of it).**
- 3 = refused: no torch build fits this machine.
- 4 = refused: the plan was refused.
- 5 = refused: an install step failed.
- 6 = refused: the run did not train.
- 2 = usage. It is not a named refusal.

**Below 525.** No such host has appeared in any screen so far. The `cpu-torch` and `no-gpu` cells cover the refusal
paths. Both this file and the README state plainly that a driver below 525 is **untested on hardware**.

## Gate and expectations

- **GATE (strict).** Every graded cell (MET or NOT_MET) exits 0 or 7, or with a named refusal: 3, 4, 5 or 6, its reason
  recorded and no traceback in it. One graded cell outside that fails the gate. A missing cell makes it INCOMPLETE.
- **Expectations (reporting).** Each graded cell reads MET or NOT_MET against the table. A NOT_MET cell is a finding to
  explain, and it does not by itself fail the gate.
- **Times.** Each trained cell's `seconds_to_trained`, with its run id. The README quotes only these measured times and
  run ids that resolve publicly, and no others.

## Boxes, cards, budget

- Three rented boxes, each pinned to a machine by `fu_screen.py`: the band's cheapest passing class among RTX A4000,
  3090, 4090 and 5090, and stable on 6 of 6 machine-filtered reads.
- **below** runs `below-default below-matched cpu-torch`, **at** runs `at-default at-matched`, and **current** runs
  `current-default current-matched`.
- Run ids are `fu-below-N`, `fu-at-N` and `fu-current-N`, with N = 1, plus spares 2 and 3.
- `no-gpu` runs at USD 0 on a Linux host with no NVIDIA GPU, never on a rented box (the runner refuses it).
- Expected cost is about USD 1.2. The worst case is about USD 4.5 (a 1.5 h guard at USD 0.45/h per box, plus download
  fees), inside the USD 15 per release candidate.
- A CELL_MISMATCH box may be re-run once on a spare id pinned to another machine.

## Kit and launch

- The kit is this directory: this file, `fu_run.sh` (box runner), `fu_reduce.py` (reducer, standard library only),
  `fu_screen.py` (host screen; read-only, rents nothing) and `staged-first_use.sha256`.
- The kit hash is sha256(`staged-first_use.sha256`), which pins the two staged files by the names the box sees.
- The launch goes through the launcher from experts4bit-qlora's `bench/first_use/fu_drive.sh`. That driver pins this kit's
  **merge** commit as a 40-hex constant.
- Before any rental action, the driver checks every staged file against `staged-first_use.sha256` at that commit.
- The execution ACK checks three things: the e4b kit hash, the Loggetta commit, and sha256(`staged-first_use.sha256` at
  that commit).
