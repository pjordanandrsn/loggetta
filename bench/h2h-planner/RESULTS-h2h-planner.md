# The planner head-to-head at the release: Loggetta 0.7.0 against the measured held-out rows, beside Accelerate 1.15.0

**Registration:** [PREREG-h2h-planner.md](PREREG-h2h-planner.md) (loggetta#125), read once, as registered, from its merge commit
`8a0b3be6` after the maintainer's Approved. **Records:** [evidence/2026-10-24-h2h-planner/](../../evidence/2026-10-24-h2h-planner/README.md)
(the raw outputs are in the private receipt store; the public copies differ from them only in local paths). CPU only, USD 0.

## The question

Planned again by the released planner, with no receipts on file, does each of SPLIT-v3's 21 held-out rows read the same as the
plan its receipt recorded when it ran? A row is graded by the planner's own rule (HO2's and HO3's reducers): COVERED when the plan total
bounds the driver peak and the allocator estimate bounds the allocated peak, and, with the experts on the host, the host estimate bounds the
host peak; otherwise each failed bound by name (UNDER, SHORT, HOST_UNDER).

## The answer

- **Every row's outcome is unchanged:** 18 COVERED and 3 HOST_UNDER now, the same 18 and 3 as recorded.
- **The plans are the recorded plans:** on 20 of 21 rows every estimate is byte-identical to the receipt's. The one other row,
  `dq7-5090-1/qwen3_14b/device/4096`, was planned when it ran by Loggetta `1dd862f7` (first released in 0.4.0). Its driver term moves +366,912 B
  and its allocator term +305,760 B (both about 0.002 %), its host term +0 B, and it reads COVERED both times.
- **The three HOST_UNDER rows** are ERNIE-4.5-21B-A3B with its experts on the host, on the RTX A4000. That is the shortfall HO3 read, and
  the band (L2) carries it:
  - `ho3-a4000-1/ernie/host/1024`: host peak 14.69 GiB against the estimate 14.39 GiB, +309 MiB
  - `ho3-a4000-r2/ernie/host/1024`: host peak 14.47 GiB against the estimate 14.39 GiB, +82 MiB
  - `ho3-a4000-r2/ernie/host/2048`: host peak 15.18 GiB against the estimate 14.39 GiB, +808 MiB

  Their device terms are covered. No device row runs over: the largest driver peak is 0.9072 of its
  plan total and the largest allocated peak 0.9855 of its allocator estimate.

| row | card | placement | seq | driver peak / plan total | allocated peak / allocator estimate | host peak / host estimate | now | then |
|---|---|---|---|---|---|---|---|---|
| `dq7-5090-1/qwen3_14b/device/4096` | RTX 5090 | resident (device) | 4,096 | 0.9072 | 0.9855 | 0.6376 | COVERED | COVERED |
| `ho2-5090-1/ernie/device/1024` | RTX 5090 | resident (device) | 1,024 | 0.8410 | 0.9625 | 0.6274 | COVERED | COVERED |
| `ho2-5090-1/ernie/device/2048` | RTX 5090 | resident (device) | 2,048 | 0.8171 | 0.9515 | 0.6385 | COVERED | COVERED |
| `ho2-5090-1/lfm2/device/1024` | RTX 5090 | resident (device) | 1,024 | 0.8742 | 0.9539 | 0.6541 | COVERED | COVERED |
| `ho2-5090-1/lfm2/device/2048` | RTX 5090 | resident (device) | 2,048 | 0.8231 | 0.9153 | 0.6213 | COVERED | COVERED |
| `ho2-4090-2/ernie/device/1024` | RTX 4090 | resident (device) | 1,024 | 0.8335 | 0.9625 | 0.5611 | COVERED | COVERED |
| `ho2-4090-2/ernie/device/2048` | RTX 4090 | resident (device) | 2,048 | 0.8102 | 0.9518 | 0.5876 | COVERED | COVERED |
| `ho2-4090-2/lfm2/device/1024` | RTX 4090 | resident (device) | 1,024 | 0.8575 | 0.9539 | 0.5925 | COVERED | COVERED |
| `ho2-4090-2/lfm2/device/2048` | RTX 4090 | resident (device) | 2,048 | 0.8057 | 0.9162 | 0.5383 | COVERED | COVERED |
| `ho3-5090-r1/ernie/device/1024` | RTX 5090 | resident (device) | 1,024 | 0.8397 | 0.9625 | 0.5982 | COVERED | COVERED |
| `ho3-5090-r1/ernie/device/2048` | RTX 5090 | resident (device) | 2,048 | 0.8166 | 0.9519 | 0.6141 | COVERED | COVERED |
| `ho3-5090-r1/lfm2/device/1024` | RTX 5090 | resident (device) | 1,024 | 0.8712 | 0.9539 | 0.6313 | COVERED | COVERED |
| `ho3-5090-r1/lfm2/device/2048` | RTX 5090 | resident (device) | 2,048 | 0.8200 | 0.9151 | 0.6006 | COVERED | COVERED |
| `ho3-a4000-1/ernie/host/1024` | RTX A4000 | host offload (experts on host) | 1,024 | 0.7881 | 0.8578 | 1.0210 | HOST_UNDER | HOST_UNDER |
| `ho3-a4000-1/ernie/host/2048` | RTX A4000 | host offload (experts on host) | 2,048 | 0.7323 | 0.8808 | 0.9959 | COVERED | COVERED |
| `ho3-a4000-1/lfm2/device/1024` | RTX A4000 | resident (device) | 1,024 | 0.8294 | 0.9539 | 0.5744 | COVERED | COVERED |
| `ho3-a4000-1/lfm2/device/2048` | RTX A4000 | resident (device) | 2,048 | 0.7766 | 0.9160 | 0.5225 | COVERED | COVERED |
| `ho3-a4000-r2/ernie/host/1024` | RTX A4000 | host offload (experts on host) | 1,024 | 0.7881 | 0.8578 | 1.0056 | HOST_UNDER | HOST_UNDER |
| `ho3-a4000-r2/ernie/host/2048` | RTX A4000 | host offload (experts on host) | 2,048 | 0.7326 | 0.8803 | 1.0548 | HOST_UNDER | HOST_UNDER |
| `ho3-a4000-r2/lfm2/device/1024` | RTX A4000 | resident (device) | 1,024 | 0.8294 | 0.9539 | 0.5722 | COVERED | COVERED |
| `ho3-a4000-r2/lfm2/device/2048` | RTX A4000 | resident (device) | 2,048 | 0.7762 | 0.9150 | 0.5178 | COVERED | COVERED |

## Beside it: Hugging Face Accelerate 1.15.0

Accelerate's own CLI (`estimate-memory MODEL --library_name transformers --dtypes float16 int4`), run once per model from the same
interpreter whose Accelerate was checked. **It is a full-precision Adam estimate, not like-for-like:** it sizes the weights from the config
and does not model QLoRA, LoRA, activations, the allocator or offload. No row is ruled on it, and no "win" is claimed. Its sizes are as
printed, in its 1024-based units. The planner's and the measured columns are the device driver terms of the rows above.

| model | placement | rows | plan total (GiB) | measured driver peak (GiB) | Accelerate int4 total | Accelerate fp16 total | Accelerate fp16 training with Adam |
|---|---|---|---|---|---|---|---|
| LiquidAI/LFM2-8B-A1B | resident | 10 | 8.74-9.10 | 6.79-7.95 | 3.88 GB | 15.53 GB | 62.14 GB |
| Qwen/Qwen3-14B | resident | 1 | 16.98 | 15.41 | 6.51 GB | 26.06 GB | 104.24 GB |
| baidu/ERNIE-4.5-21B-A3B-PT | host | 4 | 8.17-9.08 | 5.99-7.15 | 10.16 GB | 40.65 GB | 162.61 GB |
| baidu/ERNIE-4.5-21B-A3B-PT | resident | 6 | 19.45-20.36 | 15.76-17.12 | 10.16 GB | 40.65 GB | 162.61 GB |

## Provenance

The four releases, each a PyPI wheel proven byte-for-byte by `bench/release_proof.py` (`199d4f2f` at the merge commit: RECORD equal,
the import origins in the install, a fresh bytecode prefix):
- accelerate 1.15.0: `accelerate-1.15.0-py3-none-any.whl`, sha256 `97eacca0b73e…`
- experts4bit-qlora 0.58.0: `experts4bit_qlora-0.58.0-py3-none-any.whl`, sha256 `6f8e1ff189e1…`
- grouped-nf4-gemm 0.51.0: `grouped_nf4_gemm-0.51.0-py3-none-any.whl`, sha256 `1bf4906dea6e…`
- loggetta 0.7.0: `loggetta-0.7.0-py3-none-any.whl`, sha256 `297923018764…`

Resolved, never pinned: torch 2.14.1, transformers 5.20.2, bitsandbytes 0.50.2, on Python 3.14.6.

**Re-derived** by the maintainer, who ran the same command from `8a0b3be6` in a separate fresh venv of PyPI wheels; the resolver landed
on the same torch, transformers and bitsandbytes. The two raw pairs differ only in their local paths: `project_public.py` makes them
byte-identical public copies (`h2h-planner.public.json` `49e3a544…`, `accelerate.public.json` `0e4529e3…`), so every estimate, measured term and outcome agrees.

## What this does not say

- Nothing about a card, model, placement or sequence length outside SPLIT-v3's held-out rows.
- Nothing about Accelerate beyond what its CLI printed: it estimates another quantity.
- Nothing about speed.
