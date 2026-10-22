# IN-SAMPLE: where today's training plan misses, on rows the planner was built on

**Every row here is IN-SAMPLE.** The planner's split (`evidence/planner-split/SPLIT.json`) tags each of them SEEN: a planner constant
was set or tuned after looking at it. So this table shows where today's plan misses. It is **not** a band, and no band may be read
from it. The one held-out row (DQ7's resident Qwen3-14B at 4096 tokens) is left out.

## IN-SAMPLE: how it was made

- **Plans:** today's code at loggetta `90a7697` and experts4bit-qlora `46647fc3`, CPU, configs only.
  - With no receipts on file: what a first-time user would be told.
  - Each row's own recorded setup is fixed (`Constraints(fixed=…)`).
- **Measured side:** each row's own receipt, re-derived from the private receipt store:
  - DQ7's 14 reads are byte-identical to the store copies;
  - DQ9's 16 reads were taken from the store and match `bench/dq10/dq9-source-sha256.json`.
- **Peaks:**
  - **driver peak** = the receipt's sampled driver-process peak;
  - **allocated peak** = PyTorch's `max_memory_allocated`.
- **Plan total** = the allocator estimate plus the planner's reserve and context lines.
- GB = 1e9 bytes, except where a heading says GiB.

## IN-SAMPLE: the six register rows (allocator estimate vs allocated peak)

`bench/validate_register.py` at today's code (`register-insample-now.json`).

| row | estimate GB | measured GB | measured − estimate |
|---|---|---|---|
| OLMoE fused_attn4, RTX 5090 (tp4) | 7.256 | 6.419 | −0.837 |
| OLMoE reference_attn4, RTX 5090 (tp4) | 6.258 | 5.866 | −0.392 |
| Granite-3.1 fused_attn4, RTX 5090 (tp4) | 3.823 | 3.782 | −0.041 |
| Qwen3-30B fused_attn4 resident, RTX 5090 (tp4) | 26.800 | 24.581 | −2.219 |
| Qwen3-30B offload, RTX A2000 (TC3) | 10.367 | 10.463 | **+0.096** |
| Qwen3-30B offload, RTX 4090 (TC3) | 12.184 | 11.881 | −0.303 |

One row is under its estimate: the A2000 offload, by 0.096 GB. On 2026-10-04 three rows were under, by up to 1.45 GB.

## IN-SAMPLE: loggetta's own training receipts (plan total vs driver peak)

18 receipts under `evidence/`: 15 on the RTX A2000 and FP1's 3 on the RTX 5090 (`lg-insample-now.json`).

- **Plan ≥ driver peak in all 18.** The margin is 0.18–1.99 GB.

## IN-SAMPLE: DQ7's 13 SEEN arms and DQ9's 16 reads (dense)

Today's dense plans (`dq7-insample-now.json`, `dq9-insample-now.json`).

| | allocator estimate ≥ allocated peak | plan total ≥ driver peak, resident | plan total ≥ driver peak, streamed |
|---|---|---|---|
| DQ7 (13 arms) | 13 / 13 | 6 / 6 | **0 / 7**: under by 0.25–1.14 GB |
| DQ9 (16 reads) | 16 / 16 | 6 / 6 | **0 / 10**: under by 0.37–1.14 GB |

**Every streamed dense arm's driver peak is above today's plan total.** The allocator estimate is not the short line, since it covers
every arm. What separates them is the streamed arms' reserved-minus-allocated and context. The planner's streamed admission margin,
`max(2.4 GB, 20 % of the plan)` (#26), is a refusal margin outside the plan total, and it covers each of these. A first-time user is
still shown a total that is lower than what the driver will report.

## IN-SAMPLE: DQ4's brackets

`tests/test_dense_plans.py::test_the_estimate_brackets_dq4s_measured_peaks_at_qwen3_32b` passes at today's code. That is 8 cases:
resident and streamed at 2,048, 4,096, 7,168 and 14,336 tokens. Each estimate lies between DQ4's measured line and +4 % resident,
+10 % streamed. The test encodes DQ4's fitted line (intercept + 1.2557 GiB per 1,024 tokens), not each raw rung.

## IN-SAMPLE: HO1's setups (plan total vs driver peak, GiB)

HO1's rows were re-planned with `bench/ho1_replan.py` at today's code. The 14 RTX 5090 primary setups:

| outcome | 2026-10-10 | today |
|---|---|---|
| UNDER: sampled driver peak above the plan | 1 | **2** |
| estimate short: allocated peak above the allocator estimate | 5 | 5 |
| false refusals | 5 | 3 |

- **Both UNDER setups are Qwen3-30B-A3B at 4096 × 1 with the fused kernel:** bf16 at 27.07 vs a 25.99 plan (1.042), and fp32 at 31.33 vs
  29.50 (1.062).
- Today's plan prices 4096 × 1 the same as 2048 × 2, which have the same tokens per micro-batch. HO1's observation still stands: the
  estimate has a term that does not scale with `seq`.
- **Runtime drift, stated plainly:** these rows ran experts4bit-qlora 0.40.0–0.48.0. Today's plan prices 0.56's defaults, for example
  the double-quantized absmax, which old runs did not use. Part of each residual here is the runtime moving, not the plan.

HO1's post-hoc row set leaves out its arms that padded the LoRA delta without buckets, which leaves 9 RTX 5090 setups. It was
planned twice: at today's defaults, and with the estimate told the old runs' own settings (`E4B_ABSMAX_DQ=0`,
`E4B_TRAIN_FUSE_QKV=0`), the two switches the estimate reads from the environment:

| post-hoc, 9 RTX 5090 setups | today's defaults | the runs' own settings |
|---|---|---|
| UNDER | 2 | **0** |
| estimate short | 4 | 3 |
| false refusals | 3 | 5 |

**So both UNDER setups come from the runtime moving, not from the plan.** Priced for the fp32 absmax and the unfused q/k/v those runs
held, today's plan covers every post-hoc setup. The estimate is still short on 3 fused Qwen3-30B setups, and the reserve absorbs it.

## Not in this table

- **Serving receipts (SV1–SV7, P109, the A2000 serves).** Today's serve backend no longer accepts some setup fields those receipts
  fixed (`torch_threads`), and a fixed old setup matches no current candidate. Re-planning them needs `bench/replan_serve.py`'s
  mapping, not a forced fit. That is the next version of this table.
- **The held-out row.** It belongs to the split's held-out side and is read only by the band.

## IN-SAMPLE: reproduce

At loggetta `90a7697` and experts4bit-qlora `46647fc3` on the path, with the model configs cached (Hub offline), on CPU:

```
python bench/validate_register.py --out register-insample-now.json
python bench/insample_replan.py receipts --out lg-insample-now.json
python bench/insample_replan.py dense --reads 'DQ7_READS/read-*.json' --configs E4B/bench/dq7/configs \
    --skip qwen3_14b-device-4096 --out dq7-insample-now.json
python bench/insample_replan.py dense --reads 'DQ9_READS/read-*.json' --configs E4B/bench/dq7/configs --out dq9-insample-now.json
python bench/ho1_replan.py --e4b E4B --rows evidence/2026-10-10-heldout-cards/rows.json --out . --name ho1-insample-now
python bench/ho1_replan.py --e4b E4B --rows evidence/2026-10-10-heldout-cards/rows-posthoc.json --out . \
    --primary-field primary_posthoc --name ho1-posthoc-insample-now
E4B_ABSMAX_DQ=0 E4B_TRAIN_FUSE_QKV=0 python bench/ho1_replan.py … --name ho1-posthoc-insample-oldrt
python -m pytest tests/test_dense_plans.py -k brackets_dq4
```

- `DQ7_READS` holds DQ7's 14 reads: experts4bit-qlora's committed reads for box `dq7-5090-1`, under its `bench/dq7` receipt
  directory, byte-identical to the store copies.
- `DQ9_READS` holds DQ9's 16 reads from the private receipt store, each matching `bench/dq10/dq9-source-sha256.json`.
- `bench/ho1_replan.py` needs Python 3.11.4 or later, for `tarfile`'s `filter=`.
