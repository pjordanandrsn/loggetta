# HO1: Loggetta's training plans on cards it was never calibrated on

This registers a read of committed measurements. It runs nothing on a GPU and changes no estimate, default or gate.
The row set is fixed by `bench/ho1_rows.py` and committed as `evidence/2026-10-10-heldout-cards/rows.json`. That file
holds no plan. No plan for any row exists when this is registered.

## Why

Every plan-vs-driver row committed in this repository is on the RTX A2000
(`evidence/2026-10-09-moe-plan-vs-driver-no-borrow`). HO1 asks whether the released planner covers the memory of training
runs on other cards when it has no receipt from them.

## Rows

All rows come from experts4bit-qlora at `d6d27ef5cb1509a576ec78c1e5e1a71987429b3b`, under `bench/h2h-2026-10-02` and
`bench/p67`. An arm is a row when it meets all of these:
- framework `e4b`;
- GPU an RTX 4090, RTX 5090 or H100;
- status `ok`;
- it records `peak_vram_gb`, model, `seq`, `micro_batch`, `arm`, `attn_4bit` and `offload`.

There are 509 such rows. A row is in the **primary set** unless one of these holds:
- its arm is not `fused` or `reference` (the two kernels the planner prices);
- it set `gnf4_train_gemm` to anything other than `fused`;
- it offloaded checkpoints (not priced);
- it set `E4B_CHUNKED_LM_LOSS`;
- it set any `*_env` override in its A/B records;
- it used `absmax_dq` (not a planner setup field);
- its adapter dtype is unrecorded;
- it has no nvidia-smi sidecar.

The primary set is 219 rows: 215 on an RTX 5090, 4 on an H100 NVL, 0 on an RTX 4090. The 4090's 2 rows have no
sidecar, so **HO1 is in practice a 5090 read**, with 4 H100 rows reported but not graded.

Each row's setup maps as follows:
- `fused` → `expert_kernel=grouped_nf4`, `reference` → `reference`;
- `offload` → `expert_residency` (`host` or `device`);
- `attn_4bit` as recorded;
- `adapter_dtype` `fp32` → fp32, `native` → bf16.

The workload is `seq_len`, `micro_batch` and `grad_accum` from the arm, with the optimizer as recorded (`adamw_8bit`
where the receipt says so).

## Peaks

- **Driver peak, the primary measure:** the maximum `memory.used` in the arm's 1 Hz nvidia-smi sidecar
  (`vram_<arm>.txt`). It is device-wide on a single-tenant rented box.
  - **It is a lower bound.** A 1 Hz sample can miss a transient spike, so the true peak can be higher. That only ever
    turns a true UNDER into COVERED, the false-accept direction.
  - **It is read asymmetrically:** an UNDER against it is conclusive, and a COVERED means only "not shown under".
  - The result states this bias in one plain line.
- **Allocated peak, secondary:** `peak_vram_gb`, which is `torch.cuda.max_memory_allocated()` in decimal GB. It is
  exact.

## Plans

Each row is planned with Loggetta 0.5.0 and experts4bit-qlora 0.52.0 (both from PyPI), transformers 5.18.0, on CPU.
Configs come from the Hugging Face Hub, without weights. The card is the stated GPU, its memory and the receipt's
driver; the host is the receipt's host memory. The setup is fixed as above.

**No receipts are on file** (`observations=()`). That is what a user with a fresh install gets, and no 5090, 4090 or
H100 receipt can reach the plan. As a reported variant, the plan is repeated with this repository's committed A2000
training receipts on file. By the lookup rules, that should change no 5090/H100 reserve (no receipt for these cards,
no anchor pair); any difference is reported.

## Rules (bytes)

The unit graded is a **distinct setup**: (card, model, tokens = `seq` × `micro_batch`, `expert_kernel`,
`expert_residency`, adapter dtype), with `attn_4bit` recorded alongside it. Repeated arms of one setup get one plan.
A setup is UNDER when any of its arms is. The primary set has 16 distinct setups: 14 on the RTX 5090 and 2 on the
H100. The per-arm table is detail. Every rate in the headline is a rate over setups.

- **Covered (lower-bound read):** plan device total (allocator estimate + reserve + CUDA context) ≥ every arm's
  sampled driver peak. One byte short is UNDER, and conclusive. Otherwise the setup is "not shown under".
  - When the fixed setup is refused, the refused candidate's device total is still graded, and the refusal is recorded
    separately.
- **Estimate held (exact):** allocator estimate (the plan's device total less its reserve and context lines) ≥ every
  arm's allocated peak. Both quantities are exact, so this is the exact-quantity check beside the sampled one.
- **False refusal:** the plan says the fixed setup does not fit, for a run that completed. It is its own count, not
  covered.

Results are reported per setup and per arm, with counts per card, model and e4b release of the run. Every driver/plan
and allocated/estimate ratio is published. The verdict is the primary 5090 setups:
- **HO1_UNDER** if any setup is UNDER against its sampled driver peak, with the setups named;
- **HO1_NOT_SHOWN_UNDER** otherwise.

The exact estimate check is reported beside the verdict with its own count. False refusals are a separate finding and
do not change the verdict.

## Prediction (before any plan)

With no receipts, the reserve is the 20% default and the context 0.5 GiB. If e4b's estimate is close to the allocated
peak, as on the A2000, then driver/plan ≈ (allocated + ~1.2 GiB) / (1.2 × allocated + 0.5 GiB). That gives about
0.86 at Qwen3-30B's ~24 GiB and about 0.92 at OLMoE's ~7 GiB. Expected:
- **No primary setup UNDER.** Driver/plan between 0.80 and 0.95.
- **False refusals on the largest 5090 setups.** With the 20% default reserve and 0.5 GiB context, an estimate above
  about 24.75 GiB passes what a 32 GB card can give after headroom, though the run fit. By their recorded allocated
  peaks:
  - **Expected:** Qwen3-30B-A3B 4096×1 grouped_nf4, fp32 (up to 30.33 GiB) and bf16 (25.12); Mixtral-8x7B 2048×2,
    grouped_nf4 (29.11) and reference (28.23).
  - **Borderline:** Qwen3-30B-A3B 2048×2 fp32, grouped_nf4 (up to 27.15 GiB) and reference (25.28).
  - **Not expected:** every other 5090 setup, and both H100 setups (94 GB card).
- **Estimate shortfalls, if any, on runs of older e4b releases.** Many rows ran e4b 0.40–0.48 and are planned at
  0.52.0.

## What this does not hold out

- **Seen before:** the project's own estimator in experts4bit-qlora was developed with these runs on the record, and
  its maintainers have seen them. HO1 holds out Loggetta's receipt-learned terms (reserve, context, host baseline)
  and tests its stated defaults on cards no receipt informed. It does not hold out every design decision.
- **Defaults:** the 20% reserve and 0.5 GiB context defaults date from 2026-10-04, when one Qwen3-30B 5090 run (FP1)
  had been read. FP1 is not a row here: it is a Loggetta receipt, not an e4b arm.
- **Code versions:** the run's code (e4b 0.40–0.48) is older than the plan's (0.52.0). Each row records both.
- **Scope:** one model family dominates. 191 of the 215 primary 5090 rows are Qwen3-30B-A3B.

## Prior art

None of these estimators reports an under-estimate rate on GPUs held out from its calibration:
- **DeepSpeed's ZeRO estimators:** model states only, with no stated error.
- **Accelerate `estimate-memory` / the model-memory-usage Space:** load size "within a few %", shown on one model.
  Its training and inference factors are rules of thumb.
- **vLLM:** profiles rather than predicts, with a fixed 150 MiB buffer for profiling's under-estimate.
- **DNNMem (ESEC/FSE 2020):** 14–16% mean relative error, 5 models, one P40.
- **LLMem (IJCAI 2024):** 1.6% single-GPU and 3.0% multi-GPU, on V100s.
- **xMem (Middleware 2025):** about 4% median relative error on transformers. It ran on an RTX 3060, RTX 4060 and A100,
  with 3 models on the A100.
