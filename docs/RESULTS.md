# Results of the first slice and serving v1 (2026-10-04)

**Setup.**
- Hardware: one RTX A2000 12 GB (sm_86, driver 575.64.05, PCIe link reported at x8). The host is a container
  with 2 CPUs and a 28 GiB cgroup, which it shares with other sessions' jobs, so host-bound timings are noisy.
- Software: torch 2.8.0+cu128, triton 3.4.0, transformers 5.18.0, bitsandbytes 0.50.2. The experts4bit-qlora and
  grouped-nf4-gemm commits are per receipt; see Provenance.
- Workload: QLoRA (r 8, α 16, bf16 adapters, attention + expert LoRA, AdamW 2e-4) on alpaca packed into fixed
  blocks of seq 512 × micro-batch 2 (1,024 tokens per forward), 12 steps. Step time is the median of steps 3–12.

Every number below is generated from receipts by `bench/summarize_receipts.py`, except where a row says otherwise.
Receipts are kept in the private workspace (`runs/receipts/`).

## 1. One real workload through the planner

`python -m loggetta train allenai/OLMoE-1B-7B-0924 --seq 512 --micro-batch 2 --steps 12`:

1. took the hardware inventory;
2. described OLMoE from its config (16/16 MoE layers, E=64, H=2048, I=1024, top-8; 6.44B expert + 0.48B dense
   params);
3. priced 8 candidate setups and noticed 1.34 GiB of the card held by another process;
4. chose experts resident on the device, the grouped NF4 kernel (training route `fused`: sm_86 is not 9.0) and
   bf16 attention;
5. ran the selected setup through `experts4bit_qlora.prepare_qlora_training`;
6. wrote a receipt.

Correctness was checked, not assumed:
- **Every run:**
  - losses finite and falling;
  - sha256 of the frozen packed expert bytes, absmax included, identical before and after training (for
    offloaded runs, the host homes are hashed);
  - LoRA-B norm moved off zero.
- **Orchestration adds no numerical change.**
  - Step-1 loss is **bitwise identical** (1.858969) in the planned run, its repeat, and a run built by hand from
    experts4bit-qlora's public calls with no planner (`bench/direct_baseline.py`).
  - Over 12 steps, planned and hand-built differ by at most **0.0042**. Two identical planned runs differ by
    **0.0068**: the backward is not deterministic run to run, and the planner adds nothing above that.
- **Memory is identical to the hand-built path.** Allocator peak 5.4710 vs 5.4706 GiB; reserved 6.6836 GiB both;
  driver-reported 6.8164 GiB both.
- **Same trainable set.** 60,817,408 parameters and 16 fused modules in both.
- **Kernel parity.** The reference loop (R4, selected through `--fix expert_kernel=reference`) ends at 1.2271
  against 1.2203–1.2242 for the fused runs.

## 2. Speed (measured; nothing here was predicted)

| setup | s/step | relative |
|---|---|---|
| resident, grouped NF4 (R1, R1b, direct R2) | 2.35, 2.13, 2.16 | 1 (seat noise ±10%) |
| host-resident experts, grouped NF4 (R3, R3b) | 3.13, 2.79 | ≈1.3× slower |
| resident, reference per-expert loop (R4) | 6.96 | ≈3.1× slower |

The planner ordered these setups the same way from evidence measured elsewhere (claim IDs in each plan). It
predicted no times. The one number it can derive for host residency is a **lower bound**:
- 7.25 GB host-to-device per step over the measured 6.24 GB/s ⇒ **≥ 1.16 s/step**;
- measured 3.13 and 2.79 s: the bound holds and, as a bound, is loose (transfer overlaps compute).

## 3. Memory: estimate vs measurement

| run | allocator est / meas | driver est / meas | host est / meas |
|---|---|---|---|
| R1 resident (inferred overheads: no receipts yet) | 5.26 / 5.47 | 5.76 / 6.82 | 3.00 / 8.61 (VmHWM: mapped checkpoint pages) |
| R1b resident (overheads learned from R1) | 5.26 / 5.47 | 6.56 / 6.82 | 0.72 / 1.42 (anon) |
| R3 host offload | 2.09 / 2.31 | 2.69 / 3.33 | 4.10 / 1.42 (anon only: pinned memory missed, see below) |
| R3b host offload (fixed host metric, matching-setup slack) | 2.09 / 2.31 | 3.04 / 3.33 | 4.04 / 5.05 (anon + pinned) |
| R4 resident, reference loop | 5.26 / 5.27 | 7.42 / 6.28 (slack from a non-matching setup; fixed after) | 0.70 / 1.50 |
| R5 granite-3.1-3b (second family) | 3.04 / 3.09 | 4.34 / 3.93 | 0.70 / 1.86 |
| R6 granite-4.0-h-tiny (Mamba hybrid) | 6.62 / 6.69 | 8.26 / 7.41 | 0.70 / 2.79 |

All values GiB.

- **The allocator column is the estimator's own accuracy:** +0.01 to +0.21 GiB across six runs, three families,
  resident and offload.
- **The driver column adds the planner's learned overheads,** and it is only as good as the receipts available
  when the plan was made. R4 and R6 over-estimate it: R4 borrowed host-run slack before the matching-setup fix, and
  R6's granite slack (8.3%) is much lower than OLMoE's (22%).
- **The host column under-estimates when the baseline is borrowed from another model's receipt.** The load
  transient is listed as not modelled.

**What the measurements taught the planner during the session.** Each item changed code, and each change is a
commit:
1. **Allocator reserve slack.**
   - Reserved minus allocated is 22% of the allocator peak resident and 39% under offload (per-layer staging). It
     is invisible to `max_memory_allocated`, but the GPU pays it, and the 0.5 GiB headroom policy would not have
     covered it on a full card.
   - It is now a plan line, taken from the receipt with the same residency and kernel.
2. **CUDA context** measured 0.13 GiB on this stack (the default had been 0.5 GiB).
3. **Host requirement.**
   - VmHWM counts memory-mapped checkpoint shards (12.7 GiB of file pages, reclaimable).
   - Pinned memory is accounted as `RssShmem`: measured, a 512 MiB pinned tensor added 511 MiB there and nothing to
     anonymous RSS.
   - The host requirement is therefore anonymous + shared RSS.
4. **Observations** are ordered by run time, not file name.

**Against the register** (`bench/validate_register.py`: no GPU, configs only). Six recorded allocator peaks
across OLMoE, Granite and Qwen3-30B on an RTX 5090, a 4090 and the A2000. T is each receipt's largest micro-batch
(TC3 receipts) or assumed from the same token file (tp4, whose receipts are not in the repository).

| claim | estimate GB | measured GB | residual |
|---|---|---|---|
| OLMoE fused_attn4, 5090 (T assumed) | 6.03 | 6.42 | +0.39 |
| OLMoE reference_attn4, 5090 (T assumed) | 6.03 | 5.87 | −0.16 |
| Granite fused_attn4, 5090 (T assumed) | 3.61 | 3.78 | +0.17 |
| Qwen3-30B fused_attn4 resident, 5090 (T assumed) | 26.40 | 24.58 | −1.81 |
| Qwen3-30B offload, A2000 (T from receipt) | 9.48 | 10.46 | +0.99 |
| Qwen3-30B offload, 4090 (T from receipt) | 10.43 | 11.88 | +1.45 |

The derived part (weights, adapters, gradients, optimizer) is exact by construction. The error lives in the
heuristic activation line and in what is listed as not modelled (the fused-path offload staging transient
accounts for the sign of the two offload rows). **This is an estimator with a stated error, about ±2 GB at 30B
scale, not a precise one;** receipts are how it gets better.

## 4. Model families

**Planned with no per-family code anywhere in the planner.** Final sweep: `bench/family_sweep.py`, the seat's
hardware profile, every receipt above as observations, QLoRA at seq 2048 × micro-batch 1, experts4bit-qlora over
#1048.

| model (model_type) | outcome on the A2000 (10.65 GiB free, 21.9 GiB host available) |
|---|---|
| OLMoE-1B-7B (olmoe) | feasible: resident, grouped NF4, 6.45 GiB; **run** (R1–R4) |
| Qwen3-30B-A3B (qwen3_moe) | refused by 0.58 GiB of device memory with host-resident pageable experts. Suggests **"reduce tokens per micro-batch to 1024"**, which agrees with the register's A2000 run at 555 tokens |
| Qwen3.6-35B-A3B (qwen3_5_moe; 10 of 40 layers attention) | refused: device short by 8.2 GiB (248k-entry vocabulary: embeddings + logits). Suggests 256 tokens |
| granite-3.1-3b-a800m (granitemoe, pre-fused legacy spelling) | feasible, resident, 4.06 GiB; **run** (R5) |
| granite-4.0-h-tiny (granitemoehybrid: Mamba + 4 attention layers) | feasible, resident, 8.50 GiB; non-attention mixers listed as not modelled; **run** (R6): allocator estimate +0.07 GiB off |
| LFM2-8B-A1B (lfm2_moe) | feasible, resident. Before #1048 the detector found no attention and the planner turned attention LoRA off with a reason; after #1048 (out_proj attention support) it trains attention, with no planner change |
| Mixtral-8x7B (mixtral) | refused: host short by 2.5 GiB even with pageable homes |
| ERNIE-4.5-21B-A3B (ernie4_5_moe, admitted by convention) | feasible: host-resident pinned experts, 12.1 GiB host |
| DeepSeek-V2-Lite (deepseek_v2, MLA) | feasible with attention LoRA off: the detector cannot describe MLA, said in the plan |
| gpt-oss-20b (gpt_oss) | refused by structure: biased experts cannot take ExpertsLoRA |

**What the sweep forced.** Fixes made during the session:
- Two refusals were added in experts4bit-qlora: attention that cannot be described (MLA) and attention with no
  wrappable projections (LFM2). Without them, DeepSeek-V2 would have crashed in `add_attention_lora`, and LFM2
  would have "trained attention" with zero adapters.
- A not-modelled note was added for non-attention mixers.
- A `pin` axis was added to the planner.
- An unmeasured setup takes the largest reserve slack measured on the GPU (conservative). Borrowing a smaller
  figure had made the sweep pick Qwen3-30B's reference-kernel candidate only because it had not been measured.

None of these is a family-name branch.

## 5. FP1: a measured 30B receipt replaces the extrapolated overheads

**The run.** Rented through the shared launcher: experts4bit-qlora lane FP1, work item #1064, run `fp1-5090-1`.
- One RTX 5090, driver 595.84.
- **$0.848**, teardown complete; the receipt is in the private store at `a4866167`.
- Three arms, seq 512 × micro-batch 2, 12 steps, QLoRASetup defaults.
- Raw arm receipts: `evidence/2026-10-04-fp1-rtx5090/raw/`. Planner observations from `bench/import_fp1.py`.

| arm | estimate (allocator) | measured | reserve slack | context | driver peak | s/step |
|---|---|---|---|---|---|---|
| OLMoE-1B-7B resident | 5.26 | 5.47 (+4.0%) | 15.0% | 0.62 | 6.90 | 0.22 |
| Qwen3-30B-A3B resident | 22.09 | 21.91 (−0.8%) | 8.3% | 0.62 | 24.34 | 0.70 |
| Qwen3-30B-A3B host-offload | 7.22 | 6.71 (−7.1%) | 31.2% | 0.62 | 9.41 | 2.00 |

All values GiB. Integrity is clean on every arm.

**What it changed.**
- **The estimator holds at 30B:** within 1% resident, and −7% under offload.
- **Reserve slack is not one number.**
  - It depends on model size: 15.0% for OLMoE against 8.3% for Qwen3-30B on the same card.
  - It depends on the GPU: OLMoE measured 22.2% on the A2000 and 15.0% on the 5090.
  - The planner now looks it up in this order: this GPU + setup + model; else the same model and setup measured on
    another GPU, scaled by an anchor model measured on both (labelled heuristic, arithmetic in the line); else this
    GPU + setup with another model; else the largest measured on this GPU (conservative).
- **The CUDA context is stack-specific:** 0.62 GiB on the 5090 / 595.84 against 0.13 GiB on the A2000 / 575.

**Before/after, Qwen3-30B-A3B** (`bench/replan_with_fp1.py`, `evidence/fp1-replan.json`):

| plan | device total before | after | measured (driver peak) |
|---|---|---|---|
| RTX 5090, resident | 27.01 (inferred 20% slack, 0.5 context) | **24.54** (measured 8.3%, 0.62) | **24.34** |
| RTX 5090, host-offload | 9.16; transfer floor 1.04 s (PCIe ceiling) | 10.08; floor **1.58 s** (measured 20.8 GB/s) | 9.41; step **2.00 s** |
| RTX A2000, resident | 24.08 (Granite-4's 8.3%, same GPU) | 24.96 (Qwen3's 8.3% × anchor 0.222/0.150 = 12.3%) | not run: refused both times, the card has 12 GB |

The 5090 resident prediction moved from 2.7 GiB high to 0.2 GiB of the measured process peak. (Rerun after section
6's tie rule, the A2000 "before" reads 25.93: the largest same-setup slack on that card, OLMoE's 22.2%, instead of
the first one found.) On the A2000 the
30B slack is now an explicit transfer rather than another model's figure. It is labelled heuristic, because no A2000
run of Qwen3 exists.

## 6. Serving: the paged server's all-VRAM placement

**What is planned.** `kind="serve"` with context × concurrency. The plan prices `serve_paged.build_engine` under
`E4B_PAGED_PLACEMENT=all-vram` via e4b's `estimate_serve_footprint` (e4b#1080). Its items:
- expert stacks and bf16 dense weights, derived;
- the FP8 paged KV pool, from `Fp8PagedKV`'s own arithmetic and tested byte-equal to a constructed pool;
- a heuristic working set.

The plan's "Why" carries the `E4B_PAGED_*` environment that builds the priced setup. Serve plans are planned only
(not executed). The solver's tiers are refused in words.

**Checked against receipts that already existed.** Lane P109 ran `serve_paged` all-VRAM on one RTX 5090:
- Qwen3-30B-A3B, 16 sequences × 4096 tokens, NF4 experts, no int4;
- six arms, eager and graphed decode (`e4b/bench/p109/receipts/p109-5090-2`, imported by `bench/import_p109.py`).

| arms | allocator estimate | measured allocator peak | reserve slack | context (end of run) |
|---|---|---|---|---|
| eager decode (D1, E1, E2) | 21.35 | 21.51 (−0.7%) | 0.13% | 0.59 |
| decode graphs (G1, G2) | 21.36 | 21.56 (−0.9%) | 0.85% | 0.71 |
| decode graphs + prefill graph (P1) | 21.36 | 21.51 (−0.7%) | 0.06% | 0.59 |

All values GiB. The estimate does not price the prefill-graph setting; P1 is shown for completeness. P109 read memory
after load and after the runs rather than sampling it, so its reserve is an end-of-run lower bound.

**What it changed.**
- **A server's slack is not a trainer's:** 0.06–0.85% here, against 8–39% for training. The planner now matches
  slack by workload kind. Among same-setup receipts it prefers one with the candidate's whole setup, and of
  several it takes the largest. With no serve receipt for a GPU it uses the stated 20% default.

| plan (`bench/replan_with_p109.py`) | device total, before | after | measured (reserved + context) |
|---|---|---|---|
| eager decode | 26.24 (default 20% = 4.27) | **21.99** (P109's 0.13%) | 22.12 |
| decode graphs | 26.25 | **22.16** (P109's 0.85%) | 22.11–22.45 |

- **The prefill graph is planned off.** experts4bit-qlora 0.47.0 made `E4B_PAGED_PREFILL_GRAPH=auto` the server's
  default. Its private pool is +3.3 GiB at 30B, the estimate does not price it, and `auto` checks only the
  device's free memory, not the plan's budget. A plan therefore sets it to `0` and says why; a caller can fix
  `prefill_graph=auto` (e4b#1098 names the pool as not modelled).
- **Independence.** The allocator estimate is independent of P109. The context line in both columns is FP1's
  training receipt on the same card (0.61 GiB, a different driver). The "after" reserve is P109's own, so that line
  agrees by construction. Before P109, the default over-reserved 4.1 GiB at 30B: a 5090 plan would have refused
  setups that fit.

**Run on the seat: a planned serve, measured** (`bench/serve_validate.py`). The script:
- plans the workload;
- builds `serve_paged`'s engine in-process with exactly the plan's environment (`ServeSetup.to_env`);
- decodes 4 × (1024 prompt + 64 new) tokens;
- writes an `execution-receipt/1` of kind serve (`evidence/2026-10-05-rtx-a2000-serve/`).

Model and setup: OLMoE-1B-7B on the RTX A2000, all-VRAM, 4 × 4096. The arena was baked locally (grouped-nf4-gemm
`bake_nf4`, 3.6 GB). The calibration blob is P39's: the solver it feeds is overridden by all-VRAM.

- **The first attempt crashed, and that is a finding.** The plan chose decode graphs, and `serve_paged`'s
  default agrees. Graphs need grouped-nf4-gemm's fused FP8 KV append, whose e4m3 cast (`tl.float8e4nv`) Triton
  compiles only on sm_89+. On sm_86 the first graphed decode step died in Triton's compiler
  (`graphs-sm86-crash.log.txt`). Every Ampere card took that path by default.
  - e4b fix: branch `fix/fused-kv-append-sm89`. Below sm_89, `auto` decodes eagerly, the fused append degrades,
    and an explicit `=1` is refused in words.
  - Planner: decode graphs are offered only where e4b's `fused_append_unsupported(capability)` says they run.
    With an e4b that cannot say, graphs are never planned.
- **Eager decode, measured:**

| | estimate | measured |
|---|---|---|
| allocator (stacks + dense + KV + working set) | 5.40 | 5.57 peak (**−3.1%**) |
| reserve slack | 20% default (1.08) | **1.46%** |
| CUDA context | 0.17 (a training receipt) | 0.12 |
| device total | 6.64 before → **5.64** after this receipt (`bench/replan_serve.py`) | 5.78 driver peak |

All values GiB. Decode throughput was 4.9 tokens/s, eager, on a seat at load 20–40. It is recorded, not claimed.
- **One pattern, two points.** The allocator estimate missed by about the same absolute amount on both models:
  0.16–0.21 GiB at 30B and 0.17 GiB at 1B. That points to a roughly fixed unmodelled term (workspaces, prefill
  temporaries) rather than a proportional one, so the small model's miss is larger in percent. Two points make it
  a hypothesis, and nothing is fitted to it.

**Plan-only sweep** (`bench/serve_plan_sweep.py`, `evidence/serve-plan-sweep.json`; budget = the card's total,
default slack):

| model, card | weights | fits | refused, with the planner's suggestions |
|---|---|---|---|
| OLMoE-1B-7B, RTX A2000 | 4.26 | up to 32k KV tokens (2048×16, 8192×4, 32768×1) | 8192×16 → "context 5744 at 16" or "concurrency 8 at 8192"; 32768×4 → "context 22976 at 4" or "concurrency 2" |
| Qwen3-30B-A3B, RTX 5090 | 18.06 | up to 131k KV tokens (8192×16, 32768×4) | 32768×16 → "context 5648 at 16" or "concurrency 2 at 32768" |

Every suggestion is itself planned. The tests check that each one is feasible, and that the context suggestion is the
largest at 16-token-block granularity. The A2000 sweep ran before any A2000 serve receipt, so it uses the 20% default. The figures in the
"fits" column carry that over-reserve.

## 6b. Serving with tiers: VRAM, DRAM and NVMe (the solver's placement)

**What is planned.** When all-VRAM does not fit, a serve plan uses `serve_paged`'s solver placement. e4b#1115 prices
it with `solve_placement` itself: the server passes no routing profile, so VRAM fills first, then DRAM, then NVMe.
The planner sizes the two tier budgets as policy. The VRAM tier takes the largest value the device budget allows,
then the DRAM tier the largest the host budget allows less a host headroom, each found by search over the priced
estimate. The rest of the experts stream from NVMe.

Serve plans also learn two lines from serve receipts of the same model:
- the **allocator residual**: a receipt's measured peak minus today's estimate of its own setup;
- **host growth while serving**: anonymous host memory gained after load.

Allocator slack is matched by placement. The backend names the fields that separate slack regimes; measured, it
is 15% under the solver against 1–2% at all-VRAM.

**Calibration runs** (OLMoE-1B-7B on the A2000, solver budgets set by hand, `evidence/2026-10-05-rtx-a2000-serve/`):

| VRAM / DRAM GiB × hot_rows | tiers (V/D/N) | allocator: e4b estimate vs peak | host shared: items vs peak | anonymous growth while serving |
|---|---|---|---|---|
| 1.2 / 1.5 × 64 | 364 / 455 / 205 | 2.176 vs 2.355 | 0.46–0.67 vs 0.50 | 0.73 |
| 2.0 / 0.8 × 128 | 606 / 242 / 176 | 2.974 vs 3.144 | 1.13 vs 1.21 | 0.72 |

Each run replanned with the other runs' receipts only (leave-one-out, `bench/replan_serve.py`):

| run | device: plan vs driver peak | allocator: plan vs peak | host: plan vs required peak |
|---|---|---|---|
| 1.2 / 1.5 | 2.82 vs 2.83 | 2.351 vs 2.355 | 3.60 vs 3.30 |
| 2.0 / 0.8 | 3.76 vs 3.70 | 3.153 vs 3.144 | 3.37 vs 3.29 |

**End to end: the planner chose the tiers.** OLMoE, 3.0 GiB device budget, 4.5 GiB host budget,
`receipt planned-tiers-olmoe-v3.0-r4.5`:

| | plan | measured |
|---|---|---|
| tiers (V/D/N) | 272 / 421 / 331 | 272 / 421 / 331 (the server's own manifest) |
| allocator | 2.052 | 2.051 |
| device total | 2.50 of 3.00 budget | 2.35 driver peak |
| host total | 3.50 of 4.50 budget | 3.17 required peak |

All values GiB. Both totals err high, the safe direction. The learned 15% slack came from another run; this one
used 8%. Decode throughput was 0.2 tokens/s, with NVMe plus CPU tiers on a 2-core seat; it is recorded, not claimed.

**The residual.** Every serve run so far missed the e4b estimate by 0.15–0.21 GiB: two models, two cards, both
placements, P109 included. The planner charges the largest measured for the model being planned, and no prior for
other models. Attributing it (allocator snapshot) is the obvious next measurement.

## 7. Not measured, said plainly

- **No performance model.** Speed is ordered from evidence, never predicted, apart from the transfer lower bound.
- **The activation heuristic** is a formula, checked against nine allocator peaks (three here, six in the
  register), not derived.
- **Qwen3-30B ran once, on a rented RTX 5090 (FP1).** Every other model above 8B was planned, not run.
- **Serving** is planned at both placements. All-VRAM is checked on two model/card pairs. The solver's tiers are
  checked on one model and card (OLMoE, A2000) and assume uniform routing, as the server does. int4 stores, a
  measured routing profile and decode speed are not planned; see `SERVING-PRESSURE-TEST.md`.

## Provenance

- Every receipt records the git commit and dirty flag of each source tree, distribution versions, argv and the
  whole plan.
- **Two provenance defects, found from the receipts themselves and fixed.**
  1. R1's planner source was recorded against the enclosing `/home/node/work` repository. The planner was not yet
     a git repository, and `git_commit` did not check that the file was tracked. Fixed before R1b.
  2. R1–R5 and R3b read commits when `execute()` began, minutes after the process had imported the code. I was
     committing planner changes during the queue, so a receipt could name newer code than the code that ran. R5
     provably does: its receipt names planner `fd76c308`, but its plan applies the host run's 38.6% reserve slack
     to a resident setup, which is the pre-`fd76c308` lookup.
  - Fixed in `30b63b5`: provenance is taken at process start, before the measured code is imported, and
    `changed_during_run` lists any tree that moved.
  - For R1–R5 and R3b, the experts4bit-qlora and grouped-nf4-gemm commits are reliable: their branches only
    rebased; the code did not change. The planner commit is approximate.
- experts4bit-qlora's base moved during the session while main was merging, including #1048. R1–R5 ran on the
  branch over `2631d3d7` (`fa324db3`, or `65ddf4a6` for R2's same code over `5c74564e`); R3b onward over
  `520b0b5d` (`f0af2b3d`).

## Appendix: generated receipt tables

`python bench/summarize_receipts.py runs/receipts` (R1–R6, R3b, the direct arm):

| run | setup | s/step (median, steps 3+) | tokens/s | loss step 1 -> 12 | frozen bytes unchanged | adapters moved | load s |
|---|---|---|---|---|---|---|---|
| `OLMoE-1B-7B-0924-device-grouped_nf4-t1024-20261004T173932Z` | device, grouped_nf4 | 2.35 | 436 | 1.8590 -> 1.2240 | True | True | 1893 |
| `OLMoE-1B-7B-0924-device-grouped_nf4-t1024-20261004T175201Z` | device, grouped_nf4 | 2.13 | 481 | 1.8590 -> 1.2242 | True | True | 12 |
| `OLMoE-1B-7B-0924-device-reference-t1024-20261004T175624Z` | device, reference | 6.96 | 147 | 1.8580 -> 1.2271 | True | True | 12 |
| `OLMoE-1B-7B-0924-host-grouped_nf4-t1024-20261004T175357Z` | host, grouped_nf4 | 3.13 | 327 | 1.8590 -> 1.2227 | True | True | 24 |
| `OLMoE-1B-7B-0924-host-grouped_nf4-t1024-20261004T182135Z` | host, grouped_nf4 | 2.79 | 368 | 1.8590 -> 1.2219 | True | True | 11 |
| `granite-3.1-3b-a800m-instruct-device-grouped_nf4-t1024-20261` | device, grouped_nf4 | 1.77 | 578 | 1.6382 -> 1.0636 | True | True | 164 |
| `granite-4.0-h-tiny-device-grouped_nf4-t1024-20261004T183236Z` | device, grouped_nf4 | 2.30 | 446 | 1.5906 -> 0.8898 | True | True | 492 |
| `direct-OLMoE-1B-7B-0924-device-grouped_nf4-t1024-20261004T17` | device, grouped_nf4 | 2.16 | 475 | 1.8590 -> 1.2203 | True | True | 437 |

| run | allocator est / meas (GiB) | driver est / meas (GiB) | host est / meas (GiB) | measured context, reserve slack | link H2D GB/s | overhead basis in plan |
|---|---|---|---|---|---|---|
| `OLMoE-1B-7B-0924-device-grouped_nf4-t1024-20261004T173932Z` | 5.26 / 5.47 | 5.76 / 6.82 | 3.00 / 8.61 | 0.13, 22.2% | n/a | {'CUDA': 'inferred'} |
| `OLMoE-1B-7B-0924-device-grouped_nf4-t1024-20261004T175201Z` | 5.26 / 5.47 | 6.56 / 6.82 | 0.72 / 1.42 | 0.13, 22.2% | 6.24 | {'allocator': 'measured', 'CUDA': 'measured'} |
| `OLMoE-1B-7B-0924-device-reference-t1024-20261004T175624Z` | 5.26 / 5.27 | 7.42 / 6.28 | 0.70 / 1.50 | 0.13, 16.6% | 6.22 | {'allocator': 'measured', 'CUDA': 'measured'} |
| `OLMoE-1B-7B-0924-host-grouped_nf4-t1024-20261004T175357Z` | 2.09 / 2.31 | 2.69 / 3.33 | 4.10 / 1.42 | 0.13, 38.6% | 4.73 | {'allocator': 'measured', 'CUDA': 'measured'} |
| `OLMoE-1B-7B-0924-host-grouped_nf4-t1024-20261004T182135Z` | 2.09 / 2.31 | 3.04 / 3.33 | 4.04 / 5.05 | 0.13, 38.5% | 6.21 | {'allocator': 'measured', 'CUDA': 'measured'} |
| `granite-3.1-3b-a800m-instruct-device-grouped_nf4-t1024-20261` | 3.04 / 3.09 | 4.34 / 3.93 | 0.70 / 1.86 | 0.13, 22.8% | 6.14 | {'allocator': 'measured', 'CUDA': 'measured'} |
| `granite-4.0-h-tiny-device-grouped_nf4-t1024-20261004T183236Z` | 6.62 / 6.69 | 8.26 / 7.41 | 0.70 / 2.79 | 0.17, 8.3% | 6.23 | {'allocator': 'measured', 'CUDA': 'measured'} |

| run | transfer lower bound s/step (basis) | measured s/step | bound holds |
|---|---|---|---|
| `OLMoE-1B-7B-0924-host-grouped_nf4-t1024-20261004T175357Z` | 1.16 (measured, 6.24 GB/s) | 3.13 | True |
| `OLMoE-1B-7B-0924-host-grouped_nf4-t1024-20261004T182135Z` | 1.18 (measured, 6.14 GB/s) | 2.79 | True |

Provenance per run (commit, dirty):
- `OLMoE-1B-7B-0924-device-grouped_nf4-t1024-20261004T173932Z`: {'experts4bit_qlora': ('b567fc9f25', False), 'loggetta': ('a31ad13c64', True), 'nf4_grouped': ('c420a80ed8', False)}
- `OLMoE-1B-7B-0924-device-grouped_nf4-t1024-20261004T175201Z`: {'experts4bit_qlora': ('fa324db3a2', False), 'loggetta': ('8c462f710e', False), 'nf4_grouped': ('b48f2e6740', False)}
- `OLMoE-1B-7B-0924-device-reference-t1024-20261004T175624Z`: {'experts4bit_qlora': ('fa324db3a2', False), 'loggetta': ('acd2cef167', False), 'nf4_grouped': ('b48f2e6740', False)}
- `OLMoE-1B-7B-0924-host-grouped_nf4-t1024-20261004T175357Z`: {'experts4bit_qlora': ('fa324db3a2', False), 'loggetta': ('8c462f710e', False), 'nf4_grouped': ('b48f2e6740', False)}
- `OLMoE-1B-7B-0924-host-grouped_nf4-t1024-20261004T182135Z`: {'experts4bit_qlora': ('f0af2b3df8', False), 'loggetta': ('c1a1764fcf', False), 'nf4_grouped': ('b48f2e6740', False)}
- `granite-3.1-3b-a800m-instruct-device-grouped_nf4-t1024-20261004T180058Z`: {'experts4bit_qlora': ('fa324db3a2', False), 'loggetta': ('fd76c3084b', False), 'nf4_grouped': ('b48f2e6740', False)}
- `granite-4.0-h-tiny-device-grouped_nf4-t1024-20261004T183236Z`: {'experts4bit_qlora': ('f0af2b3df8', False), 'loggetta': ('c1a1764fcf', False), 'nf4_grouped': ('b48f2e6740', False)}
- `direct-OLMoE-1B-7B-0924-device-grouped_nf4-t1024-20261004T173932Z`: {'experts4bit_qlora': ('65ddf4a6ba', False), 'loggetta': ('8c462f710e', False), 'nf4_grouped': ('c420a80ed8', False)}
