# Results of the first slice (2026-10-04)

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

## 6. FP1: a measured 30B receipt replaces the extrapolated overheads

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

The 5090 resident prediction moved from 2.7 GiB high to 0.2 GiB of the measured process peak. On the A2000 the
30B slack is now an explicit transfer rather than another model's figure. It is labelled heuristic, because no A2000
run of Qwen3 exists.

## 5. Not measured, said plainly

- **No performance model.** Speed is ordered from evidence, never predicted, apart from the transfer lower bound.
- **The activation heuristic** is a formula, checked against nine allocator peaks (three here, six in the
  register), not derived.
- **Qwen3-30B ran once, on a rented RTX 5090 (FP1).** Every other model above 8B was planned, not run.
- **Serving** is represented and refused; see `SERVING-PRESSURE-TEST.md`.

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
