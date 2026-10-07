# Results of the first slice and serving v1 (2026-10-04)

**Setup.**
- Hardware: one RTX A2000 12 GB (sm_86, driver 575.64.05, PCIe link reported at x8). The host is a container
  with 2 CPUs and a 28 GiB cgroup, which it shares with other sessions' jobs, so host-bound timings are noisy.
- Software: torch 2.8.0+cu128, triton 3.4.0, transformers 5.18.0, bitsandbytes 0.50.2. The experts4bit-qlora and
  grouped-nf4-gemm commits are per receipt; see Provenance.
- Workload: QLoRA (r 8, α 16, bf16 adapters, attention + expert LoRA, AdamW 2e-4) on alpaca packed into fixed
  blocks of seq 512 × micro-batch 2 (1,024 tokens per forward), 12 steps. Step time is the median of steps 3–12.

Each result compares an `ExecutionReceipt` with the `ExecutionPlan` that produced it, which the receipt embeds.
Every number below is generated from receipts by `bench/summarize_receipts.py`, except where a row says otherwise.
Runs write raw receipts under `runs/receipts/`; curated public evidence is committed under `evidence/`.

## 1. One real workload through the planner

`python -m loggetta train allenai/OLMoE-1B-7B-0924 --seq 512 --micro-batch 2 --steps 12`:

1. took the hardware inventory;
2. described OLMoE from its config (16/16 MoE layers, E=64, H=2048, I=1024, top-8; 6.44B expert + 0.48B dense
   params);
3. priced 8 candidate setups and noticed 1.34 GiB of the card held by another process;
4. chose experts resident on the device, the grouped NF4 kernel (training route `fused`: sm_86 is not 9.0) and
   bf16 attention;
5. handed the plan to its backend: `experts4bit_qlora.prepare_qlora_training` built the selected setup, and the
   executor ran the measured loop around it;
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
- **One pattern, two points, later explained.** The allocator estimate missed by a similar absolute amount on both
  models (0.16–0.21 GiB at 30B, 0.17 GiB at 1B). This looked like a fixed term but was not. Section 6b's attribution
  found it was prefill staging and, under the solver, the cold-row stack. Both are now priced.

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

**The residual, attributed.** `bench/serve_residual.py` rebuilt the server under the caching allocator's history
recording (Python stacks), replayed the trace, and grouped what was live at the generation peak by source line
(`evidence/2026-10-05-residual-attribution/`). OLMoE on the A2000:

| run | estimate vs peak | what the gap was |
|---|---|---|
| all-VRAM, 1 × 512, 128-token prompt | 8 MiB **over** | nothing missing |
| solver 1.2 / 1.5 GiB, same prompt | 183 MiB under | `hot_residency._cold_contrib`: one layer's routed NVMe experts streamed to the GPU (54 rows × 3.375 MiB + outputs) |
| all-VRAM, 4 × 4096, 1024-token prompts | 179 MiB under | bf16 **prefill staging**, one prompt's K/V for all 16 layers (128 MiB), plus ~50 MiB of MoE workspace above the stated heuristic |

So it was never a fixed term, and the 0.15–0.21 GiB seen everywhere was a coincidence of these runs' prompt
lengths. Both causes are now priced in e4b at their ceilings:
- the cold-row stack (#1139), at `min_hot_rows ×` row bytes;
- prefill staging (#1141), at `(max_tokens_per_seq + chunk_tokens) ×` bf16 K/V bytes per token.

The staging ceiling matters for long prompts, and the long-prompt check bears it out
(`receipt longprompt-olmoe-2x4096-p4000`: OLMoE, all-VRAM, two 4,000-token prompts):
- the allocator estimate is **5.420 GiB, with 0.56 GiB of staging**, against a **5.414 GiB** measured peak;
- without the staging item the estimate would have been 4.858 GiB, 10% under;
- serving slack was 2.1%. The planner's learned residual stays as a safety line, recomputed against today's estimate, so it shrinks
to the leftover workspace.

## 6c. Serving across the ten families (plan-only)

`bench/serve_family_sweep.py` covered the training sweep's ten families on four cards at 4096 × 1 and 8192 × 8.
- **Cards:** the seat's A2000 (probed, 21.9 GiB host), FP1's RTX 5090, and a stated RTX 4090 and RTX 3090 (24 GB,
  64 GiB host; what-ifs).
- **Where it ran:** on Necessity through the local pool (`evidence/serve-family-sweep.*`): plan-venv, torch 2.8.0+cpu,
  transformers 5.18.0, the native CPU kernels built. No family code was added.

Cells read "VRAM" for all-VRAM, else "VRAM / DRAM / NVMe" tiers in GiB:

| model | ctx × seqs | A2000 12 GB | 3090 / 4090 24 GB | 5090 32 GB |
|---|---|---|---|---|
| OLMoE-1B-7B, granite-3.1-3b, granite-4.0-h-tiny, LFM2-8B | both | VRAM | VRAM | VRAM |
| DeepSeek-V2-Lite | 4096 × 1 / 8192 × 8 | VRAM · 2.7 / 4.9 / 0 | VRAM | VRAM |
| ERNIE-4.5-21B-A3B | 4096 × 1 / 8192 × 8 | 6.7 / 4.0 / 0 · 4.8 / 5.9 / 0 | VRAM | VRAM |
| Qwen3-30B-A3B | 4096 × 1 / 8192 × 8 | 6.3 / 8.9 / 0 · 2.8 / 12.4 / 0 | VRAM · 12.5 / 2.7 / 0 | VRAM |
| Qwen3.6-35B-A3B | 4096 × 1 / 8192 × 8 | 5.0 / 11.8 / 0 · 4.3 / 12.6 / 0 | VRAM · 14.0 / 2.9 / 0 | VRAM |
| Mixtral-8x7B | 4096 × 1 / 8192 × 8 | 5.8 / 17.8 / 0 · 0.5 / 17.5 / **5.6** | 15.5 / 8.1 / 0 · 10.9 / 12.7 / 0 | VRAM · 20.9 / 2.8 / 0 |
| gpt-oss-20b | both | **refused** | VRAM | VRAM |

(Re-run after e4b#1139/#1141 priced prefill staging and the cold-row stack. Staging takes device memory from the
VRAM tier at long contexts, so the 8192 × 8 rows shifted, and Qwen3-30B at 8192 × 8 on 24 GB moved back to tiers.)

**What the sweep exposed, and what changed because of it:**
- **`hot_rows` is now planned.** At the server's default of 64, Mixtral's ~99 MB experts made the cold tier's
  pinned landing, cold view and setup tier ~20 GiB on the A2000. That squeezed the DRAM tier to 0.18 GiB and put
  17 GiB on NVMe. The same default is *below* what Qwen3-30B (128 experts, top-8) needs once a layer is cold.
  - e4b's `min_hot_rows` gives the cold tier's own minimum: top_k × max(chunk, seqs), at most n_experts and the
    NVMe rows. A solver setup below it is refused in words.
  - The planner plans exactly that minimum. Mixtral's A2000 single-user plan now holds 17.3 GiB in DRAM with
    nothing on NVMe.
- **Unseen cards borrow serving slack.** The stated 4090/3090 have no receipts and took the 20% default, which put
  Qwen3-30B on tiers on 24 GB. Serving plans now borrow the largest slack measured for the same setup on any GPU,
  labelled heuristic: all-VRAM serving measured 0.06–1.46% on two cards. Qwen3-30B and single-user Qwen3.6 now plan
  all-VRAM on 24 GB.
  - Training keeps its default: its slack moved 8–39% with model and card.
- **gpt-oss-20b on 12 GB is a correct refusal.** Its per-expert biases do not ride the arena, so the hybrid tier
  cannot serve it, and all-VRAM needs 13.9 GiB + headroom.

**Update, 2026-10-06: serving three more families showed the sweep was too generous.** I tried to serve
granite-3.1-3b, LFM2-8B and granite-4.0-h-tiny on the A2000 through `bench/serve_validate.py`. All three were
planned feasible above; none could be served as planned.
- **LFM2-8B and granite-4.0-h-tiny are refused by the server.**
  - LFM2's `conv` layers are a type the paged runner keeps no state for.
  - granite-4.0-h's Mamba layers are labelled `linear_attention`, but the per-slot pool drives Gated DeltaNet only.
  - experts4bit-qlora#1215 states the runner's own rules (`paged_state_refusal`), and the estimate now refuses both
    on every card (`evidence/serve-family-sweep-hybrids.*`).
- **granite-3.1-3b could not be baked.** gnf4's `bake_nf4` found fused expert stacks only under an `.experts.` name,
  and GraniteMoe's are `block_sparse_moe.input_linear` / `output_linear`.
  - grouped-nf4-gemm#488 adds `fused_marker`: the arena baked in 47 s.
  - The planned serve then ran: **allocator peak 2.959 GiB against 2.997 estimated (−1.3%)**
    (`evidence/2026-10-06-a2000-family-serve/`).
  - That makes Granite the third family checked serving, after OLMoE and Qwen3-30B. `--bake-kw` passes the
    layout to the harness.
- **Qwen3.6-35B's linear-attention state is now priced** (experts4bit-qlora#1219). It is a bf16 conv window and an
  fp32 recurrent state, ~2.06 MiB per layer per slot. Its RTX 5090 plans grew from 22.40 to 23.44 GiB (4096 × 1)
  and from 23.13 to 24.59 GiB (8192 × 8).
- **Decode-graph buckets are capped at the sequences.** Most of that 4096 × 1 growth came from 16 scratch slots a
  single-user server never uses. A decode step never carries more rows than sequences, so serve plans now keep the
  buckets up to `max_seqs` (`[1]` for one user).
- Not checked by the planner: whether a model's checkpoint layout can be baked at all. That is the bake's knowledge,
  not the topology's.
- **ERNIE-4.5-21B-A3B: two server bugs, then a fourth served family** (`evidence/2026-10-06-a2000-family-serve/`).
  ERNIE's layer 0 is dense, and DeepSeek-V2's first layer is too.
  - The arena keyed rows by checkpoint layer (1–27) while the server asked by MoE ordinal (`KeyError: (0, 0)`).
  - The KV pool was sized by MoE layers (27), not decoder layers (28).
  - Both are fixed in experts4bit-qlora#1228. The planned tiered serve (736 rows in VRAM, 992 in DRAM) then ran.
  - It peaked 0.33 GiB over the estimate. An allocator-history replay (`ernie-residual-attribution.txt`) put
    457.5 MiB in the DRAM tier's prefill on the GPU, which #1229 now prices; it lands 117 MiB over.
- **DeepSeek-V2-Lite is refused.** Its multi-head latent attention hands the pool keys of 192 and values of 128
  against a 64-wide pool. The first prompt failed after the whole model had loaded. #1233 refuses it from the config,
  before loading, and in the estimate.

## 6d. SV1: the graphs, measured on a rented RTX 5090

Lane SV1 is experts4bit-qlora#1152, registered before the box in #1153 and read in #1161.
- **The box:** `sv1-5090-1`, one RTX 5090, **$1.50**, teardown proven.
- **The build:** `serve_paged` built in-process with the environment `ServeSetup.to_env()` gives, all-VRAM, from
  arenas baked on the box.
- **The workload:** 16 seeded 1,024-token prompts × 32 new tokens.

| arm | estimate | allocator peak | driver peak | tok/s |
|---|---|---|---|---|
| OLMoE eager | 9.196 | 8.812 | 9.506 | 33.9 |
| OLMoE decode graphs | 9.213 | 8.870 | 9.811 | 143.3 |
| OLMoE + prefill graph (pool 224 MiB) | 9.213 | 9.104 | 10.207 | 148.8 |
| Qwen3-30B decode graphs | 21.786 | 21.612 | 22.619 | 47.7 |
| Qwen3-30B + prefill graph (pool 310 MiB) | 21.786 | 22.169 | 23.375 | 49.8 |

All values GiB unless marked.

- **The estimate held** at −4.2% (OLMoE eager; the margin is prefill staging at its ceiling) and −0.8% (Qwen3-30B with
  decode graphs).
- **The two pools the estimate leaves unpriced, at NF4:**
  - decode graphs: +60 MiB;
  - the prefill graph: +0.24 GiB (OLMoE) and +0.57 GiB (Qwen3-30B).
  - SC2b's +3.3 GiB was the int4 stack.
  - e4b's notes now cite these numbers; nothing is priced from two points.
- **Planning with them** (`bench/import_sv1.py`, `bench/replan_serve.py` on FP1's RTX 5090 profile). The serve slack
  keys and the learned residual now include `prefill_graph`. A receipt with the same graph settings teaches its pool:

| plan (plan device total vs driver peak) | before SV1's receipts | after |
|---|---|---|
| Qwen3-30B, decode graphs | 22.59 vs 22.62 | 22.69 |
| Qwen3-30B, + prefill graph | 22.59 vs 23.38 (pool unpriced) | **23.24** |
| OLMoE, decode graphs | 9.91 vs 9.81 | 10.09 |

- **The incident.** The runner left the 16 GB Qwen3 arena where the driver's final fetch copies everything back.
  I stopped that fetch after saving the receipts, so the launcher records HARNESS_ERROR; the lane itself
  finished OK. Fixed in #1160.

## 6e. The int4 serving levers, measured on the A2000

experts4bit-qlora#1182 prices `ServeSetup.exp_int4` and `ServeSetup.attn_int4`, the server's int4 levers:
- **`exp_int4`:** int4-b32 expert stores replace the NF4 stacks, repacked at load from the source checkpoint.
- **`attn_int4`:** attention projections stored on the int4-b32 grid.

The predictions were written to `evidence/2026-10-05-a2000-int4-serve/predictions.txt` before any arm ran.

**The run.** One RTX A2000, `bench/serve_validate.py`, OLMoE-1B-7B, all-VRAM, eager, 4 sequences × 4,096 tokens,
1,024-token prompts, one arm per process.

| arm | estimate | allocator peak | vs the NF4 arm, priced / measured | plan total after receipts vs driver peak |
|---|---|---|---|---|
| NF4 | 5.959 | 5.571 | — | 6.209 vs 5.775 |
| `exp_int4` | 5.983 | 5.579 | +24 / +8 MiB (load peak +24 MiB exactly) | 6.193 vs 5.783 |
| `attn_int4` | 6.139 | 5.751 | +184.1 / +184.1 MiB | 6.311 vs 5.920 |
| both | 6.162 | 5.758 | +208 / +192 MiB | 6.322 vs 5.916 |

All values GiB unless marked.

- **Every estimate sits about 0.39 GiB above its peak.** That is the prefill-staging ceiling: 4,608 tokens priced,
  1,536 staged by these prompts. It is the same as SV1's S1.
- **The int4 attention items are exact.** The expert stores are exact at load. Serving peaks 16 MiB under the
  price, because the int4 prefill route's transients are smaller than NF4's.
- **int4 attention costs memory.** Each `Int4Linear` keeps a bf16 copy of its weight for calls over 16 rows (every
  prefill chunk). So attention costs more than in bf16 once a prompt is served. It is a decode-speed lever, not a
  memory one, and the plan says so.
- **The planner plans the levers only when fixed**, since they change the served weights. They are slack keys. Before
  these receipts, an int4 plan was charged the GPU's largest measured slack (0.89–0.92 GiB, conservative). After,
  each arm gets its own (0.04–0.09 GiB).
- **Found on the way: the levers kept their host heap.**
  - glibc kept the freed 8–16 MiB fp32 host tensors of the repack and the attention swap for the life of the server:
    +3.6 GB and +2.3 GB of anonymous memory after load.
  - One `malloc_trim(0)` returned 3.9 GB (`before-heap-fix/trim_probe.log`).
  - #1182 now trims per layer and per projection and drops each layer's fp32 stacks before the next read.
  - After load, every arm now sits below the NF4 build (0.57–0.61 GB against 0.70 GB).
  - The repack's own peak (4.88 GB) is priced at 12 B per parameter of a layer; 10.4–11.2 was measured.
  - The pre-fix receipts are in `before-heap-fix/`, which is not loaded as observations.
- **Receipts now record serving's own host peak** (`host_anon_serving_peak_bytes`). A load can peak above serving, as
  the repack does, so the planner learns host growth from that phase alone. The planner still sums the repack's
  load-time line with that growth, which over-reserves host memory by the smaller of the two (0.66 GiB here).
- **Speed is not read from this box.** The seat's two cores were contended. Arms on identical expert routes printed
  1.5 to 20 tok/s.
- **The source checkpoint is read whole at load.** That took 34 min cold off the seat volume (~5 MB/s) and ~6 min
  warm. It is listed as not modelled.

## 6f. SV2: the int4 levers on Qwen3-30B-A3B, decode graphs on, a rented RTX 5090

Lane SV2 is experts4bit-qlora#1207 (owner-approved, $5 cap), registered in #1208 and read in #1210. The box launched
before its registration was reviewed; see "How SV2–SV5 ran" below 6i.
- **The box:** `sv2-5090-1`, one RTX 5090, **$0.45**, teardown proven. Announced on the session bus before launch and
  reported there after.
- **The build:** `serve_paged` built in-process with the environment `ServeSetup.to_env()` gives, all-VRAM, decode
  graphs on.
- **The workload:** 16 seeded 1,024-token prompts × 32 new tokens.

| arm | estimate | allocator peak | driver peak | plan total before SV2 | plan total after |
|---|---|---|---|---|---|
| NF4 | 21.786 | 21.612 | 22.619 | 22.810 | 22.810 |
| int4 experts | 21.839 | 21.664 | 22.656 | 23.942 | **22.847** |
| int4 experts + attention | 22.410 | 22.236 | 23.252 | 24.539 | **23.446** |
| int4 experts + prefill graph | 21.839 | 22.222 | 23.398 | 23.942 | **23.391** |

All values GiB. Plans use FP1's RTX 5090 profile (`bench/replan_serve.py`; "before" = SV1 + P109 receipts).

- **Every registered reading held:**
  - the estimate 0.8% over the int4 experts' peak;
  - the int4 stores +54.0 MiB at load (priced +54);
  - int4 attention +585.2 MiB (priced +585.0);
  - the repack's host peak 3.30 GB under its 6.9 GiB price;
  - after load, the int4 builds hold 1.5 GB less host memory than NF4.
- **The prefill graph at int4 costs +571 MiB**, SV1's NF4 figure. SC2b's +3.3 GiB is not reproduced at this head. With
  that receipt on file, the prefill-graph plan learns its pool and lands within 0.03% of the driver peak.
- **Before SV2's receipts, an int4 plan was conservative by design:** no receipt had its slack keys, so it was charged
  this card's largest slack and this model's largest residual (SV1's prefill-graph pool). That is 1.3 GiB over. After,
  each arm's plan sits 0.19 GiB above its driver peak, as the NF4 arm does.
- **The NF4 build itself kept 3.21 GB of freed host heap** on this 48-core host. The int4 builds' trims (#1182) give it
  back, which is why they end below NF4. Trimming after every build is a separate e4b change.
- **The host plan stays conservative** (10.3 GiB planned against ~6.5 GiB measured at the repack). The repack price is
  a ceiling, and the planner still sums it with serving's growth, though the two do not coincide.

## 6g. SV3: the hybrid state pool and gpt-oss on a rented RTX 5090

Lane SV3 is experts4bit-qlora#1224 ($10 cap, within the owner's $50 approval), registered in #1225, read in #1232. The
box launched before its registration merged; see "How SV2–SV5 ran" below 6i.
- **The box:** `sv3-5090-1`, one RTX 5090, **$0.51**, teardown proven, announced on the bus before launch.
- **The arenas:** baked on the box through the loader.

| arm | estimate | allocator peak | driver peak | plan total before SV3 | after |
|---|---|---|---|---|---|
| Qwen3.6, 16 seqs, eager | 23.217 | 23.408 | 24.082 | 24.183 | 24.239 |
| Qwen3.6, 16 seqs, graphs | 24.187 | 24.423 | 25.568 | 25.614 | 25.593 |
| Qwen3.6, 1 seq, bucket 1 | 21.720 | 21.948 | 22.732 | 23.077 | 22.818 |
| gpt-oss-20b, 16 seqs, graphs | 15.391 | 15.673 | 17.109 | **16.567** | 17.162 |

All values GiB.

- **The linear-attention state pool matched its price to the byte** in all four Qwen3.6 arms: 16, 32, 17 and 2
  slots, 30 layers. That is #1219's per-slot arithmetic on the full model.
- **The estimate held:** every peak sat 0.8–1.8% over its estimate, inside ±5%. That is the allocator residual
  earlier runs measured, and the planner learns it.
- **Six families now served beside the estimate:** OLMoE, Qwen3-30B, granite-3.1, ERNIE-4.5, Qwen3.6 and
  gpt-oss-20b.
- **gpt-oss-20b was under-planned until now.** Before SV3 its plan was **0.54 GiB under** the driver peak: its 4.9%
  slack exceeds what it borrowed from other models' receipts. With its own receipt, +0.05 GiB.
- **The bucket cap's reading is an ALARM, and a finding.**
  - With one sequence and the default buckets, buckets 2–16 failed to capture on Qwen3.6, so that arm is not used.
    `bench/import_sv3.py` marks it `ALARM`, and the planner does not learn from it.
  - The server now captures only the buckets its sequences can use (experts4bit-qlora#1234). The planner defers to
    that rule.

## 6h. SV4: 30B on a real 24 GB card, at all-VRAM and on the solver's tiers

Lane SV4 is experts4bit-qlora#1236 ($10 cap within the owner's $50), registered in #1239, read in #1240. The box
launched before review; see "How SV2–SV5 ran" below 6i.
- **The box:** `sv4-4090-1`, one RTX 4090 (sm_89), **$0.15**, teardown proven.

| arm | estimate | allocator peak | the server's split (VRAM / DRAM / NVMe rows) | slack |
|---|---|---|---|---|
| all-VRAM, 1 × 4096, graphs | 18.685 | 18.668 (−0.1%) | — | 0.42% |
| the planner's tiers, 8 × 8192 | 18.264 | 17.471 (−4.3%) | 4,422 / 1,722 / 0, as priced | 4.08% |
| tiers 8 / 3 GiB, 4 × 4096 | 12.528 | 12.131 (−3.2%) | 3,236 / 1,213 / 1,695, as priced | 6.68% |

All values GiB.

- **The estimate held at 30B on all three placements.** The server split the experts exactly as priced, including
  4.2 GiB streamed from NVMe.
- **Tiered slack at 30B is 4–7%, not the 15% the planner had borrowed** from OLMoE's tiered runs on an A2000.
- **Measured slack changes the plan, not just the reserve.** With these receipts, the planner's 8 × 8192 serve on a
  24 GB card moves from tiers (11.7 GiB in VRAM, 15.2 GiB in DRAM) to **all-VRAM with decode graphs, 22.74 GiB of
  24** (`evidence/2026-10-06-sv4-rtx4090/replan-24gb-before-after.txt`). SV5 checks that plan on the same card class.
- **The pinned-memory reading missed, then got fixed.**
  - Every arm pinned 1.1–1.3 GB beyond the priced cold-tier landing.
  - A phase probe on the A2000 (`2026-10-06-a2000-family-serve/pinned-host-memory-probe.txt`) traced it to torch's
    caching host allocator. It kept the loader's freed pinned staging: 1.09 GB after loading Qwen3-30B, 1.37 GB after
    the hybrid tier, 4 MB of it in use.
  - experts4bit-qlora#1241 releases it at the end of the build: pinned memory after the build went from 1,314 MiB to
    14 MiB.

## 6i. SV5: the plan SV4 produced, checked on its card, ran out of memory

Lane SV5 is experts4bit-qlora#1242 ($5 cap within the owner's $50). The box launched from #1243's unreviewed head 37 s
after it opened. #1243 then merged after review, with consequences and a reducer written after the data. Under its
"What this file licenses", the read (#1257) is **exploratory** and licenses no change in experts4bit-qlora or here.
Through the reducer:
- Z1 MISSED;
- Z2 MISSED, on a lower bound;
- Z3 NO_READING;
- Z4 HELD.

This planner's own change after SV5 (`usable_capacity`, OOM lower bounds) is loggetta's decision, not licensed by
that read.
- **The box:** `sv5-4090-1`, one RTX 4090, **$0.13**, teardown proven.
- **The plan under test:** after SV4, the planner moved Qwen3-30B at 8 × 8192 on a 24 GB card to all-VRAM with
  decode graphs, 22.74 GiB planned. SV5 served exactly that plan.

| arm | estimate | allocator peak | driver peak | outcome |
|---|---|---|---|---|
| 8 × 1,024-token prompts | 22.150 | 21.734 | 22.477 | OK, under the plan |
| 8 × 8,000-token prompts | 22.150 | 22.59 at failure (+32 MiB requested) | — | **out of memory** |

All values GiB.

- **The plan was wrong twice.**
  - **The bulk KV flush was unpriced.** Its finished prompt's K/V across all 48 layers is quantized and held until one
    write, 526 MiB at a full 8,192-token slot. Short prompts never reach it. experts4bit-qlora#1247 prices it from
    the pool's own bound, reviewed on its code rather than on this read; the estimate becomes 22.664 GiB. The failed
    arm's 22.62 GiB is only where it stopped, a floor on its need, so an estimate above it does not show the
    estimate covers the peak.
  - **The card was smaller than stated.** The RTX 4090 reports 24,564 MiB and gave the process 23.52 GiB. The
    planner now caps a GPU class's budget at the capacity a receipt's out-of-memory message reports
    (`usable_capacity`).
  - It also reads an out-of-memory receipt as a lower bound on that setup's need (allocated + requested), charged
    where today's estimate falls short.
- **Replanned with the fixes** (`evidence/2026-10-06-sv5-rtx4090/replan-24gb-with-fixes.txt`): with SV4's receipts
  alone the planner already returns to tiers (13.06 GiB in VRAM); with SV5's too, the budget is 23.52 GiB and the
  VRAM tier 12.63 GiB. Neither repeats the failing plan.
- **The check.** A plan that changed on new evidence was run on the card it targets before anyone relied on it, and
  it failed in a way that names two causes.

### How SV2–SV5 ran

All four boxes launched before their registrations had merged after review:
- SV2 launched one minute after a change request;
- SV4 launched 1 min 41 s after its registration opened;
- SV5 launched 37 s after its registration opened.

experts4bit-qlora's maintainer posted the requirements on each work item before the registration PR. This session
skipped them each time:
- the registration merges first;
- registered consequences;
- a reducer with a self-test;
- host-only exit codes.

The maintainer's dated notes (experts4bit-qlora#1213, #1237, #1244, and the review on #1243) record that SV2–SV4's
reads license no change to that package. SV5's read (#1257) is exploratory under #1243's merged text.

The planner consumes these receipts as data, by design: they are measurements, and the estimate items they inform are
re-priced in experts4bit-qlora only through PRs reviewed on their code. Weigh SV2–SV5 as measurements taken outside
the registration process, not as pre-registered tests.

## 6j. SV6: the revised 24 GB plan, run at the longest prompts, registered first

Lane SV6 is experts4bit-qlora#1267 ($5 cap within the owner's $50). It was the first serving lane run in order:
- the registration (#1268) and a driver-floor amendment (#1272) merged after review before any data;
- the box launched from the merge (`af0d95df`) through the launcher's prereg check;
- the read (#1275) was reviewed before it merged.

- **The box:** `sv6-4090-3`, one RTX 4090, **$0.32**, teardown proven.
  - Two earlier attempts reached no arm ($0.052). The cheapest verified 4090 runs driver 535, where the image's CUDA
    12.8 cannot initialise.
  - The amendment's driver floor turns that into a host refusal the launcher can exclude. adertha-agents#184 tracks
    an offer-side filter.
- **The plan under test:** this planner at `dd4783f`. With #1247's bulk flush priced and the 23.52 GiB capacity cap,
  it refused all-VRAM and planned the solver's tiers: VRAM 12.631 / DRAM 15.187 GiB, eager decode, 22.343 GiB.

| arm | estimate | allocator peak | driver peak | plan |
|---|---|---|---|---|
| 8 × 1,024-token prompts | 20.476 | 19.168 (−6.4%) | 20.154 | 22.343 |
| 8 × 8,000-token prompts | 20.476 | 20.146 (−1.6%) | 21.357 | 22.343 |

All values GiB.

- **The plan held at the longest prompts.** The driver peak was 0.985 GiB under the plan, with no out-of-memory error.
  The server's tier rows equalled the estimate's (5,109 / 1,035 / 0).
- **#1247's bulk flush, measured.** Long prompts minus short added 1,001.0 MiB to the allocator peak. The flush and
  staging formulas give 1,003.9 MiB at those lengths: −0.3%.
- **What the planner takes from it, and only that.** The registration licenses these receipts as same-setup evidence
  for the allocator reserve and the CUDA context on this card class, "and nothing more" (the maintainer, #1275).
  - Receipts can now carry `licensed_for`, and the planner learns only the named uses from them. A receipt without the
    field teaches every use, as before.
  - SV6's are imported with `licensed_for: [reserve, context]` (`bench/import_sv4.py --lane SV6 --licensed-for
    reserve,context`).
  - Replanned (`evidence/2026-10-06-sv6-rtx4090/replan-24gb-scoped.txt`): the reserve falls from 1.367 GiB (borrowed
    from SV4's deepest tier arm, 6.7%) to 0.769 GiB (SV6's 3.8%), and the plan from 22.343 to 21.744 GiB.
  - The tier split does not move. SV6's slack matches only its exact setup, so a larger VRAM tier would still borrow
    6.7%.
  - Unscoped, SV6's long arm would also have raised host growth while serving from 1.05 to 3.45 GiB. The scope keeps
    that out, because nothing registered it.

## 6k. Reserve matching: by workload shape, with every receipt's `bulk_kv`, and no anchor transfer for serving

Three planner fixes to how a serve plan borrows its allocator reserve. All planning only; no box ran
(`evidence/2026-10-07-reserve-matching/`).

- **Old receipts could not match a current plan.** experts4bit-qlora #1247 added `bulk_kv` to the server's setup, so
  every plan names it and no receipt imported before then did. Whole-setup matches stopped firing and plans fell back to
  the most conservative same-key reserve.
  - The server's default changed with #1200 (merged 2026-10-05T19:32Z). Lane SV2 ran three hours later from a branch
    without it, so the field cannot be filled from a date.
  - `bench/annotate_bulk_kv.py` recorded each value from the code its run used: the lane's commit, or a seat run that
    predates #1200. It records the basis in `provenance.setup_inferred`.
  - The 2026-10-06 A2000 family runs record no commit, so they stay without the field. A test now requires every other
    serve receipt to name it.
- **Same shape before same key.** Tier budgets (and `hot_rows`, which the server derives from them) move a few one-time
  allocations, not the workload's transient ones.
  - A receipt that differs from the plan only there is now preferred to a same-key receipt of another shape. Lanes SV4
    and SV6 measured 4.1% and 3.8% at two VRAM tiers of one 8 × 8192 shape, against 6.7% for SV4's 4 × 4096 arm with an
    NVMe tier.
  - A receipt scoped by `licensed_for` still counts only for its own whole setup: SV6's licence.
- **No anchor transfer for serving.** Moving another GPU's slack through an anchor model's ratio was built for training
  (8–39% slack). Serving slack is 0.06–1.5%, and the ratio of two such numbers is noise: a ~14× ratio turned OLMoE's
  2.8% on an RTX 5090 into 4.0 GiB of reserve on an RTX 4090, and gpt-oss-20b's into 10.4 GiB. Serving now takes this
  GPU's same-setup slack from another model, as the next rule always did.

**What moved** (`serve-sweep-ab.txt`, all 10 families × 4 cards × 2 workloads): 14 of 80 plans. No status, placement
or decode-graph mode changed.
- Larger VRAM tiers at 8 × 8192 on the 24 GB card:
  - Qwen3-30B: 12.63 → 13.14 GiB, 5,316 of 6,144 expert rows;
  - Qwen3.6: 13.80 → 14.31 GiB;
  - Mixtral: 10.52 → 11.07 GiB.
- The OLMoE and gpt-oss reserves on that card fell 1–5 GiB.
- Restored whole-setup matches trimmed 0.05–1.2 GiB elsewhere.

**Not run.** The Qwen3-30B 24 GB plan (VRAM 13.143 GiB, 22.344 GiB planned) is new; SV6 ran 12.631. Its reserve
comes from SV4's same-shape arm. That read licenses no change in experts4bit-qlora, but the planner uses its receipts as
data, as it always has. The plan is a planner output, not a measured one, and a lane registered first would check it on
the card.

## 7. Not measured, said plainly

- **No performance model.** Speed is ordered from evidence, never predicted, apart from the transfer lower bound.
- **The activation heuristic** is a formula, checked against nine allocator peaks (three here, six in the
  register), not derived.
- **Qwen3-30B ran once, on a rented RTX 5090 (FP1).** Every other model above 8B was planned, not run.
- **Serving** is planned at both placements. All-VRAM is checked on two model/card pairs. The solver's tiers are
  checked on one model and card (OLMoE, A2000) and assume uniform routing, as the server does. The int4 levers
  are priced and checked on OLMoE / A2000 (6e) and Qwen3-30B / RTX 5090 with decode graphs (6f), and planned only
  when fixed. A measured routing profile and
  decode speed are not planned; see `SERVING-PRESSURE-TEST.md`.

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
