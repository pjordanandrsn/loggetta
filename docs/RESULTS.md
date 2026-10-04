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

All values GiB.

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

**Planned with no per-family code anywhere in the planner** (`bench/family_sweep.py`, seat hardware profile,
QLoRA at seq 2048):

| model (model_type) | outcome on the A2000 |
|---|---|
| OLMoE-1B-7B (olmoe) | feasible: resident, grouped NF4 |
| Qwen3-30B-A3B (qwen3_moe) | feasible after the pageable-host axis: host-resident pageable experts, about 18 GiB host. Pinned power-of-two homes would need more host RAM than is free now |
| Qwen3.6-35B-A3B (qwen3_5_moe; 10 of 40 layers attention) | refused at seq 2048: device short by about 3.5 GiB. Suggests "reduce tokens per micro-batch to 512" |
| granite-3.1-3b-a800m (granitemoe, pre-fused legacy spelling) | feasible, resident; **run for real** (R5) |
| granite-4.0-h-tiny (granitemoehybrid: Mamba + 4 attention layers) | feasible, resident; the non-attention mixers are listed as not modelled; run in R6 |
| LFM2-8B-A1B (lfm2_moe; out_proj attention) | feasible; attention LoRA turned off by policy, with the reason |
| Mixtral-8x7B (mixtral) | refused: host short |
| ERNIE-4.5-21B-A3B (ernie4_5_moe, admitted by convention) | feasible: host-resident experts |
| DeepSeek-V2-Lite (deepseek_v2, MLA) | feasible with attention LoRA off: the detector cannot describe MLA |
| gpt-oss-20b (gpt_oss) | refused by structure: biased experts cannot take ExpertsLoRA, and biased attention cannot go NF4 |

**What the sweep forced.**
- Two refusals were added in experts4bit-qlora: attention that cannot be described (MLA) and attention with no
  wrappable projections (LFM2). Without them, DeepSeek-V2 would have crashed in `add_attention_lora`, and LFM2
  would have "trained attention" with zero adapters.
- A not-modelled note was added for non-attention mixers.
- A `pin` axis was added to the planner.

None of these is a family-name branch.

## 5. Not measured, said plainly

- **No performance model.** Speed is ordered from evidence, never predicted, apart from the transfer lower bound.
- **The activation heuristic** is a formula, checked against nine allocator peaks (three here, six in the
  register), not derived.
- **Qwen3-30B and every model above 8B were planned, not run here.** The A2000 runs are OLMoE, Granite-3.1 and
  Granite-4.0-h-tiny.
- **Serving** is represented and refused; see `SERVING-PRESSURE-TEST.md`.

## Provenance

- Every receipt records the git commit and dirty flag of each imported source tree, distribution versions, argv
  and the whole plan.
- R1's planner source was recorded against the enclosing `/home/node/work` repository: the planner was not yet a
  git repository, and `git_commit` did not check that the file was tracked. That was fixed before R1b; later
  receipts are clean.
- experts4bit-qlora's base moved during the session (main was merging, including #1048). R1–R4 ran on the branch
  rebased over `5c74564e`/`2631d3d7`, R5 onward over `520b0b5d`; the receipts say which.
