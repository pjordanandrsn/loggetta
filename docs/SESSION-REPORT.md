# Session report: should the umbrella layer exist, and what is it? (2026-10-04 to 10-05)

Short answers. Detail is in [ARCHITECTURE.md](ARCHITECTURE.md), [RESULTS.md](RESULTS.md) and
[SERVING-PRESSURE-TEST.md](SERVING-PRESSURE-TEST.md).

> **Correction (2026-10-09).** This report is dated and is left as written. A claims audit against the committed
> evidence found these lines out of date:
> - **Prefill-graph pool:** +0.23 / +0.56 GiB, not +0.24 / +0.57.
> - **Serving slack:** up to 4.88% at all-VRAM (gpt-oss, SV3) and as low as 4.08% under the solver (SV4), not
>   0.06–1.5% and 8–15%.
> - **The tiered placement** was checked on two models and cards: Qwen3-30B-A3B on an RTX 4090 (SV4) and ERNIE-4.5-21B
>   on the RTX A2000.
> - **int4 plans:** q_exp and q_both land 0.19 GiB over the driver peak; the int4 + prefill-graph plan lands 0.007 GiB
>   under it.
> - **Six families served:** +0.8–1.8% is SV3's four arms; across the six families the peaks run −4.2% to +4.1% of
>   the estimate.
> - **A2000 runs:** seven planned receipts, not six (R1, R1b, R3, R3b, R4, R5, R6); the +0.01 to +0.21 GiB range holds.
> - **e4b #1141** merged on 2026-10-05. Loggetta now has two backends (experts4bit, dense).
> - **Above 8B:** Qwen3.6-35B-A3B and gpt-oss-20b (SV3, RTX 5090) and ERNIE-4.5-21B (A2000) were also run, not only
>   Qwen3-30B-A3B.
>
> Current figures are in [RESULTS.md](RESULTS.md).

**What it is, in one sentence.** Loggetta turns a workload, a machine and constraints into an inspectable
`ExecutionPlan`. It hands that plan to the backend that knows how to run it, and keeps the `ExecutionReceipt` as
evidence for the next plan. Loggetta decides what should execute; the backend knows how to execute it.

## Does the layer have a reason to exist?

**Yes, a narrow one: deciding what should execute (policy), and recording how that turned out (the receipt).**
- The two packages already had nearly all the *mechanism* a planner needs: a placement solver, a time-cost model,
  pinned-memory costing, a cgroup-aware RAM reader, capability floors and a measured-claims register.
- What neither had was a place to decide before loading, to compare alternatives, and to refuse with reasons.
  Their configuration was about 130 env vars, and the only automatic memory decision was halving the batch after an
  OOM.
- That missing piece is policy, and it does not belong in either package.

**Most of what the request imagined for the umbrella belongs below it.** Topology, footprint, refusal rules and
the kernel route were all added to the packages that own them.

## What belongs where

| | owns | added this session |
|---|---|---|
| **grouped-nf4-gemm** | kernels, packed layouts, kernel dispatch, host/NVMe residency primitives, pinned-memory costing | `nf4_route.route_for(capability, *, has_grouped_mm, requested, n_groups)`, the training-route decision as a pure function, with `MIN_CAPABILITY` / `GROUPED_MM_CAPABILITY` as data |
| **experts4bit-qlora** | model families, loading, adapters, the training/serving runtime, residency integration, what its own mechanisms cost | `describe_moe` (topology from config + meta tree), `QLoRASetup`, `estimate_qlora_footprint` (itemized, derived vs heuristic, including link traffic), `setup_refusals`, `prepare_qlora_training`; for the paged server `ServeSetup` (+ `to_env`), `estimate_serve_footprint`, `paged_kv_pool_bytes`, `solver_tiers`, `bytes_per_expert`, `min_hot_rows`, prefill staging and the cold-row stack (#1080, #1098, #1115, #1122, #1130, #1139, #1141); `fused_append_unsupported` and the sm_89 decode fix (#1090); `loader.check_admission` / `admission_refusal`; one routed-top-k alias list |
| **Loggetta** (this repo) | the Planner, the `ExecutionPlan` and the `ExecutionReceipt`; hardware inventory with provenance; budgets, headroom and runtime overheads (learned from receipts); candidate ordering by objective with cited evidence; refusal and computed suggestions; which measured evidence later plans trust; execution orchestration (dispatch to the backend, the measured loop around e4b's prepared model, the receipt); CLI | everything here |

## Interfaces between the layers (the complete list)

- **grouped-nf4-gemm → experts4bit-qlora:** `route_for(...) -> (route | None, reason)`.
- **experts4bit-qlora → Loggetta:**
  - training: `describe_moe`, `setup_refusals`, `estimate_qlora_footprint`, `prepare_qlora_training`;
  - serving: `ServeSetup` / `to_env`, `estimate_serve_footprint`, `min_hot_rows`, and `fused_append_unsupported`
    (where decode graphs can run).
- **inside the planner:** one backend module, a plain tuple rather than a plugin registry, exposing:
  - `probe`, `candidates`, `estimate`, `speed_rank`, `label`, `explain`, `policy_notes`, `describe_kernel`;
  - the handoff: `executor(kind)` (`None` for kinds planned only) and `run_tag`;
  - for serving, `fill_knobs` (fields the planner sizes to a budget), `resolve` (fields left to the mechanism),
    `SLACK_KEYS` and `KERNELS_FOR`.

## Dependency direction

`loggetta → experts4bit-qlora → grouped-nf4-gemm`. No cycles. Neither package imports the planner, and their APIs
and changelogs do not mention it.

**Correction to the requested diagram.** Serving lives in experts4bit-qlora today, because it needs model
loading and the expert stores. A future serving backend that skips experts4bit-qlora can still call the kernels
directly; nothing prevents it.

## What changed in each repository

- **grouped-nf4-gemm, [PR #464](https://github.com/pjordanandrsn/grouped-nf4-gemm/pull/464):**
  - `route_for` and its CPU tests;
  - CHANGELOG, README, KERNEL_CONTRACT and capabilities entries;
  - rebased over #459 and #463 (dense routes); route suites 42 passed on the A2000.
- **experts4bit-qlora, [PR #1055](https://github.com/pjordanandrsn/experts4bit-qlora/pull/1055):**
  - topology, recipe, admission and top-k changes, with tests;
  - companions: CHANGELOG, capabilities, README door table, llms bundle;
  - `train.py` untouched.
- **experts4bit-qlora, serving** (all merged except #1141, on auto-merge). The estimate:
  - #1080 serve estimate;
  - #1098 the prefill graph's pool named;
  - #1115 the solver's tiers and the hybrid tier's host buffers;
  - #1122 exact bytes per expert;
  - #1130 `min_hot_rows` + refusal;
  - #1139 the cold-row stack;
  - #1141 prefill staging.

  One bug fix: #1090, eager decode below sm_89 instead of a crash in Triton's compiler.
- **Loggetta** (`pjordanandrsn/loggetta`): new.

## What works now

```
python -m loggetta inspect
python -m loggetta plan  <model> [--vram G --ram G --experts device|host --objective speed|min_vram|min_ram \
                                  --target-s-per-step S --fix FIELD=VALUE --hardware saved.json --observations DIR]
python -m loggetta execute plan.json  -> a saved plan (plan --out), through its backend, to a receipt
python -m loggetta train <model> ...  -> plan and execute in one step
```

- **Real QLoRA runs through it on the A2000:**
  - OLMoE-1B-7B: resident fused, host-offload fused, reference loop, plus a repeat;
  - Granite-3.1-3B (second family, no code changes);
  - Granite-4.0-h-tiny (Mamba hybrid).
- **Plan-only, ten families** with zero per-family planner code. The sweep exposed four real gaps, fixed at the
  owning layer.
- **Serving:** `plan <model> --workload serve --context T --concurrency N`.
  - All-VRAM when it fits, else the solver's VRAM/DRAM/NVMe tiers, sized to the budgets.
  - `hot_rows` is planned, and the plan's "Why" carries the server's environment.
  - `bench/serve_validate.py` builds `serve_paged` with exactly that environment and writes a receipt.
  - `execute` refuses serve plans: they are planned only.

## What was measured (see RESULTS.md)

- **Orchestration changes no numbers.**
  - Step-1 loss is bitwise identical between the planned run and the hand-composed public-API run.
  - Over 12 steps they differ by at most 0.0042, against 0.0068 between two identical planned runs.
  - Memory matches the hand-composed run to 0.4 MB.
- **Allocator-peak estimates** are within +0.01 to +0.21 GiB on our six runs (three families, resident and offload), and within −1.8 / +1.5 GB of six
  register peaks (three models, three cards).
- **Runtime overheads learned from receipts** (CUDA context, allocator reserve slack, host baseline) moved the
  driver-view error from +1.06 to +0.26 GiB.
- **Transfer lower bound** ≥1.16 s/step against 2.79–3.13 s measured: it holds.
- **Fused vs reference loop** on this card: 3.1×. Host-offload vs resident: about 1.3×.
- **Serving (RESULTS 6–6c):**
  - The serve estimate is 0.7–0.9% under P109's Qwen3-30B allocator peaks on an RTX 5090.
  - A planned tiered OLMoE serve on the A2000 matched the server's own split (272 / 421 / 331 experts), with the
    allocator at 2.052 planned against 2.051 measured.
  - The leftover ~0.18 GiB was attributed by allocator-history replay to prefill staging and the cold-row stack, now
    both priced.
  - With two 4,000-token prompts, the estimate (5.420 GiB, 0.56 GiB of it staging) sits 6 MiB above the measured
    peak.
  - **SV1, on a rented RTX 5090** ($1.50, e4b#1152/#1161):
    - the estimate held with decode graphs and the prefill graph on: −4.2% on OLMoE eager, −0.8% on Qwen3-30B with
      graphs;
    - the two unpriced pools measured at NF4: decode graphs +60 MiB, prefill graph +0.24 / +0.57 GiB;
    - learned by graph settings, the Qwen3-30B prefill-graph plan is within 0.6% of its driver peak.
  - **The int4 levers (RESULTS 6e, A2000, e4b#1182)** are priced:
    - int4 attention exactly, +184.1 MiB both ways;
    - int4 experts +24 MiB priced, against +8 MiB at the serving peak and +24 MiB at load.
    - int4 attention costs device memory (each projection keeps a bf16 copy), so it is a speed lever.
    - Measuring it found glibc keeping 2.3–3.6 GB of the levers' freed host heap; the fix in #1182 trims it.
  - **SV2, on a rented RTX 5090** ($0.45, e4b#1207/#1210, owner-approved, coordinated over the bus): every reading
    held on Qwen3-30B with decode graphs.
    - int4 experts −0.8% against the estimate;
    - the int4 stores +54.0 MiB at load, priced +54;
    - int4 attention +585.2 MiB, priced +585.0;
    - the prefill graph at int4 +571 MiB, as at NF4, not SC2b's 3.3 GiB.
    - With the receipts on file, the int4 plans land 0.19 GiB above the driver peak.
  - **SV3 and the A2000 family runs (RESULTS 6c update, 6g; $0.51 rented, e4b#1224/#1232):**
    - Qwen3.6's linear-attention state pool matched its price to the byte.
    - Six families are now served beside the estimate (+0.8–1.8%). gpt-oss's plan was 0.54 GiB under until it had a
      receipt of its own.
    - Serving them found server bugs, each fixed: arenas of models with leading dense layers (#1228); MLA refused
      before loading (#1233); buckets above the sequences fail to capture (#1234); the DRAM tier's prefill transient
      priced (#1229).
  - **SV4 and SV5 on a rented RTX 4090 (RESULTS 6h, 6i; $0.15 + $0.13):**
    - 30B serves beside the estimate at all-VRAM and on the solver's tiers.
    - The all-VRAM plan SV4's receipts produced ran out of memory at 8,000-token prompts. Two causes: an unpriced bulk
      KV flush (e4b#1247) and a card that gives the process 23.52 GiB.
  - **Provenance:** SV2–SV5's boxes all launched before their registrations merged after review (RESULTS, "How SV2–SV5
    ran"). experts4bit-qlora's maintainer notes say their reads license no change there, and SV5's read is unreviewed.
    They are measurements, not pre-registered tests.
  - Serving slack is 0.06–1.5% at all-VRAM and 8–15% under the solver, against training's 8–39%.

## What remains speculative

- **No performance model.** Times are not predicted, apart from the transfer lower bound.
- **The activation term is heuristic.**
- **Reserve slack** is measured at 30B on an RTX 5090 (FP1). On other GPUs it is transferred through an anchor ratio, stated as heuristic.
- **Above 8B, only Qwen3-30B-A3B was run** (FP1, RTX 5090, $0.85); the rest were planned, not run.
- **Serving assumes uniform routing**, as the server does (no profile reaches `solve_placement`). A measured hot
  set and decode speed are not planned.
  - The int4 levers are planned only when fixed, and checked on two model/card pairs (OLMoE on the A2000; Qwen3-30B on
    an RTX 5090 with graphs).
  - The graph pools are learned from receipts, not priced: two points each.
  - The tiered placement is checked on one model and one card.
- **Ceilings, not expectations.** Prefill staging and the cold-row stack are priced at their worst case, so short
  prompts leave margin unused.
- **Multi-GPU is a list in the data model,** nothing more.

## Architectural risks

1. **The packages move fast.** Main merged three times under this work: #459/#463 changed exactly the routing
   `route_for` encodes. Mitigations: `route_for` is in the same file as the rule, and the CUDA test asserts that
   the two agree. It still has to be kept up by whoever changes routing.
2. **Estimator coupling.** `estimate_qlora_footprint` is correct only while it builds the same classes as the
   loader, which it does by construction. A new engine with a different memory pattern needs its own items, or it
   will be under-priced silently. The "not modelled" list is the guard; it must be kept honest.
3. **Env-var policy is still below.** About 25 knobs in grouped-nf4-gemm and about 130 in experts4bit-qlora
   remain. The planner records the ones `prepare_qlora_training` can see (fused RoPE / RMSNorm,
   `E4B_MOE_KEEP_LAYERS`) but does not control them. A plan is reproducible only together with the environment
   it ran in.
4. **Shared hardware.** The A2000 and the host are shared with other sessions. "Free now" is a moving budget;
   plans state it and warn, but they cannot reserve it.
5. **Server defaults the planner overrides.** `E4B_PAGED_HOT_ROWS=64` is too few for Qwen3-30B once a layer is on
   NVMe and far too many for Mixtral. `E4B_PAGED_PREFILL_GRAPH=auto` holds an unpriced pool. Plans set both, but a
   hand-started server still gets the defaults.

## Could the project be renamed tomorrow?

**Yes.** A rename touches the package directory, two lines of `pyproject.toml`, the README title, the GitHub
repository name and the reserved PyPI project.
- No schema, environment variable, protocol string, serialized field or lower-layer API carries the name.
- `tests/test_renameable.py` fails if any of that changes.
- The two public packages never mention it.
- Existing receipts name it only in provenance: argv, source file paths and the import-name key of the code that
  ran. That is the historical record of what executed, correct to keep. A renamed package writes its new name there
  automatically.

## What the final standard asks, answered

1. **The existing packages are still healthy independent projects.** Additive APIs; behaviour unchanged; their
   own checks green.
2. **The new layer has a clear reason to exist.** Yes: the plan (policy) and the receipt (evidence), as above.
3. **The dependency direction makes sense.** Yes.
4. **A real workload runs through it.** Yes: six A2000 runs, two families plus a hybrid.
5. **Decisions are explicit and inspectable.** Every line of every plan carries its basis.
6. **It is useful already.** It says what fits on this card, why, and what to change. It caught a would-be crash
   (MLA attention in `add_attention_lora`) and a silent no-op (out_proj-only attention, before #1048 taught the
   detector). After #1048 it picked up LFM2's attention with no planner change.
7. **Serving can grow into it without a rewrite.** Yes. Serving was added as a second workload kind, then
   tiered:
   - the planner gained generic hooks (fill knobs, resolve, slack keys, learned serve overheads);
   - all mechanism stayed in e4b;
   - running it found a real e4b crash on Ampere cards (#1090).

   The pressure-test doc names what is still missing.
8. **Families can grow into it without special cases.** Yes, demonstrated on ten with no family branches.
9. **Performance claims are measured and reproducible.** Every number comes from a receipt with commits.
   Provenance defects found during the session are stated and fixed.
10. **Renameable.** Yes.

## Boundary check before public release (2026-10-05)

The code already matched *plan here, mechanism below*. The planner is pure, and model building, engines and
kernels come from experts4bit-qlora through `prepare_qlora_training`. Four concrete leaks were fixed; no schema
changed.
- **`runtime.py` → `execution.py`.** The module that dispatches a plan and writes the receipt no longer claims a
  runtime. The old import path still works.
- **Dispatch through the selected backend.** `execute` asks the backend module for its executor and its run-id tag,
  instead of naming the backend and reading its setup fields itself.
- **A plan runs only on the GPU it was made for.** Before this, `train --hardware other.json` ran here and wrote a
  receipt naming the other card, which later plans would have learned from.
- **`execute PLAN.json`.** The saved plan, not a re-plan, is what runs.

**Left as is, deliberately.**
- The executor's integrity check, `_expert_digest`, reads experts4bit-qlora's storage attributes. Moving it below
  would need a new public API there.
- The measured loop stays here. It is the receipt's instrument, and the e4b training engines it drives come from
  e4b itself.
