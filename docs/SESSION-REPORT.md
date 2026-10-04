# Session report: should the umbrella layer exist, and what is it? (2026-10-04)

Short answers. Detail is in [ARCHITECTURE.md](ARCHITECTURE.md), [RESULTS.md](RESULTS.md) and
[SERVING-PRESSURE-TEST.md](SERVING-PRESSURE-TEST.md).

## Does the layer have a reason to exist?

**Yes, a narrow one: policy.**
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
| **experts4bit-qlora** | model families, loading, adapters, the training/serving runtime, residency integration, what its own mechanisms cost | `describe_moe` (topology from config + meta tree), `QLoRASetup`, `estimate_qlora_footprint` (itemized, derived vs heuristic, including link traffic), `setup_refusals`, `prepare_qlora_training`; `estimate_serve_footprint` / `ServeSetup` / `paged_kv_pool_bytes` (the paged server, e4b#1080); `loader.check_admission` / `admission_refusal`; one routed-top-k alias list |
| **planner layer** (this repo) | hardware inventory with provenance; budgets, headroom and runtime overheads (learned from receipts); candidate ordering by objective with cited evidence; refusal and computed suggestions; plan and receipt formats; execution harness; CLI | everything here |

## Interfaces between the layers (the complete list)

- **kernel → runtime:** `route_for(...) -> (route | None, reason)`.
- **runtime → planner:** `describe_moe`, `setup_refusals`, `estimate_qlora_footprint`, `prepare_qlora_training`.
- **inside the planner:** one backend module exposing `probe`, `candidates`, `policy_notes`, `estimate`,
  `speed_rank`, `describe_kernel` and an executor. It is a plain tuple, not a plugin registry.

## Dependency direction

`planner → experts4bit-qlora → grouped-nf4-gemm`. No cycles. Neither package knows the planner exists: their
APIs, docs, changelogs and PR descriptions do not mention it.

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
- **planner** (private `pjordanandrsn/loggetta`): new.

## What works now

```
python -m loggetta inspect
python -m loggetta plan  <model> [--vram G --ram G --experts device|host --objective speed|min_vram|min_ram \
                                  --target-s-per-step S --fix FIELD=VALUE --hardware saved.json --observations DIR]
python -m loggetta train <model> ...  -> plan, then execute, then write a receipt
```

- **Real QLoRA runs through it on the A2000:**
  - OLMoE-1B-7B: resident fused, host-offload fused, reference loop, plus a repeat;
  - Granite-3.1-3B (second family, no code changes);
  - Granite-4.0-h-tiny (Mamba hybrid).
- **Plan-only, ten families** with zero per-family planner code. The sweep exposed four real gaps, fixed at the
  owning layer.

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

## What remains speculative

- **No performance model.** Times are not predicted, apart from the transfer lower bound.
- **The activation term is heuristic.**
- **Reserve slack** is measured at 30B on an RTX 5090 (FP1). On other GPUs it is transferred through an anchor ratio, stated as heuristic.
- **Above 8B, only Qwen3-30B-A3B was run** (FP1, RTX 5090, $0.85); the rest were planned, not run.
- **Serving is planned for one placement only** (all-VRAM, context × concurrency, e4b#1080). The KV pool is the
  server's own arithmetic; no serve receipt has checked the total yet. Tiered placements are refused in words.
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

## Could the project be renamed tomorrow?

**Yes.** A rename touches the package directory, two lines of `pyproject.toml`, the README title and the GitHub
repository name.
- No schema, environment variable, protocol string, serialized field or lower-layer API carries the name.
- `tests/test_renameable.py` fails if any of that changes.
- The two public packages never mention it.
- Existing receipts name it only in provenance: argv, source file paths and the import-name key of the code that
  ran. That is the historical record of what executed, correct to keep. A renamed package writes its new name there
  automatically.

## What the final standard asks, answered

1. **The existing packages are still healthy independent projects.** Additive APIs; behaviour unchanged; their
   own checks green.
2. **The new layer has a clear reason to exist.** Yes: policy, as above.
3. **The dependency direction makes sense.** Yes.
4. **A real workload runs through it.** Yes: six A2000 runs, two families plus a hybrid.
5. **Decisions are explicit and inspectable.** Every line of every plan carries its basis.
6. **It is useful already.** It says what fits on this card, why, and what to change. It caught a would-be crash
   (MLA attention in `add_attention_lora`) and a silent no-op (out_proj-only attention, before #1048 taught the
   detector). After #1048 it picked up LFM2's attention with no planner change.
7. **Serving can grow into it without a rewrite.** Yes: the all-VRAM placement was added as a second workload
   kind with no planner rewrite (backend candidates, estimate, label and explanation; a serve suggestion search).
   The pressure-test doc names what is still missing.
8. **Families can grow into it without special cases.** Yes, demonstrated on ten with no family branches.
9. **Performance claims are measured and reproducible.** Every number comes from a receipt with commits.
   Provenance defects found during the session are stated and fixed.
10. **Renameable.** Yes.
