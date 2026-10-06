# Serving pressure test

**Status (third pass).**
- **Scenario A** is planned: the paged server's all-VRAM placement, context × concurrency, with its FP8 paged KV pool
  priced by the server's own arithmetic.
- **Scenarios B and C** are planned as the server runs them: the solver's VRAM/DRAM/NVMe tiers with uniform routing
  (e4b#1115). The planner sizes the tiers to the budgets, and a run on the A2000 matched the planned split exactly
  (RESULTS 6b).
- **Not planned:** a measured routing profile (scenario B's real hot set), and throughput (D).

The first pass refused every `kind="serve"` workload and checked only that the concepts in place could represent
serving without a rewrite. Each scenario is mapped onto three things: the code that already serves in
experts4bit-qlora (e4b), the plan data model, and what is missing.

## What already exists below the planner

- **Serving engine.**
  - `serve_paged` provides a paged runner, a continuous scheduler (`ContinuousScheduler.plan() -> StepPlan`) and an
    FP8 paged KV cache.
  - Hot / pipelined / hybrid / NVMe residency engines.
  - int4 serving lanes.
- **Placement.** `engines/placement.solve_placement` places each (layer, expert) in VRAM, DRAM or NVMe. It is
  driven by measured bandwidths (a calibration blob) and a routing profile, and its batch law is
  `expected_weight_reads(p, B) = 1 - (1-p)^B`.
- **Kernel package:**
  - decode dispatch (`nf4_grouped.decode_dispatch`, `int4_b32._plan`);
  - KV block sizing (`fp8_kv.kv_block_bytes`);
  - pinned-tier sizing (`nvme_residency.capacity_for_bytes`, `pinned_request_cost`);
  - a time-to-contribution cost model (`cold_deadline.Costs`).

So serving *mechanism* is mature, and it stays in experts4bit-qlora. What was missing is the **planner**: a way to
choose among those mechanisms before loading, to say why, and to record the choice as an `ExecutionPlan`.

## The scenarios

| scenario | workload | placement in the plan | memory lines | traffic / bounds | status |
|---|---|---|---|---|---|
| **A. Fully resident, low latency** | `serve, phase=decode, concurrency=1` | experts: device (1.0) | stack bytes (`derived`, the same e4b classes) + dense + KV(context) | none | **planned** (v1): e4b `estimate_serve_footprint`, KV pool from `paged_kv_pool_bytes` over `MoETopology.kv_*` |
| **B. Partially resident experts** | same, `context_len` set | experts: device k per layer (hot set) + host the rest | device = k/E of the slab + slot rows; host = pinned rest | link bytes per token = E[unique cold experts touched] × bytes per expert (`expected_weight_reads`) | **planned** with uniform routing (solver tiers, e4b#1115; `vram_gb`/`dram_gb` sized by the planner); a measured hot set needs a routing profile |
| **C. Zero residency, transfer on demand** | same | experts: host (ColdTier / pipelined slots), dense: device | device = slot rows only | link bytes per token = all touched experts; lower bound = bytes ÷ link | **memory planned** (`vram_gb=0` puts every expert in DRAM/NVMe); no per-token transfer bound yet |
| **D. Batched throughput** | `concurrency=B` (continuous batching) | same axes as A–C | KV × B × context dominates | reads per step amortized by `1-(1-p)^B`; throughput bound = link ÷ amortized bytes | representable; the objective becomes `throughput` |

## What the pressure test changed now

**1. Traffic is a first-class plan quantity.**
- Before this test the plan accounted memory only.
- Memory lines now carry `where="link"` (bytes moved host to device per micro-batch). e4b's estimate emits the
  offload staging bytes (each layer twice per micro-batch: forward + checkpoint recompute).
- The planner turns traffic into a **lower bound** on step time, using the measured link bandwidth when a receipt
  has one, else the PCIe theoretical ceiling, labelled `inferred`.
- `Constraints.target_s_per_step` refuses setups whose traffic alone rules the target out. This is exactly
  scenario C's "possible but cannot meet the target", and it uses only a bound, so it never refuses a setup that
  could have met the target.

**2. Setup fields are opaque to the planner.**
- `Candidate.setup` is a dict, and ranking is a backend function. A serving backend adds its own fields (for
  example `hot_per_layer`, `kv_dtype`, `decode_route`) without touching the planner.
- The leaks were `Candidate.label()` and the planner's "why" text, which read training field names. Serving v1
  moved both into the backend (`label(setup)`, `explain(...)`). Three more moved later: slack matching uses the
  backend's `SLACK_KEYS` only, and the PCIe-width warning and the "allow host-backed experts" suggestion are the
  backend's `plan_warnings` and `relaxed_candidates`. The planner still names setup fields in one place: the serve
  suggestion's context/concurrency search (`max_tokens_per_seq`, `max_seqs`).

**3. Multi-GPU is not assumed away.** `HardwareProfile.gpus` is a list and `Constraints.device` an index. A
multi-GPU plan would add per-device budgets; nothing assumes there is exactly one.

## What does not fit yet (deferred, stated)

- **The first-chunk prefill graph's pool** (on by default in experts4bit-qlora 0.47.0, +3.3 GiB at 30B) is not
  priced. Plans set `prefill_graph=0` unless the caller fixes it.
- **Decode graphs below sm_89.** They need the fused FP8 KV append, which Triton cannot compile there (found on the
  A2000). Plans offer graphs only where experts4bit-qlora's `fused_append_unsupported` says they run (e4b#1090).

- **Performance prediction.** Latency and throughput targets beyond the transfer bound need a calibrated model.
  e4b's serving harness (`bench/hybrid-g9/step_decomp.py`) and gnf4's `cold_deadline.Costs` are the measured
  inputs. Receipts with `link_h2d_gbps` and step times are the planner's first calibration data.
- **Routing popularity** (scenario B's hot set) needs a profile (`E4B_EXPERT_PROFILE`). Without one,
  `solve_placement` declares "uniform-assumed", and a serving plan must do the same.
- **Prefill vs decode** split the workload. Prefill is compute-bound (grouped GEMM), decode bandwidth-bound
  (GEMV). A plan for both phases is two candidates sharing one placement.
- **NVMe tier.** It is a third `where` (`nvme`) with its own link, and the arena row size from
  `nvme_arena.load_index`.
