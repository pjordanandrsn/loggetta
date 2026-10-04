# Serving pressure test

**Status (second pass).** Scenario A is planned now: the paged server's all-VRAM placement, context × concurrency,
with its FP8 paged KV pool priced by the server's own arithmetic (see "Serving v1" below). Scenarios B–D are still
refused, in words: the solver's tiered placement is "not priced yet".

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

So serving *mechanism* is mature. What is missing is the **policy layer**: a way to choose among those
mechanisms before loading, and to say why.

## The scenarios

| scenario | workload | placement in the plan | memory lines | traffic / bounds | status |
|---|---|---|---|---|---|
| **A. Fully resident, low latency** | `serve, phase=decode, concurrency=1` | experts: device (1.0) | stack bytes (`derived`, the same e4b classes) + dense + KV(context) | none | **planned** (v1): e4b `estimate_serve_footprint`, KV pool from `paged_kv_pool_bytes` over `MoETopology.kv_*` |
| **B. Partially resident experts** | same, `context_len` set | experts: device k per layer (hot set) + host the rest | device = k/E of the slab + slot rows; host = pinned rest | link bytes per token = E[unique cold experts touched] × bytes per expert (`expected_weight_reads`) | representable. A setup field `hot_per_layer` replaces the training string `expert_residency` |
| **C. Zero residency, transfer on demand** | same | experts: host (ColdTier / pipelined slots), dense: device | device = slot rows only | link bytes per token = all touched experts; lower bound = bytes ÷ link | **the transfer bound added this session already covers it**; refusal on `target_s_per_step` generalizes to a per-token target |
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
  moved both into the backend (`label(setup)`, `explain(...)`). The planner still names setup fields in four
  places: receipt matching for reserve slack (`expert_residency`, `expert_kernel`), the PCIe-width warning, the
  "allow host-backed experts" suggestion, and the serve suggestion's context/concurrency search. Each is a small
  leak a third backend would have to match or move.

**3. Multi-GPU is not assumed away.** `HardwareProfile.gpus` is a list and `Constraints.device` an index. A
multi-GPU plan would add per-device budgets; nothing assumes there is exactly one.

## What does not fit yet (deferred, stated)

- **Performance prediction.** Latency and throughput targets beyond the transfer bound need a calibrated model.
  e4b's serving harness (`bench/hybrid-g9/step_decomp.py`) and gnf4's `cold_deadline.Costs` are the measured
  inputs. Receipts with `link_h2d_gbps` and step times are the planner's first calibration data.
- **Routing popularity** (scenario B's hot set) needs a profile (`E4B_EXPERT_PROFILE`). Without one,
  `solve_placement` declares "uniform-assumed", and a serving plan must do the same.
- **Prefill vs decode** split the workload. Prefill is compute-bound (grouped GEMM), decode bandwidth-bound
  (GEMV). A plan for both phases is two candidates sharing one placement.
- **NVMe tier.** It is a third `where` (`nvme`) with its own link, and the arena row size from
  `nvme_arena.load_index`.
