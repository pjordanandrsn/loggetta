# Architecture

This document was derived from the code of experts4bit-qlora (e4b, v0.45.0, origin/main `7b3b6aa5`) and
grouped-nf4-gemm (gnf4, v0.37.0, origin/main `c6455de`), inspected on 2026-10-04. The code and the tests are the
source of truth; this file says where things live and why.

## 1. What the two packages already were (from the code, not the READMEs)

**grouped-nf4-gemm (mechanism: kernels and residency primitives).**
- **Kernels:**
  - grouped NF4/MXFP4 GEMM and its dgrad (`nf4_grouped`, `nf4_qlora`, `nf4_route`, `mxfp4_grouped`);
  - int4 decode GEMV/GEMM (`int4_b32`, `int4_smallm`, `nf4_smallm`);
  - FP8 paged attention and KV appends;
  - decode glue kernels;
  - pack/reference ops (`*_pack_ref`, `gptq_pack`).
- **Host/NVMe primitives:** arena bake/read (`nvme_*`), the cold tier (`ColdTier`), VRAM slot and row caches,
  the CPU kernels (`cpu_grouped`, `gnf4_native`).
- **Cost and sizing:** a time-to-contribution cost model (`cold_deadline`) and pinned-memory cost
  (`pinned_request_cost`).
- **No upward dependency:** it never imports e4b in shipped code.
- **Family knowledge it carries anyway:** Kimi/gpt-oss knowledge in `moonshot_gather`, `arena_moe_patch`,
  `mxfp4_*`, and Qwen3-shape-tuned dispatch tables.
- **Policy is in env vars:** about 25 `GNF4_*`/`NF4_QLORA_*` knobs. Capability floors (sm_80) were prose only.

**experts4bit-qlora (runtime: models, training, serving, residency integration).**
- **Model loading:** the streaming 4-bit loader, `MoEConvention` storage conventions for 14 layouts, admission
  gates.
- **Adapters and training:** `ExpertsLoRA`, the training engines (`fast`, `batched`, `offload`, `nvme_train`,
  `hybrid_train`, `moe_keep`).
- **Serving:** `serve`, `serve_paged` with a continuous scheduler, hot/pipelined/hybrid/nvme residency, the int4
  serving lanes.
- **Placement:** a measured-bandwidth placement solver (`engines/placement.py`).
- **It imports gnf4 only inside functions** (about 95 sites), so `[fast]` is truly optional.
- **What it did not have:**
  - no topology object;
  - no memory estimate before load;
  - no capability query on the device;
  - about 97 `E4B_*` plus about 30 bare env vars as configuration, some read on every forward;
  - the training CLI (`train.py`) never reaches the grouped kernel;
  - the only automatic memory decision was halving `TOKEN_BUDGET` after an OOM.

**Already right, and kept.** The two packages share one byte-identical `docs/system-manifest.json` that states
the ownership split and the dependency direction (`experts4bit-qlora -> grouped-nf4-gemm`). It leaves
`umbrella_name: null`, "under consideration".

## 2. Layers and dependency direction

```
            planner layer (this repo; private, renameable)
     hardware inventory · plan types · planner policy · runtime/receipts · CLI
                 │  imports (lazily, only to plan or run)
                 ▼
     experts4bit-qlora  (model-family layer + training/serving runtime)
     arch/topology · recipe · loader · engines · serve_paged
                 │  imports inside functions ([fast] extra)
                 ▼
     grouped-nf4-gemm   (kernels + residency primitives)
                 │
                GPU / host RAM / NVMe
```

**The diagram in the request was wrong in one respect.**
- The request had serving using the kernel package directly, beside the training runtime.
- In the code, serving lives in e4b (`serve_paged`, the residency engines), because serving needs what e4b owns:
  model loading, the expert stores and the KV engines.
- A *future* serving backend that does not use e4b could use gnf4 directly. Nothing here prevents that: the
  planner never assumes a backend is e4b. But no such backend exists today, and inventing one would be
  scaffolding.

**There are no cycles.**
- gnf4 does not know e4b.
- Neither lower package knows the planner. Their new APIs are named for what they do (`describe_moe`,
  `estimate_qlora_footprint`, `route_for`) and documented without reference to any consumer.

## 3. What belongs where (the rule: knowledge lives with its owner)

| knowledge | lives in | why there |
|---|---|---|
| which layers carry experts, stack shapes, dense params, attention projections | e4b `arch/topology.describe_moe` | e4b owns model loading and family conventions. It is built from the same meta tree and expert path the loader uses, so a family the loader admits is described with no per-family code |
| whether a config is loadable at all | e4b `loader.check_admission` / `admission_refusal` | the loader's own gates, now callable before loading |
| bytes of a frozen NF4 stack, adapter counts, optimizer state | e4b `recipe.estimate_qlora_footprint` | sized by constructing e4b's own `Experts4bit`/`ExpertsLoRA` on meta, so the arithmetic cannot drift from the classes |
| which setups are invalid for a model (biased experts, non-NF4 with the grouped kernel, …) | e4b `recipe.setup_refusals` | structural facts about e4b's mechanisms |
| bytes the paged server holds (expert stacks, dense weights, the FP8 paged KV pool) and the env that builds it | e4b `serve_recipe.estimate_serve_footprint`, `paged_kv_pool_bytes`, `ServeSetup.to_env` | the pool size is `Fp8PagedKV`'s own arithmetic (tested equal to a constructed pool); the KV geometry is read by the server's own `_kv_geometry` / `kv_layers` |
| building exactly the priced setup | e4b `recipe.prepare_qlora_training` | one call for the documented fast path, asserting every engine engaged |
| which training kernel route a device gets; the sm_80 floor | gnf4 `nf4_route.route_for`, `MIN_CAPABILITY` | the kernel package's own dispatch rule, now a pure function |
| pinned host-memory cost | gnf4 `pinned_request_cost` (#71) | the allocator rounding is measured there; e4b's estimate applies the same rule |
| GPU/host inventory with provenance | planner `hardware` | machine knowledge, not model or kernel knowledge |
| budgets, headroom, runtime overhead, candidate ordering, refusal and suggestions | planner `planner` | **policy** |
| plan / receipt formats | planner `plan`, `runtime` | the artifacts this layer exists to produce |

**Rejected abstractions.**
- No plugin framework. `backends/__init__.py` is a one-element tuple.
- No `KernelProvider` interface. The one kernel question the planner needs ("can this device train through the
  grouped kernel, by which route?") is one function in the kernel package.
- No YAML config. Plans are typed dataclasses, and the CLI maps flags onto them.
- No topology type in the planner. It uses e4b's `MoETopology` directly rather than mirroring it.

## 4. Interfaces between the layers (all that exist)

**Kernel → runtime (new):**
```python
nf4_route.route_for(capability, *, has_grouped_mm, requested="auto") -> (route | None, reason)
```

**Runtime → planner (new):**
```python
describe_moe(model, *, revision=None, trust_remote_code=False) -> MoETopology   # config + meta tree, no weights
setup_refusals(topology, QLoRASetup) -> tuple[str]                             # words, not exceptions
estimate_qlora_footprint(topology, QLoRASetup, *, tokens_per_microbatch, optimizer) -> Footprint
prepare_qlora_training(model_id, QLoRASetup, *, device, revision) -> PreparedQLoRA
estimate_serve_footprint(topology, ServeSetup) -> Footprint                    # paged server, all-VRAM only (e4b#1080)
ServeSetup.to_env() -> {"E4B_PAGED_*": str}                                    # what serve_paged reads back
```

**Backend contract inside the planner** (`backends/experts4bit.py`): `WORKLOADS`, `KERNELS_FOR`, `probe(gpu)`,
`candidates(...)`, `estimate(...)`, `speed_rank(setup)`, `label(setup)`, `explain(...)`, `policy_notes(...)`,
`describe_kernel(...)`, plus an executor for training (`experts4bit_train.run`; serve plans are planned only). A
second backend implements the same functions; generalize into a protocol only then.

## 5. The plan

`plan(topology, hardware, workload, constraints) -> ExecutionPlan`. It is pure and deterministic: the same inputs
give byte-identical JSON. It loads no weights and touches no device.

**What a plan contains:**
- the model identity and topology summary;
- the hardware facts each with its source;
- the workload and the constraints;
- the device and host budgets, and where each came from;
- the selected candidate's setup and its memory lines, each `derived` / `heuristic` / `measured` / `inferred`;
- what the estimate leaves out;
- every alternative with the reason it lost;
- the reasoning, with the evidence for the speed ordering as claim IDs;
- warnings;
- a performance section that says plainly it is **not predicted** (there is no calibrated performance model).

**Refusal is a status.** A refused plan carries:
- the reasons, per constraint;
- the closest candidate's itemized memory;
- suggestions, each computed rather than templated: the extra VRAM needed, the extra RAM needed, "allow
  host-backed experts" (only if that would fit), and "reduce tokens per micro-batch to N" (found by search).

**Three levels of control:**
- Automatic: `plan MODEL`.
- Directed: budgets, allowed expert residency, objective `speed` / `min_vram` / `min_ram`.
- Expert: `--fix FIELD=VALUE` pins any `QLoRASetup` field, and the planner only varies the rest.

## 6. Runtime, observation, provenance

`execute(plan)` refuses a refused plan, then builds the setup with `prepare_qlora_training` and runs a fixed-shape
loop (packed alpaca blocks, AdamW or AdamW8bit, clip 1.0). It writes an `execution-receipt/1` with:

- **measured memory:**
  - allocator peak and reserved peak;
  - the driver-reported process peak, sampled on a thread;
  - CUDA context = driver peak − reserved peak;
  - host baseline and VmHWM;
- **timing:** load time, per-step times, median over the steps after warm-up, tokens/s;
- **correctness:**
  - losses, finite, and whether the last third is below the first;
  - sha256 of frozen expert bytes before and after (must match), including host homes under offload;
  - whether the LoRA-B norm moved;
- **engagement:** what the recipe actually enabled (fused modules, the kernel route, rope/rmsnorm flags read from
  env);
- **comparison:** `measured − estimated`, both numbers present, for the allocator, driver and host views;
- **provenance:** the git commit and dirty flag of each imported source tree, distribution versions, argv, the
  plan.

Receipts are also the planner's memory: `--observations DIR` replaces the inferred CUDA-context default with the
measured value for the same GPU and driver, and notes an earlier measured peak for an identical setup.

(The measured results of the first slice are in section 8.)

## 7. Adding things

**A new model family.**
- If e4b's loader admits it (a `SUPPORTED_ARCHITECTURES` row or a convention), it is already described, priced
  and planned.
- If not, the work is in e4b's `arch/` and `formats/` (the README-LAYOUT rule), and `admission_refusal` says why
  until then.
- Nothing in the planner changes.

**A new backend.** Add `backends/<name>.py` with the five functions above and an executor, list it in
`BACKENDS`, and give it a `WORKLOADS` tuple. The planner's selection, refusal and rendering are backend-agnostic.

**A new kernel** behind e4b is e4b's business: it shows up as a `QLoRASetup.expert_kernel` value with a
`setup_refusals` rule and a capability probe answered by the kernel package.

## 8. Measured (filled from receipts)

_See `docs/RESULTS.md`._

## 9. Renameability

**What a rename touches:**
- the package directory;
- `pyproject.toml` (name, script);
- the README title and this repo's name.

**What a rename does not touch:**
- No serialized format names the project: `execution-plan/1`, `execution-receipt/1`, `estimate-validation/1`.
- No env var is read.
- No protocol string exists.
- The lower packages' new APIs do not mention it, and neither do their changelogs or PRs.

`tests/test_renameable.py` fails if a code file spells the package name, a schema carries it, or an env var
appears.
