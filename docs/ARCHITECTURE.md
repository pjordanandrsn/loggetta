# Architecture

**Loggetta's primary output is an `ExecutionPlan`.**
- The planner is pure policy. It determines what should run.
- Execution is delegated to the backend the plan names.
- Loggetta's execution layer is orchestration around that backend call, plus construction of the resulting
  `ExecutionReceipt`.
- Later plans read the receipt as evidence.

```
   workload + machine + constraints + evidence (earlier receipts)
                          │
                          ▼
                       Planner            pure, deterministic; loads no weights, touches no device
                          │
                          ▼
                    ExecutionPlan         the product: selection, estimates with their basis, what lost, refusal
                          │  execute(plan): check, dispatch, record
                          ▼
     backend: experts4bit-qlora → grouped-nf4-gemm      how it runs: loading, adapters, engines, kernels
                          │
                          ▼
                  ExecutionReceipt ──── measured feedback ────► Planner
```

The code and the tests are the source of truth; this file says where things live and why.
- Section 1 records the two lower packages as they were when this layer was started: experts4bit-qlora (e4b)
  v0.45.0, origin/main `7b3b6aa5`, and grouped-nf4-gemm (gnf4) v0.37.0, origin/main `c6455de`, inspected on
  2026-10-04.
- The interfaces in section 4 are released in e4b 0.48.0 and gnf4 0.41.0. The serve estimate's later items are in
  e4b's next release.

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
     Loggetta (this repo): what should execute
     hardware · planner · plan (ExecutionPlan) · execution + measure (ExecutionReceipt) · cli
     backends/experts4bit*.py: the questions it asks e4b, and the handoff of a plan to e4b
                 │  imports (lazily, only to plan or run)
                 ▼
     experts4bit-qlora  (how: model-family layer + training/serving runtime)
     arch/topology · recipe · serve_recipe · loader · engines · serve_paged
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
- Neither lower package imports the planner. Their new APIs are named for what they do (`describe_moe`,
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
| which measured evidence a plan trusts, and for what (same GPU, setup, model, workload kind) | planner `planner` | **policy**: the feedback half of the loop |
| plan / receipt formats | planner `plan`, `execution` | the artifacts this layer exists to produce |
| checking a plan may run here, dispatching it to its backend, writing the receipt | planner `execution` | orchestration around one backend call, not mechanism |
| running a training plan: building the model, adapters, engines and kernels | e4b `prepare_qlora_training`, called by the backend's executor | e4b knows how; the executor adds only the measured loop and integrity checks |

**Rejected abstractions.**
- No plugin framework. `backends/__init__.py` is a one-element tuple.
- No `KernelProvider` interface. The one kernel question the planner needs ("can this device train through the
  grouped kernel, by which route?") is one function in the kernel package.
- No YAML config. Plans are typed dataclasses, and the CLI maps flags onto them.
- No topology type in the planner. It uses e4b's `MoETopology` directly rather than mirroring it.

## 4. Interfaces between the layers (all that exist)

**gnf4 → e4b (new):**
```python
nf4_route.route_for(capability, *, has_grouped_mm, requested="auto") -> (route | None, reason)
```

**e4b → Loggetta (new):**
```python
describe_moe(model, *, revision=None, trust_remote_code=False) -> MoETopology   # config + meta tree, no weights
setup_refusals(topology, QLoRASetup) -> tuple[str]                             # words, not exceptions
estimate_qlora_footprint(topology, QLoRASetup, *, tokens_per_microbatch, optimizer) -> Footprint
prepare_qlora_training(model_id, QLoRASetup, *, device, revision) -> PreparedQLoRA   # the run: e4b builds the setup
estimate_serve_footprint(topology, ServeSetup) -> Footprint                    # paged server: all-VRAM and the solver's tiers
min_hot_rows(topology, ServeSetup) -> int                                      # cold tier's floor (next e4b release)
fused_append_unsupported(capability) -> str | None                             # where decode graphs cannot run
ServeSetup.to_env() -> {"E4B_PAGED_*": str}                                    # what serve_paged reads back
```

**Backend contract inside the planner** (`backends/experts4bit.py`):
- the model: `describe(model, ...)` returns the backend's own description (e4b's `MoETopology` here; the planner
  never reads its fields), `refusal(topology)` says why the backend can plan nothing for it (or `None`), and
  `summary(topology)` is the plan's `model` section;
- questions: `WORKLOADS`, `SLACK_KEYS`, `KERNELS_FOR`, `probe(gpu)`, `candidates(...)`, `estimate(...)`,
  `speed_rank(setup)`, `label(setup)`, `explain(...)`, `policy_notes(...)`, `describe_kernel(...)`;
- about a setup: `residency(setup)` (where its frozen weights live, for reading receipts), `plan_warnings(setup,
  gpu)`, and `relaxed_candidates(...)` (setups outside the caller's constraints, with the words for the change, which
  the planner prices into a refusal's suggestions);
- for serving: `fill_knobs`, `resolve`, `SLACK_DEFAULTS`;
- the handoff: `executor(kind)` returns the function that runs a feasible plan of that workload kind, or `None` when
  the kind is planned only (serve, today); `run_tag(setup)` names the setup in a receipt's run id.

A second backend implements the same functions; generalize into a protocol only then.
`tests/test_planner_second_backend.py` is such a backend in pure Python (its own topology type and setup fields): the
planner plans, refuses, suggests, warns and learns from receipts through it without reading a setup field.

## 5. The plan

`plan(topology, hardware, workload, constraints, observations=receipts) -> ExecutionPlan`. The plan is the
product, not an internal return value: it is rendered, saved (`plan --out`), reviewed, and executed as a file
(`execute PLAN.json`).

It is pure and deterministic: the same inputs give byte-identical JSON. It loads no weights and touches no device.
Model facts come from the backends: `describe_model` asks each one's `describe` in order and returns the first
description its backend can plan for (today e4b's `describe_moe`). The planner has no topology type of its own, and a
model is refused only when every backend refuses it, each with its own reason.

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
- Expert: `--fix FIELD=VALUE` pins any backend setup field (`QLoRASetup` or `ServeSetup`), and the planner only
  varies the rest.

## 6. Execution, the receipt, and feedback

`execute(plan)` (module `execution`) is orchestration. In order, it:

1. refuses a refused plan, with the refusal's reasons;
2. finds the backend the plan selected, and that backend's executor for the workload kind. Serve plans have none
   yet: their *Why* carries the server's environment, and `bench/serve_validate.py` starts the server from it;
3. refuses a plan made for another GPU (`check_here`, by UUID, else by name). Its receipt would name the wrong card,
   and later plans would learn this card's overheads under that card's name;
4. calls the executor with the plan;
5. wraps what comes back into an `execution-receipt/1`, written beside earlier ones.

Nothing is loaded before step 4, and nothing in this module knows how a model runs.

**The experts4bit executor** (`backends/experts4bit_train.py`) has `prepare_qlora_training` build the model from
exactly the selected setup. Loading, the 4-bit stores, adapters, the fused engines, offload and kernels are e4b's.
Around it, the executor adds only the measured loop:
- packed alpaca blocks of fixed shape;
- AdamW or AdamW8bit, clip 1.0;
- timing, memory sampling, and the integrity checks.

**The receipt carries:**

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
- **provenance:** the git commit and dirty flag of each imported source tree (taken at process start), distribution
  versions, argv;
- **the plan:** the whole `ExecutionPlan` that produced it, so every comparison can be traced to the estimate it
  checks.

**Receipts are the planner's evidence** (`--observations DIR`, `load_observations`). Each lookup says which receipt
it used, and with what basis:
- the CUDA context and host baseline for the same GPU and driver;
- allocator reserve slack, matched by setup key, model, GPU and workload kind. Where none matches, it is transferred
  through an anchor model (heuristic);
- the measured host-to-device bandwidth;
- for serving, this model's allocator residual and host growth;
- an earlier measured peak for an identical setup.

A receipt of one workload kind never informs the other: a server's allocation pattern says nothing about a
trainer's.

(The measured results of the first slice are in section 8.)

## 7. Adding things

**A new model family.**
- If e4b's loader admits it (a `SUPPORTED_ARCHITECTURES` row or a convention), it is already described, priced
  and planned.
- If not, the work is in e4b's `arch/` and `formats/` (the README-LAYOUT rule), and `admission_refusal` says why
  until then.
- Nothing in the planner changes.

**The dense backend** (`backends/dense.py`, after experts4bit in `BACKENDS`) describes what the MoE loader refuses.
- It owns that description, because transformers is its loader: the config, plus the module tree on `meta`.
- Decoder-layer linears are classified by role from shape against the config's widths (`classify`), so LoRA targets
  are not a per-family name list.
- What structure cannot show comes from the package that knows it: the model class's declared attention
  implementations (transformers) and chunked-loss coverage (experts4bit-qlora's `chunked_lm_loss_refusal`).
- It plans no workload yet (`WORKLOADS = ()`). Its candidates, estimate and executor are the next steps.

**A new backend.** Add `backends/<name>.py` with the contract in section 4, including `executor` and `run_tag`.
List it in `BACKENDS`, and give it a `WORKLOADS` tuple.
- Selection, refusal, rendering and `execute` are backend-agnostic.
- The planner names no training setup field: slack is matched on the backend's `SLACK_KEYS`, a receipt's
  residency is asked of the backend that ran it, and warnings and relaxations are the backend's. The serve
  suggestion's context/concurrency search still reads two `ServeSetup` fields (SERVING-PRESSURE-TEST).

**A new kernel** behind e4b is e4b's business: it shows up as a `QLoRASetup.expert_kernel` value with a
`setup_refusals` rule and a capability probe answered by the kernel package.

## 8. Measured (filled from receipts)

_See `docs/RESULTS.md`._

## 9. Renameability

**What a rename touches:**
- the package directory;
- `pyproject.toml` (name, script);
- the README title and this repo's name;
- the PyPI project (`0.0.1` is reserved under the current name).

**What a rename does not touch:**
- No serialized format names the project: `execution-plan/1`, `execution-receipt/1`, `estimate-validation/1`.
- No env var is read.
- No protocol string exists.
- The lower packages' new APIs do not mention it, and neither do their changelogs or PRs.

`tests/test_renameable.py` fails if a code file spells the package name, a schema carries it, or an env var
appears.


## User data and adapter artifacts (0.2.0)

Optional `Workload.data` and `Workload.learning_rate` fields extend `execution-plan/1`; old plans load with their
original Alpaca demonstration and learning-rate defaults, and serializing those defaults keeps the old wire shape.
Older releases cannot interpret new non-default fields and should be upgraded before executing a new plan.

Dataset preparation is separate from pure planning. Execution validates and packs the requested token budget into
a temporary memory-mapped file before loading model weights. The runtime-specific executor calls the existing
QLoRA recipe, trains its adapters, then exports them. `execution.execute` chooses the artifact destination and
records returned artifact metadata in the run report; it does not know model tensor layouts.

The native adapter serializer lives in `backends/experts4bit_adapters.py`. The runtime's structural LoRA discovery
identifies trainable adapter tensors and the runtime's recipe rebuilds the recorded setup. Serialization uses
safetensors plus a manifest and tokenizer files. No full model state_dict, frozen base weights, optimizer state,
or private training-example text is included. The public `load_adapter` function delegates to that backend helper.
It is a native artifact, not a PEFT conversion or an exact training-resume checkpoint.

Exports never overwrite an existing directory. Export errors mark the report `SAVE_FAILED`; integrity failures
suppress adapter export. The data record identifies the actual token stream and explicitly labels full-sequence
loss, concatenated packing, shuffling and repetition. GPU throughput claims still require measured GPU runs.
