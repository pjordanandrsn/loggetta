# Loggetta

**One install for planning and running single-GPU MoE QLoRA workloads.**

```bash
pip install loggetta
```

That installs Loggetta plus its current runtime and kernel stack:

- **Loggetta** — hardware inventory, planning, `ExecutionPlan`, execution orchestration, and `ExecutionReceipt`
- **experts4bit-qlora** — model loading, QLoRA, training, serving mechanisms, and CPU/NVMe offload
- **grouped-nf4-gemm** — packed low-bit kernels and residency primitives

Loggetta examines the model, machine, workload, and constraints **before loading model weights**. It chooses a supported setup, explains why alternatives lost, and refuses configurations estimated not to fit. For supported training plans it dispatches into the included runtime and records what actually happened.

```bash
loggetta inspect
loggetta plan Qwen/Qwen3-30B-A3B --seq 2048
loggetta train allenai/OLMoE-1B-7B-0924 --seq 512 --micro-batch 2 --steps 12 --out receipts/
```

A feasible plan is an estimate, not an OOM guarantee. Serving placement can be planned; Loggetta does not launch the server yet.

## Plan, run, measure

A plan is a serializable artifact, not just terminal output. It records the selected backend setup, budgets, memory estimates, rejected alternatives, warnings, and anything the planner does not model. You can save it, inspect it, then execute that exact decision:

```bash
loggetta plan allenai/OLMoE-1B-7B-0924 \
  --seq 512 --micro-batch 2 --steps 12 --out plan.json

loggetta execute plan.json --out receipts/
```

The resulting receipt records the plan plus measured allocator/reserved/driver memory, timing, correctness checks, and runtime provenance. Earlier receipts can be passed back as observations so later plans use measurements from the same model/setup when available:

```bash
loggetta plan allenai/OLMoE-1B-7B-0924 \
  --seq 512 --micro-batch 2 --observations receipts/
```

The planner does not load weights while choosing. Hardware probing and model topology discovery happen first; candidate selection is deterministic policy over those facts and constraints.

## Measured speed

These are measurements of the **runtime/kernel stack that Loggetta installs and dispatches into**, not speedups caused by the planner.

| Result | Measured comparison |
| :--- | :--- |
| **2.352x faster/step vs Unsloth** | Qwen3-30B-A3B QLoRA on one RTX 5090, matched adapters/init/tokens and the same torch 2.12.1+cu130 / transformers 5.5.0 stack: **3.494 vs 8.218 s/step**. Held-out loss was COMPARABLE. Unsloth used less peak VRAM: **24.27 vs 27.49 GB**. |
| **2.468x replication** | Same-stack Qwen3 comparison on a second RTX 5090 host. The registered position remains 2.352x; the replication is reported separately. |
| **2.775x vs Axolotl** | Matched-work Qwen3-30B-A3B comparison on an RTX 5090 / Ryzen 9 9950X3D host: **2.147 vs 5.956 s/step**. A separate EPYC-host reading was **1.979x**, so this ratio is host-sensitive. |
| **2.33x packed-compute throughput** | H100 synthetic expert-offload pipeline vs bitsandbytes CUDA dequantization + cuBLAS: **6.466 vs 2.773 pipeline tok/s**, **26.8 vs 59.1 J/token**. This is not an end-to-end serving claim. |

Evidence: [Unsloth same-stack](https://github.com/pjordanandrsn/experts4bit-qlora/blob/main/bench/h2h-2026-10-02/tc1/RESULTS-tc1-samestack-box4.md) · [second-host replication](https://github.com/pjordanandrsn/experts4bit-qlora/blob/main/changelog.d/tc1-amendment-42-read-samestack-host2.md) · [Axolotl/Unsloth matched work](https://github.com/pjordanandrsn/experts4bit-qlora/blob/main/bench/h2h-2026-10-02/tc1/RESULTS-tc1-matched19.md) · [H100 pipeline](https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/main/bench/phase3/flagship/RESULTS-flagship-bnb-baseline.md)

## Models: supported vs tested

**Runtime-supported** means the included `experts4bit-qlora` fast-training path is evidence-gated supported with a real-weight PASS receipt. The Loggetta column says what this repository itself has exercised.

| Model / family | Included runtime QLoRA | Loggetta test status |
| :--- | :---: | :--- |
| **OLMoE-1B-7B-0924** (`olmoe`) | Supported | **Run: training + serving** |
| **Qwen3-30B-A3B** (`qwen3_moe`) | Supported | Planner + serving validated; backend training receipts imported for calibration |
| **Granite-3.1-3B-A800M** (`granitemoe`) | Supported | **Run: training + serving** |
| **Granite-4.0-H-tiny** (`granitemoehybrid`) | Supported | **Run: training**; serving refusal validated (Mamba state unsupported by current paged runner) |
| **Qwen3.6-35B-A3B** (`qwen3_5_moe`) | Supported | Planner-tested for training + serving; not yet executed here |
| **LFM2-8B-A1B** (`lfm2_moe`) | Supported | Planner-tested; serving refusal validated (conv state unsupported by current paged runner) |
| **Mixtral-8x7B-Instruct-v0.1** (`mixtral`) | Supported | Planner-tested; not yet executed here |
| **ERNIE-4.5-21B-A3B** (`ernie4_5_moe`) | Supported | Planner-tested; not yet executed here |
| **Gemma-4-26B-A4B-it** (`gemma4_text`) | Supported | Runtime-supported; not yet in Loggetta's model sweep |
| **NVIDIA Nemotron-3.5-Lightning-30B-A3B** (`nemotron_h`) | Supported | Runtime-supported; not yet in Loggetta's model sweep |

Also exercised by the planner but **not advertised as supported expert-QLoRA rows**: `DeepSeek-V2-Lite` is planned with attention LoRA disabled because its MLA attention is not described by the current adapter path; `gpt-oss-20b` is deliberately refused for expert QLoRA because its biased/clamped expert structure does not satisfy `ExpertsLoRA`'s contract.

Backend support is evidence-gated and can move independently of Loggetta's own test matrix. See the [runtime capability register](https://github.com/pjordanandrsn/experts4bit-qlora/blob/main/docs/capabilities.json) and Loggetta's [measured results](https://github.com/pjordanandrsn/loggetta/blob/main/docs/RESULTS.md).

## Memory planning

Loggetta labels each estimate as measured, derived, inferred, or heuristic instead of presenting every number as equally certain. An `ExecutionReceipt` records estimated vs measured memory and can be fed back into later plans.

Selected checks:

- A2000 training runs across OLMoE and Granite families put allocator estimates within **0.01-0.21 GiB** of measured peaks.
- Qwen3-30B-A3B on an RTX 5090: allocator **22.09 GiB estimated vs 21.91 GiB measured**; after receipt-calibrated runtime overheads, process peak **24.54 GiB planned vs 24.34 GiB measured**.
- OLMoE serving on an A2000: planned VRAM/DRAM/NVMe expert placement matched the server's **272 / 421 / 331** split; allocator **2.052 GiB planned vs 2.051 GiB measured**.

Loggetta does **not** currently predict throughput. It may use measured evidence to order valid setups, but unknown speed remains unknown.

## Current scope

**Available:** one-command install of the full stack; single-GPU MoE planning; saved `ExecutionPlan` and `ExecutionReceipt`; QLoRA training execution; device/host/storage serving placement planning; measured-memory feedback; explicit refusals.

**Not claimed:** first-class dense-model planning; multi-GPU planning/execution; calibrated throughput prediction; a Loggetta server-launch command; universal exposure of every mechanism in the lower packages.

GPU execution currently targets **Linux + NVIDIA CUDA** and requires a compatible driver/PyTorch environment. The package is pre-1.0.

## More detail

The PyPI page is intentionally short. The GitHub repository carries the architecture, full benchmark caveats, receipts, family sweeps, and reproducibility notes:

- [Full README](https://github.com/pjordanandrsn/loggetta)
- [Results](https://github.com/pjordanandrsn/loggetta/blob/main/docs/RESULTS.md)
- [Architecture](https://github.com/pjordanandrsn/loggetta/blob/main/docs/ARCHITECTURE.md)
- [experts4bit-qlora capabilities](https://github.com/pjordanandrsn/experts4bit-qlora/blob/main/docs/capabilities.json)
