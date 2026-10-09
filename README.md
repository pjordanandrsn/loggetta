<div align="center">

# Loggetta

### Large models. The hardware you have.

**Plan the run. Train the model. Keep the evidence.**

[![PyPI](https://img.shields.io/pypi/v/loggetta)](https://pypi.org/project/loggetta/)
[![CI](https://github.com/pjordanandrsn/loggetta/actions/workflows/ci.yml/badge.svg)](https://github.com/pjordanandrsn/loggetta/actions/workflows/ci.yml)

[Get started](#get-started) · [Train on your data](#train-on-your-data) · [Measured results](#measured-results) · [Docs](#documentation) · [Hugging Face](https://huggingface.co/spaces/pjordanandrsn/research)

</div>

Loggetta checks your hardware, chooses a supported configuration for a Mixture-of-Experts (MoE) model,
and runs QLoRA fine-tuning. It saves the plan and a JSON run report with memory use, timing, and checks
that the selected optimizations actually ran.

**One install includes the runtime and GPU kernels.**

## Get started

```bash
pip install loggetta
loggetta inspect
loggetta plan Qwen/Qwen3-30B-A3B --seq 2048
```

The plan shows where weights will live, estimated memory use, and why alternatives were rejected.
It checks the budget before downloading model weights. Estimates can miss; they are not an out-of-memory guarantee.

**GPU training needs Linux, a supported NVIDIA CUDA GPU, and compatible PyTorch.**
The released training path is single-GPU MoE training. Dense execution still requires `--allow-development-executor`:
its memory estimate has not yet passed a capacity reading. See [Dense training](docs/DENSE.md). Serving placement can be planned; server launch uses the runtime separately.

## New in 0.4.0

- **Dense models: planned, not yet supported for training.** Dense plans are estimates: they held on the runs
  they were fitted to but missed a held-out check, and a further reading is pending. Dense execution needs
  `--allow-development-executor` until a 24 GB capacity reading passes. See [Dense training](docs/DENSE.md).
- **Registered dense runs can opt into a separate reserve estimate** (`dense_reserve_policy="dq10"`). Defaults do not change.
- **MoE training estimates price the `grouped_nf4` backward pass.** OLMoE estimates on an RTX A2000 were about 0.2 GiB short;
  in sample they now cover the measured peak. Near a budget, a plan may pick the reference kernel or host residency
  where it picked resident `grouped_nf4` before.
- **Single-stream serve plans** name the speed-ups a default server runs for the model's family, and quote a decode
  speed only from a measured run of the same setup.
- **Fix:** resident `grouped_nf4` training runs again with experts4bit-qlora 0.49.0 or later. Loggetta 0.3.x stopped
  before the first step ([#47](https://github.com/pjordanandrsn/loggetta/issues/47)); the workaround was `E4B_ABSMAX_DQ=0`.

Full list: [CHANGELOG](CHANGELOG.md).

## Train on your data

Custom datasets and reusable adapters arrived in 0.3.0:

```bash
pip install -U loggetta
loggetta train Qwen/Qwen3-30B-A3B \
  --dataset ./data/train.jsonl --format text \
  --seq 512 --micro-batch 1 --steps 20 --seed 42 \
  --out runs/my-training --adapter-out adapters/my-adapter
```

Use a JSONL file with a `text` field and enough tokens for the run. Local JSON, CSV, Parquet and TXT files,
Hub datasets, Alpaca instructions and text-only chats are also supported.

You keep **adapter tensors, tokenizer files, a manifest, and a run report**. Data is validated and tokenized before
weights load. A plan that would read the dataset more than once is refused unless you pass `--epochs N` or
`--repeat-data`; existing adapter directories are never overwritten.

```python
from loggetta import load_adapter

model = load_adapter("adapters/my-adapter", device="cuda")
```

By default, chat data trains only the assistant turns and Alpaca data only the response; plain text and `--loss all`
score every token. Adapters use the native runtime format, not PEFT or
optimizer-resume checkpoints. Save/reload passes real-adapter CPU tests; the full CUDA integration test remains unverified.
[Training and reload guide](https://github.com/pjordanandrsn/loggetta/blob/main/docs/TRAINING.md)

## Keep control of the run

| Need | Command or option |
| :--- | :--- |
| Set a memory budget | `--vram 12 --ram 64` (GiB) |
| Choose where experts live | `--experts device` or `--experts host` |
| Save a plan | `loggetta plan MODEL --out plan.json` |
| Run that plan | `loggetta execute plan.json --out runs/` |
| Reuse measurements | `loggetta plan MODEL --observations runs/` |
| Plan serving placement | `loggetta plan MODEL --workload serve --context 4096 --concurrency 1` |

`train` combines planning and execution. Dense training can be planned and executed with the development opt-in.
Multi-GPU execution, server launch and throughput prediction remain outside the current command surface.

## Measured results

The included **experts4bit-qlora runtime and grouped-nf4-gemm kernels** do the compute.
Their speed results below come from matched training runs, not planner benchmarks.

| Workload | Result |
| :--- | :--- |
| **Qwen3-30B-A3B QLoRA · RTX 5090** | Unsloth spends **1.92×** e4b's GPU time per step, and **2.80×** its wall-clock time on an AMD EPYC 7713 host. Comparable held-out loss; Unsloth peaked lower (24.27 vs 26.16 GB). [Result](https://github.com/pjordanandrsn/experts4bit-qlora/blob/main/bench/h2h-2026-10-02/tc1/RESULTS-tc1-pos69.md) |
| **Why two numbers** | GPU time doesn't depend on the host. Unsloth runs about 14× e4b's CPU operations per step, so its wall-clock time grows on a slower host. Earlier wall-clock readings, before e4b's current defaults: 2.352× and 2.468×. |
| **Planner memory check · Qwen3-30B-A3B · RTX 5090** | After calibration from earlier runs: **24.54 GiB** estimated process peak, **24.34 GiB** measured. One in-sample case, not a guarantee. In the MoE audit re-run with training plans no longer borrowing another model's reserve (`evidence/2026-10-09-moe-plan-vs-driver-no-borrow`), today's planner puts no RTX A2000 training plan under its measured peak, in sample (the replan section; older recorded plans were). [Plan vs run](https://github.com/pjordanandrsn/loggetta/blob/main/docs/RESULTS.md) · [MoE audit](https://github.com/pjordanandrsn/loggetta/blob/main/evidence/2026-10-09-moe-plan-vs-driver-no-borrow/README.md) |

The comparison used torch 2.12.1+cu130 and transformers 5.5.0 for both frameworks, with matched adapters, initialization, and tokens.
Results apply to those workloads and hosts. See the runtime's [current results](https://github.com/pjordanandrsn/experts4bit-qlora/blob/main/docs/STATUS.md) for newer packed-training work.

## Three projects, one install

| Project | Job |
| :--- | :--- |
| **Loggetta** | Inspect, plan, run, and save reports |
| [experts4bit-qlora](https://github.com/pjordanandrsn/experts4bit-qlora) | Load, fine-tune, serve, and offload model weights |
| [grouped-nf4-gemm](https://github.com/pjordanandrsn/grouped-nf4-gemm) | Compute on packed weights and move expert data |

The packages can also be used independently. [Architecture and ownership](https://github.com/pjordanandrsn/loggetta/blob/main/docs/ARCHITECTURE.md)

## Documentation

- [Training and adapter reload](https://github.com/pjordanandrsn/loggetta/blob/main/docs/TRAINING.md)
- [Results and memory-estimate checks](https://github.com/pjordanandrsn/loggetta/blob/main/docs/RESULTS.md)
- [Serving placement tests](https://github.com/pjordanandrsn/loggetta/blob/main/docs/SERVING-PRESSURE-TEST.md)
- [Session report, 2026-10-04/05 (dated; predates dataset training)](https://github.com/pjordanandrsn/loggetta/blob/main/docs/SESSION-REPORT.md)
- [Research, releases, and machine-readable evidence](https://cerinamroth.com/ml/)

<details>
<summary><strong>How this relates to Accelerate</strong></summary>

[Accelerate](https://huggingface.co/docs/accelerate/usage_guides/model_size_estimator) also estimates model memory before downloading weights and provides broader training and distributed infrastructure.
Loggetta focuses on choosing a configuration for its MoE runtime, running it, and feeding measured results into the next plan.
No matched estimator-accuracy comparison has been run.

</details>

<details>
<summary><strong>Development checks</strong></summary>

```bash
pip install -e ".[test]"
pytest tests/
```

CPU tests cover planning and handoff. GPU benchmarks and CUDA adapter integration are separate checks.

</details>
