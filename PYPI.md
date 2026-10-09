# Loggetta

**Plan the run. Train the model. Keep the evidence.**

Loggetta checks your hardware, chooses a supported configuration for a Mixture-of-Experts (MoE) model, and runs QLoRA
fine-tuning. It saves the plan and a JSON run report with memory use, timing, and checks that the selected
optimizations actually ran. One install includes the runtime
([experts4bit-qlora](https://pypi.org/project/experts4bit-qlora/)) and the GPU kernels
([grouped-nf4-gemm](https://pypi.org/project/grouped-nf4-gemm/)).

```bash
pip install loggetta
loggetta inspect
loggetta plan Qwen/Qwen3-30B-A3B --seq 2048
```

The plan shows where weights will live, estimated memory use, and why alternatives were rejected. It checks the budget
before downloading model weights. Estimates can miss; they are not an out-of-memory guarantee.

## New in 0.4.0

- **Dense models, development-gated.** `plan` and `inspect` handle dense decoder models. Their training runs only with
  `--allow-development-executor`: the dense memory estimate has not passed a capacity reading.
  See [Dense training](https://github.com/pjordanandrsn/loggetta/blob/main/docs/DENSE.md).
- **The DQ10 reserve policy is opt-in,** for registered dense runs. Defaults do not change.
- **MoE training estimates price the `grouped_nf4` backward pass.** OLMoE plans on an RTX A2000 were about 0.2 GiB short;
  in sample they now cover the measured peak. Near a budget, a plan may pick the reference kernel or host residency
  where it picked resident `grouped_nf4` before.
- **Single-stream serve plans** name the speed-ups a default server runs for the model's family, and quote a decode
  speed only from a measured run of the same setup.
<!-- RELEASE: add the absmax digest fix here once it merges. -->

Full list: [CHANGELOG](https://github.com/pjordanandrsn/loggetta/blob/main/CHANGELOG.md).

## Train on your data

```bash
loggetta train Qwen/Qwen3-30B-A3B \
  --dataset ./data/train.jsonl --format text \
  --seq 512 --micro-batch 1 --steps 20 --seed 42 \
  --out runs/my-training --adapter-out adapters/my-adapter
```

Local JSONL, JSON, CSV, Parquet and TXT files, Hub datasets, Alpaca instructions and text-only chats are supported.
Data is validated and tokenized before weights load. Chat data trains only the assistant turns by default. Reload the
adapter with `loggetta.load_adapter("adapters/my-adapter")`. See the
[training guide](https://github.com/pjordanandrsn/loggetta/blob/main/docs/TRAINING.md).

To run a saved decision later: `loggetta plan MODEL --out plan.json`, then `loggetta execute plan.json --out runs/`.
Pass earlier run reports back with `--observations runs/` and later plans use those measurements.

## Measured results

The included runtime and kernels do the compute; these are matched training runs, not planner benchmarks.

| Workload | Result |
| :--- | :--- |
| **Qwen3-30B-A3B QLoRA · RTX 5090** | Unsloth spends **1.92×** e4b's GPU time per step, and **2.80×** its wall-clock time on an AMD EPYC 7713 host. Comparable held-out loss; Unsloth peaked lower (24.27 vs 26.16 GB). [Result](https://github.com/pjordanandrsn/experts4bit-qlora/blob/main/bench/h2h-2026-10-02/tc1/RESULTS-tc1-pos69.md) |
| **Why two numbers** | GPU time doesn't depend on the host. Unsloth runs about 14× e4b's CPU operations per step, so its wall-clock time grows on a slower host. Earlier wall-clock readings, before e4b's current defaults: 2.352× and 2.468×. |
| **Planner memory check · Qwen3-30B-A3B · RTX 5090** | **24.54 GiB** estimated process peak, **24.34 GiB** measured, after calibration from earlier runs. One in-sample case, not a guarantee. In the MoE audit, one RTX A2000 training plan is still under its measured peak (1.014×), where it borrows another model's reserve. [Plan vs run](https://github.com/pjordanandrsn/loggetta/blob/main/docs/RESULTS.md) · [MoE audit](https://github.com/pjordanandrsn/loggetta/blob/main/evidence/2026-10-09-moe-plan-vs-driver-after-gnf4/README.md) |

The comparison used torch 2.12.1+cu130 and transformers 5.5.0 for both frameworks, with matched adapters, initialization and
tokens. Loggetta does not predict throughput.

## Models

| Model | Tested in Loggetta |
| :--- | :--- |
| OLMoE-1B-7B-0924, Granite-3.1-3B-A800M | training and serving run |
| Granite-4.0-H-tiny | training run; serving refused (Mamba state) |
| Qwen3-30B-A3B, Qwen3.6-35B-A3B, ERNIE-4.5-21B-A3B | planned; serving validated |
| LFM2-8B-A1B | planned; serving refused (conv state) |
| Mixtral-8x7B-Instruct | planned |
| Gemma-4-26B-A4B-it, Nemotron-3.5-Lightning-30B-A3B | supported by the runtime; not yet in Loggetta's sweep |

Every row is supported by the included runtime's QLoRA path. Hybrid models (Qwen3.6, Granite-4.0-H, LFM2, Nemotron-H)
need `--packing concat` with chat or Alpaca data. The runtime's
[capability register](https://github.com/pjordanandrsn/experts4bit-qlora/blob/main/docs/capabilities.json) is the
authority.

## Scope

Single-GPU MoE planning and QLoRA training; serving placement is planned, not launched. Dense models are planned, and
train only behind `--allow-development-executor`. Not yet: multi-GPU, throughput prediction. GPU runs need Linux, an
NVIDIA CUDA GPU and a compatible PyTorch. Pre-1.0.

[GitHub](https://github.com/pjordanandrsn/loggetta) ·
[Results](https://github.com/pjordanandrsn/loggetta/blob/main/docs/RESULTS.md) ·
[Architecture](https://github.com/pjordanandrsn/loggetta/blob/main/docs/ARCHITECTURE.md) ·
[Research and releases](https://cerinamroth.com/ml/)
