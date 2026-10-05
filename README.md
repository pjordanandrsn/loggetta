# Loggetta

**Decide how a workload should run on this machine, explain why, and refuse impossible plans before loading weights.**

Loggetta is a planning and policy layer above
[experts4bit-qlora](https://github.com/pjordanandrsn/experts4bit-qlora) and
[grouped-nf4-gemm](https://github.com/pjordanandrsn/grouped-nf4-gemm). Its current focus is single-GPU MoE
training and serving: inventory the machine, price candidate configurations, choose among them under explicit
budgets and objectives, and say *why* a plan won or why nothing fits.

The lower layers own mechanism. Loggetta owns policy.

```text
Loggetta                 budgets, headroom, objectives, candidate ordering, refusal, receipts
    |
experts4bit-qlora        model structure, QLoRA preparation, residency/offload mechanisms
    |
grouped-nf4-gemm         packed low-bit kernel routes
```

## Quick start

```bash
python -m loggetta inspect
python -m loggetta plan Qwen/Qwen3-30B-A3B --seq 2048
python -m loggetta plan allenai/OLMoE-1B-7B-0924 --experts device --vram 6
python -m loggetta train allenai/OLMoE-1B-7B-0924 --seq 512 --micro-batch 2 --steps 12 --out receipts/
```

Three levels of control are intentional:

- **Automatic:** `plan MODEL`
- **Directed:** `--vram`, `--ram`, `--experts device|host`, `--objective`
- **Expert:** `--fix FIELD=VALUE` pins a backend setup field

`plan()` returns a deterministic, serializable `ExecutionPlan`. It includes the alternatives that lost and
why they lost. `execute(plan)` runs a supported plan and writes an `ExecutionReceipt` that puts the estimate
and the measurement side by side.

## Install

Loggetta currently installs from source:

```bash
git clone https://github.com/pjordanandrsn/loggetta
cd loggetta
python -m pip install -e ".[experts4bit]"
```

The current dependency floor is the released lower-layer API surface in:

- `experts4bit-qlora >= 0.48.0`
- `grouped-nf4-gemm >= 0.41.0`

No development branches or `PYTHONPATH` overrides are required for those interfaces.

The initial PyPI `0.0.1` release is a metadata-only preview used to establish the project name. Functional PyPI
releases will supersede it.

## What is measured

Loggetta is deliberately not a performance oracle. It records what it knows, labels heuristics, refuses plans it
cannot support, and learns memory overheads from receipts.

A few current checks from [`docs/RESULTS.md`](docs/RESULTS.md):

- On an RTX A2000 OLMoE training run, the planned path and a hand-composed
  `experts4bit-qlora` path had **bitwise-identical step-1 loss** (1.858969), effectively identical allocator
  peaks (5.4710 vs 5.4706 GiB), and the same 60,817,408 trainable parameters.
- Across six measured A2000 runs spanning three model families, allocator estimates were **+0.01 to +0.21 GiB**
  from measured peaks.
- On an RTX 5090 Qwen3-30B-A3B resident run, the allocator estimate was **22.09 GiB vs 21.91 GiB measured**.
  After the receipt calibrated runtime overheads, the planned process peak was **24.54 GiB vs 24.34 GiB
  measured**.
- Host-offload transfer time is treated as a **lower bound**, not a predicted step time. Measured timings remain
  evidence, not something Loggetta pretends it knew beforehand.

Receipts capture provenance, allocator/reserved/driver peaks, correctness checks, estimate-vs-measured deltas and
engagement evidence so the planner's decisions can be inspected after the fact.

## Current scope

Loggetta is pre-1.0 and under active development.

Supported today:

- single-GPU MoE planning;
- QLoRA training planning and supported execution through `experts4bit-qlora`;
- serving placement planning across device, host and storage tiers;
- measured-memory feedback through receipts;
- deterministic, inspectable plans and refusals;
- explicit user constraints instead of hidden "magic" defaults.

Not claimed yet:

- dense-model planning as a first-class Loggetta backend;
- multi-GPU placement or execution planning;
- a calibrated throughput predictor;
- profile-driven serving hot sets;
- automatic execution of every serving plan.

Those are roadmap items, not assumptions hidden inside current results.

## Why a separate planner?

The backend packages should stay independently useful. Kernel selection belongs with kernels; model topology and
QLoRA mechanisms belong with the training backend. The decision *which combination should run on this machine
under this budget and objective* is a different concern.

That separation keeps the dependency direction simple:

```text
planner -> experts4bit-qlora -> grouped-nf4-gemm
```

No plugin framework is required for the first backend. A second backend is the point at which the abstraction
earns the generalization.

## Documentation

Start with [`docs/SESSION-REPORT.md`](docs/SESSION-REPORT.md) for the short answers, then:

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- [`docs/RESULTS.md`](docs/RESULTS.md)
- [`docs/SERVING-PRESSURE-TEST.md`](docs/SERVING-PRESSURE-TEST.md)

## Tests

```bash
pytest tests/
python bench/direct_baseline.py --receipt R.json
python bench/validate_register.py
python bench/family_sweep.py --hardware hw.json
python bench/summarize_receipts.py runs/receipts
```

The fast test suite is CPU-only. GPU benchmarks and validation runs are kept separate so ordinary correctness
tests do not silently become hardware-dependent.
