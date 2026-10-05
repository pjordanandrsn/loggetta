<div align="center">

# Loggetta

### Plan first. Execute through the backend.

**Loggetta turns a workload, a machine, and constraints into an inspectable execution plan.**

It selects a supported configuration, explains why it won, records why alternatives lost, and refuses impossible plans before loading weights.

<p>
  <img src="https://img.shields.io/badge/Python-%E2%89%A53.10-3776AB?logo=python&logoColor=white" alt="Python >=3.10">
  <img src="https://img.shields.io/badge/status-pre--1.0-F59E0B" alt="pre-1.0">
  <img src="https://img.shields.io/badge/scope-single--GPU%20MoE-6F42C1" alt="single-GPU MoE">
  <img src="https://img.shields.io/badge/artifacts-ExecutionPlan%20%2B%20ExecutionReceipt-2EA44F" alt="ExecutionPlan + ExecutionReceipt">
</p>

</div>

---

> [!NOTE]
> **Loggetta decides what should execute. The backend knows how to execute it.**

Loggetta owns the **Planner**, the **ExecutionPlan**, and the **ExecutionReceipt**. Its current focus is single-GPU
MoE training and serving through [experts4bit-qlora](https://github.com/pjordanandrsn/experts4bit-qlora), with
[grouped-nf4-gemm](https://github.com/pjordanandrsn/grouped-nf4-gemm) providing the packed low-bit kernel and
residency primitives beneath that runtime.

| Planner | ExecutionPlan | ExecutionReceipt |
| :--- | :--- | :--- |
| Turns hardware, workload, constraints, objectives, and evidence into a decision. | Captures the selected backend/setup, budgets, estimates, rejected alternatives, refusal reasons, warnings, and evidence quality. | Records what actually happened: memory, timing, correctness, engagement, provenance, and estimate-vs-measured deltas. |

## The loop

```mermaid
flowchart LR
    I["Workload + machine<br/>constraints + evidence"] --> P["Loggetta Planner"]
    P --> X["ExecutionPlan<br/>what should run + why"]
    X --> E["experts4bit-qlora<br/>executes the plan"]
    E --> G["grouped-nf4-gemm<br/>kernels + primitives"]
    E --> R["ExecutionReceipt<br/>what actually happened"]
    R -. "measured feedback" .-> P
```

The ownership split is deliberate:

| Layer | Owns | Does **not** own |
| :--- | :--- | :--- |
| **Loggetta** | Hardware inventory/provenance, budgets, objectives, constraints, candidate ordering, refusal policy, `ExecutionPlan`, `ExecutionReceipt`, thin execution orchestration | Model loaders, QLoRA machinery, serving engines, residency engines, kernels |
| **experts4bit-qlora** | Model topology/conventions, admission, footprint primitives tied to its runtime, loading, adapters, QLoRA preparation, training, serving, residency/offload mechanisms | Cross-backend planning policy |
| **grouped-nf4-gemm** | Packed low-bit kernels, routing/capability facts, and low-level residency primitives | Model/runtime policy or planner decisions |

```text
Loggetta -> experts4bit-qlora -> grouped-nf4-gemm
```

No cycles. No duplicated topology. No plugin framework until a second backend actually earns one.

## Quick start

```bash
# What machine am I actually on?
python -m loggetta inspect

# Produce the artifact Loggetta exists to produce
python -m loggetta plan Qwen/Qwen3-30B-A3B --seq 2048

# Constrain the planner deliberately
python -m loggetta plan allenai/OLMoE-1B-7B-0924 --experts device --vram 6

# Keep the plan as a file, then execute that plan through the backend; the receipt lands in receipts/
python -m loggetta plan allenai/OLMoE-1B-7B-0924 --seq 512 --micro-batch 2 --steps 12 --out plan.json
python -m loggetta execute plan.json --out receipts/

# Plan again from what was measured
python -m loggetta plan allenai/OLMoE-1B-7B-0924 --seq 512 --micro-batch 2 --observations receipts/
```

`train MODEL ...` is `plan` and `execute` in one step. Serving is planned with `plan MODEL --workload serve
--context T --concurrency N`. The plan's *Why* carries the server's exact environment; `execute` does not start
servers yet.

### Three levels of control

| Mode | Example | What it means |
| :--- | :--- | :--- |
| **Automatic** | `plan MODEL` | Let the planner choose among supported candidates. |
| **Directed** | `--vram`, `--ram`, `--experts`, `--objective` | State the budget or goal without hand-building a backend setup. |
| **Expert** | `--fix FIELD=VALUE` | Pin a backend setup field and make the remaining search work around it. |

`plan()` is pure policy: for the same inputs it returns the same serializable `ExecutionPlan`, without loading
weights or touching the device. The plan includes the winner, every relevant loser and why it lost, budget
sources, evidence labels, warnings, and explicit “not modeled” items.

`execute(plan)` is intentionally thin. It validates the plan, delegates execution to the selected backend, and
constructs an `ExecutionReceipt` from the backend result. It does **not** reimplement the runtime. Before anything
loads, it refuses a refused plan, a workload kind its backend only plans, and a plan made for a different GPU
(`plan --hardware`), whose receipt would name the wrong card.

<details>
<summary><strong>A real plan, abridged</strong> (OLMoE-1B-7B on an RTX A2000, from <code>evidence/2026-10-04-rtx-a2000/</code>)</summary>

```text
Budget    device  10.65 GiB [reported: free now]   host  23.49 GiB [reported: available now]   headroom   0.53 GiB [policy]

Selected  backend experts4bit: experts on device, grouped_nf4 kernel

Estimated memory (each line says how it is known)
  device frozen expert stacks                    3.38 GiB  [derived]  16 stacks in nf4, blocksize 64 (packed + absmax)
  device dense weights (bf16)                    0.89 GiB  [derived]
  device optimizer state (adamw)                 0.23 GiB  [derived]
  device activations                             0.54 GiB  [heuristic]  16 saved layer inputs (T x H bf16) + ...
  device allocator reserve (cached, unallocated blocks)   1.17 GiB  [measured]  receipt ...173932Z: ... = 0.222
  device CUDA context + library workspaces       0.13 GiB  [measured]  receipt ...173932Z
  device total                                   6.56 GiB   of  10.65 GiB budget

Why
  - experts resident on the device: 6.56 GiB estimated + 0.53 GiB headroom fits the 10.65 GiB budget
  - ordering, expert_kernel: grouped_nf4 before reference: e4b.train.h2h.unsloth.olmoe.5090.2026-09-19 (1.39 vs 14.88 s/step), ...

Performance  not predicted: no performance model is calibrated for this backend yet; ...

Alternatives considered
  [fits ] experts on device, grouped_nf4 kernel, NF4 attention       device   6.10 GiB  host   0.72 GiB  ...
  [fits ] experts on host, grouped_nf4 kernel                        device   2.69 GiB  host   4.10 GiB  ...
  ...
```

Its receipt: allocator 5.26 GiB estimated, 5.47 GiB measured; driver 6.56 GiB estimated, 6.82 GiB measured.

</details>

## Evidence, not vibes

Loggetta is deliberately **not** a performance oracle. It records what it knows, labels heuristics, refuses plans
it cannot support, and feeds measured receipts back into future planning.

Current checks from [`docs/RESULTS.md`](docs/RESULTS.md):

| Check | Measured result |
| :--- | :--- |
| **Planner parity, A2000 / OLMoE** | Planned and hand-composed paths had **bitwise-identical step-1 loss** (`1.858969`), effectively identical allocator peaks (`5.4710` vs `5.4706 GiB`), and the same `60,817,408` trainable parameters. |
| **Allocator estimates, A2000** | Across six measured runs spanning three model families, estimates landed **+0.01 to +0.21 GiB** from measured peaks. |
| **Qwen3-30B-A3B, RTX 5090** | Allocator estimate **22.09 GiB**, measured **21.91 GiB**. After receipt-calibrated runtime overheads: planned process peak **24.54 GiB**, measured **24.34 GiB**. |
| **Serving tiers, A2000 / OLMoE** | The planned VRAM / DRAM / NVMe split matched the server's own (**272 / 421 / 331** experts); allocator **2.052** GiB planned, **2.051** GiB measured. |
| **Host offload** | Transfer time is treated as a **lower bound**, not a fabricated step-time prediction. |

> [!IMPORTANT]
> Evidence has a type. Measured facts stay measured; derived values stay derived; inferred values and heuristics
> stay labelled. Unknowns do not quietly become numbers.

An `ExecutionReceipt` carries provenance, allocator / reserved / driver peaks, correctness checks, engagement
evidence, timing, and estimate-vs-measured deltas. That receipt is the measured counterpart to the plan that
produced it.

## Current scope

<table>
<tr>
<th>Supported today</th>
<th>Not claimed yet</th>
</tr>
<tr>
<td valign="top">

- ✅ Single-GPU MoE planning
- ✅ First-class `ExecutionPlan` and `ExecutionReceipt` artifacts
- ✅ QLoRA planning with execution delegated to `experts4bit-qlora`
- ✅ Serving placement planning across device, host, and storage tiers
- ✅ Measured-memory feedback through receipts
- ✅ Deterministic, inspectable plans and refusals
- ✅ Explicit constraints instead of hidden “magic” defaults

</td>
<td valign="top">

- ⏳ Dense-model planning as a first-class Loggetta backend
- ⏳ Multi-GPU placement or execution planning
- ⏳ Calibrated throughput prediction
- ⏳ Profile-driven serving hot sets
- ⏳ Automatic execution of every serving plan

</td>
</tr>
</table>

Those are roadmap items, not assumptions hidden inside current results.

## Install

Until the functional PyPI release lands, install from source:

```bash
git clone https://github.com/pjordanandrsn/loggetta
cd loggetta
python -m pip install -e ".[experts4bit]"
```

The required lower-layer APIs are present in:

- `experts4bit-qlora >= 0.48.0`
- `grouped-nf4-gemm >= 0.41.0`

No development branches or `PYTHONPATH` overrides are required for training plans and their execution. Serve plans
work with those releases too. The serve-estimate refinements in RESULTS 6b–6e are in experts4bit-qlora's next
release: exact bytes per expert, the cold tier's minimum `hot_rows`, the cold-row stack, prefill staging and the int4
levers. Until it ships, reproducing those numbers needs its `main`.

> [!TIP]
> PyPI `0.0.1` is the metadata-only preview used to establish the project name. Functional releases will
> supersede it.

## Why a separate planner?

Kernel selection belongs with kernels. Model topology and QLoRA mechanisms belong with the runtime. The decision
**which combination should run on this machine, under this budget, for this objective** is a separate concern,
and the `ExecutionPlan` is the durable artifact of that decision.

Keeping that boundary clean means:

- backend packages remain independently useful;
- planning stays deterministic and inspectable;
- the plan can explain both selection and refusal before weight loading;
- the runtime can evolve without absorbing cross-machine policy;
- receipts can improve future policy without contaminating kernel or model code;
- a second backend can earn a general abstraction instead of forcing one prematurely.

## Documentation

| Read this | For |
| :--- | :--- |
| [`SESSION-REPORT.md`](docs/SESSION-REPORT.md) | Short answers and current state |
| [`ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Ownership boundaries and the Planner → Plan → Backend → Receipt model |
| [`RESULTS.md`](docs/RESULTS.md) | Measurements, estimator error, receipts, and what changed because of them |
| [`SERVING-PRESSURE-TEST.md`](docs/SERVING-PRESSURE-TEST.md) | Serving placement and pressure-test evidence |

<details>
<summary><strong>Tests and validation commands</strong></summary>

```bash
pytest tests/
python bench/direct_baseline.py --receipt R.json
python bench/validate_register.py
python bench/family_sweep.py --hardware hw.json
python bench/summarize_receipts.py runs/receipts
```

The fast test suite is CPU-only. Planner tests run when `experts4bit-qlora` is importable. The execution tests
check the handoff to the backend with a stand-in executor, so they need nothing installed. GPU benchmarks and
validation runs stay separate so ordinary correctness tests do not silently become hardware-dependent.

</details>

---

<div align="center">

**Plan first. Execute through the backend. Measure. Feed the receipt back.**

</div>
