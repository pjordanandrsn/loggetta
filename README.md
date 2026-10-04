# loggetta (working name; private incubation)

A planning layer above [experts4bit-qlora](https://github.com/pjordanandrsn/experts4bit-qlora) and
[grouped-nf4-gemm](https://github.com/pjordanandrsn/grouped-nf4-gemm). It decides how a MoE workload should run
on *this* machine, shows why, and refuses with reasons when nothing fits. It does this before loading a single
weight.

```
python -m loggetta inspect                         # hardware inventory, every value labelled with its source
python -m loggetta plan Qwen/Qwen3-30B-A3B --seq 2048
python -m loggetta plan allenai/OLMoE-1B-7B-0924 --experts device --vram 6
python -m loggetta train allenai/OLMoE-1B-7B-0924 --seq 512 --micro-batch 2 --steps 12 --out receipts/
```

- **Policy here, mechanism below.** Budgets, headroom, runtime overhead, the order an objective tries setups in,
  and refusal live in this package. Model structure, memory per item and kernel routes come from the packages that
  own them.
- **Plan first, execute second.** `plan()` returns an `ExecutionPlan`: deterministic, serializable, renderable. It
  includes the alternatives that lost and why they lost. `execute(plan)` runs it and writes a receipt that puts
  the estimate and the measurement side by side.
- **Three levels of control.**
  - Automatic: `plan MODEL`.
  - Directed: `--vram`, `--ram`, `--experts device|host`, `--objective`.
  - Expert: `--fix FIELD=VALUE` pins any backend setup field.

Read [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for what lives where and what is measured versus speculative.

## Status

- **Private.** Not on any package index and not announced.
- **The name is a label.** Nothing serialized carries it; schemas are `execution-plan/1` and
  `execution-receipt/1`. `tests/test_renameable.py` enforces this.
- **Needs unreleased lower-layer changes**:
  - experts4bit-qlora branch `feat/config-only-footprint` (`describe_moe`, `recipe`);
  - grouped-nf4-gemm branch `feat/route-decision-without-device` (`nf4_route.route_for`).

## Tests

```
pytest tests/                         # fast; CPU; tiny configs through the real model-family layer
pytest tests/ -m gpu                  # (none yet) GPU integration lives in the lower packages' suites
python bench/direct_baseline.py ...   # the hand-composed comparison arm for a planned run
```
