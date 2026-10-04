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

Read [`docs/SESSION-REPORT.md`](docs/SESSION-REPORT.md) first (short answers), then [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md),
[`docs/RESULTS.md`](docs/RESULTS.md) and [`docs/SERVING-PRESSURE-TEST.md`](docs/SERVING-PRESSURE-TEST.md).

## Status

- **Private.** Not on any package index and not announced.
- **The name is a label.** Nothing serialized carries it; schemas are `execution-plan/1` and
  `execution-receipt/1`. `tests/test_renameable.py` enforces this.
- **Needs unreleased lower-layer changes**:
  - experts4bit-qlora [#1055](https://github.com/pjordanandrsn/experts4bit-qlora/pull/1055) (`describe_moe`, `recipe`);
  - grouped-nf4-gemm [#464](https://github.com/pjordanandrsn/grouped-nf4-gemm/pull/464) (`nf4_route.route_for`).
  Until they merge, put their branches on `PYTHONPATH`, as the receipts record.

## Tests

```
pytest tests/                         # fast; CPU; tiny configs through the real model-family layer
python bench/direct_baseline.py --receipt R.json   # the hand-composed comparison arm for a planned run (GPU)
python bench/validate_register.py                  # price recipes the register measured; compare (no GPU)
python bench/family_sweep.py --hardware hw.json    # plan ten MoE families; per-family code stays zero
python bench/summarize_receipts.py runs/receipts   # the tables in docs/RESULTS.md
```
