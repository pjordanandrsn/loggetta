# Contributing

Thanks for looking. Loggetta plans and runs; the runtime
([`experts4bit-qlora`](https://github.com/pjordanandrsn/experts4bit-qlora)) loads,
trains and serves, and the kernels
([`grouped-nf4-gemm`](https://github.com/pjordanandrsn/grouped-nf4-gemm)) do the
4-bit expert math. A plan that is wrong belongs here; a crash inside training or a
slow kernel usually belongs in one of those two.

## Before filing a bug

Attach the plan and the run report. `loggetta plan … --json` and the report JSON
from `loggetta train` say which configuration was chosen, why alternatives were
rejected, and which optimizations actually ran. Most reports are answered by them.

## Running the checks

```bash
pip install -e ".[test]"
pytest tests/ -q              # GPU-only tests skip with a reason; do not filter by name
```

## Changelog entries

Add `changelog.d/<pr-or-slug>.md` holding your `### Title` and body; never edit
`CHANGELOG.md`'s `## Unreleased` by hand. The release writes the fragments into
`CHANGELOG.md`.

## Numbers in the docs

Measured numbers in the README and PYPI.md link to their evidence (under `evidence/`
or a linked benchmark). A change to a number needs the run that supports it.
