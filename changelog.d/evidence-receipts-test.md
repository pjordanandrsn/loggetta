### Tests: every run an evidence log names has a committed receipt, or a note says why not

`tests/test_evidence_receipts.py` reads every committed log under `evidence/` for the run IDs it names, as loggetta's
`receipt: <path>.json` line or `bench/train_residual.py`'s `run <id> status` line. It requires a committed
`<run_id>.json` under `evidence/`, or a `no receipt: <run_id>` line in the directory's Markdown. `receipts/` is
gitignored, so a receipt has to be force-added. The step was missed twice (#49, #48), and both times a reader found it.
On main before #55 the test fails with exactly those runs. The granite directory now notes its lost run G. No code
changes.
