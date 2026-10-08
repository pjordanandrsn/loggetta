# Estimator-only dense plans with enforced streamed headroom (2026-10-08)

These are config-only planner choices at Loggetta `efe1c93`, seq4096, micro-batch1, stated 16–48 GB card classes and ample host memory. No weights, training, CUDA execution or capacity measurements were taken. The inferred 20% reserve and 0.5 GiB context remain. Streamed admission additionally enforces max(2,400,000,000 bytes, ceil(20% of the full device estimate), normal policy/caller headroom). This margin is outside the itemized device total; a feasible plan is not a fit guarantee.

[dense-plan-sweep.md](dense-plan-sweep.md) is the script's unedited file output; [stdout.txt](stdout.txt) is its unedited table output. The file's original generic header does not describe the streamed admission floor, which is stated above. [sources.json](sources.json) records fourteen config-only Hub commit pins and hashes; their exact JSON bytes are under `configs/`. [provenance.json](provenance.json) records runtime versions and source hashes. All files have SHA-256 checksums.

The 2026-10-07 table had no pinned configs; its inputs are unavailable. [comparison.json](comparison.json) compares by model name only, without asserting byte-identical inputs or attributing every difference to a code change. Thirty-seven of fifty-six displayed cells differ. Four formerly feasible model-name/card cells now refuse: Qwen2.5-72B on 32 GB, Qwen3-32B on 16 GB, Gemma-2-9B on 24 GB and Gemma-3-12B on 24 GB. Neither sweep licenses capacity. The old evidence directory is unchanged.

Reproduce from the recorded Loggetta commit with the recorded package versions:

```sh
python bench/dense_plan_sweep.py --configs evidence/2026-10-08-dense-plan-sweep/configs --seq 4096 --out /tmp/dense-plan-sweep.md
```

Compare the output bytes with the committed file and verify `SHA256SUMS` from this directory. This evidence changes no estimate formula, coefficient, reserve import, executor opt-in or registered gate.
