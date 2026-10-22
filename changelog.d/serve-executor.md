### `execute` runs a serve plan (development executor)

- A feasible serve plan now executes through experts4bit-qlora's paged engine, the same build-decode-measure path
  `bench/serve_validate.py` has used for the committed serve receipts: `loggetta execute PLAN --arena ARENA --calib CALIB
  --allow-development-executor`. The receipt carries the same measured fields: device and driver peaks, load and
  generate seconds, tokens per second, and at one sequence the traced single-stream decode step.
- It is a development executor until one GPU smoke proves it, so it runs only with `--allow-development-executor`.
- It refuses by name, before anything loads, when the arena or the calibration file is missing, or when the request
  shape does not fit the plan's context. It never bakes an arena; that stays an explicit step.
- It also refuses by name when this environment sets any backend switch (`E4B_*`, `GNF4_*`) that the plan does not.
  This is denied by default, because the backends route kernels on switches no list could keep up with. The plan's
  settings are merged into the process environment, so such a switch would change the run while the receipt describes
  the plan. Only plumbing the engine build never reads may be inherited: the HTTP server's `E4B_TOKEN`, `E4B_HOST` and
  `E4B_PORT`, and the rental launcher's `E4B_RENT_*`. The refusal names the switch and never prints its value.
- Path-valued entries in the engine's own report (an adapter's path, a lever's artifact directory) are recorded by name
  only.
- The receipt names the arena by base name and the sha256 of its first MiB, and the calibration file by base name and
  sha256. Neither appears by path, and the recorded server environment names a local model directory by its name only
  (`local: <name>`), as `first_run.json` does.
- A serve receipt's run id names the context and concurrency (`...-serve-all-vram-c4096x1-...`).
- `bench/serve_validate.py` keeps its command line and receipt format, and now calls the same measuring code.
