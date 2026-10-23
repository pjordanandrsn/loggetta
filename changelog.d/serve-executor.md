### `execute` runs a serve plan

- A feasible serve plan now executes through experts4bit-qlora's paged engine, the same build-decode-measure path
  `bench/serve_validate.py` has used for the committed serve receipts: `loggetta execute PLAN --arena ARENA --calib
  CALIB`. The receipt carries the same measured fields: device and driver peaks, load and
  generate seconds, tokens per second, and at one sequence the traced single-stream decode step.
- It needs no development opt-in. A GPU smoke on an RTX 4090 ran the bench path, the plan and execute, and the
  inherited-switch refusal at experts4bit-qlora 0.57.0 (`evidence/2026-10-23-serve-executor-rtx4090`). The dense
  executor keeps its opt-in.
- A receipt's provenance names the loggetta that wrote it: the installed distribution's version, and the module's own
  `__version__` (as `module_version`), which is still there when it runs from a source tree with no installed
  distribution. Before, `sources.loggetta.version` was always null.
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
- The serve executor steps the engine as experts4bit-qlora's server loop does, opening and closing a trace row around
  each step, so a one-sequence run's decode step is traced. `scheduler.run_until_idle()` steps without opening a row,
  so the trace stayed empty. A missing or empty trace is now recorded as a note, never a crash. A GPU smoke on an RTX
  4090 found this; the same gap was in `bench/serve_validate.py` before the executor moved into the package.
