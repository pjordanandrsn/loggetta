### Plans record the backend switches their estimate read; execute warns when the run differs

- **What changed.** A feasible plan's provenance records `estimate_env`: the environment switches the selected
  backend's estimate read, with their values in the planning process. For experts4bit-qlora that is what its own
  accessor (`recipe.estimate_env`) reports. `execute` compares them with the running process's before anything loads.
  On a difference it logs a warning naming each switch, planned and running, and it records the comparison in the
  receipt's `provenance.estimate_env`.
- **Why.** experts4bit-qlora's training estimate reads `E4B_CHUNKED_LM_LOSS` (experts4bit-qlora#1491), which decides
  whether the loss is priced chunked or whole. `loggetta train` plans and runs in one process. A plan written by
  `loggetta plan --out` and run by `loggetta execute` in another process could price a loss the run does not take, and
  nothing would say so.
- **Unchanged where nothing is recorded.** A backend release without the accessor records nothing and the check is
  skipped. Older plans run as before. loggetta still reads no environment variable itself.
- **Tests.** `tests/test_estimate_env.py`.
