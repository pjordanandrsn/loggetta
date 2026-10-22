### Receipts: a run whose device check had no driver sample reads UNMEASURED, not OK

A training run's device peak is checked against the plan through `nvidia-smi`'s per-process memory. Under WSL2, and in
some containers, `nvidia-smi` lists no per-process memory, so the check took no sample (`driver_samples` 0). The run
still read `OK`, a status claiming a check that never happened. Such a run now reads `UNMEASURED`:
- The receipt's `measured.driver_unmeasured` says why: no `nvidia-smi` on PATH, every read failed, or reads that never
  listed this process.
- The summary prints "device check NOT RUN".
- `execute` and `train` exit with code 4, which `--help` lists: trained, but the device check took no driver sample.
- The run trained normally, so its adapter is saved. A fallback to a `/1` manifest is still reported.
- Its allocator slack still counts as planning evidence, but it never teaches the CUDA-context term, which needs a
  driver peak.

An integrity failure (`ALARM`) still takes precedence.
