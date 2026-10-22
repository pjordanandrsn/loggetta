### Plans: an upper bound on the load's host memory, warned and graded, never a refusal

The plan's host estimate prices the steady state. Loading also maps the checkpoint's weight files, and their pages count
in the process's RSS: under WSL2 and in containers they count against the memory cap even when the kernel calls them
reclaimable. On one WSL2 run the host estimate was 6.38 GiB and the measured peak RSS 13.1 GiB, against 14.5 GiB
available.
- `plan` reads the checkpoint's weight-file sizes without downloading them: from the Hub API's file metadata, a local
  directory, or the local Hub cache. When none answers, nothing is priced and the plan is unchanged.
- The load-time bound is the host estimate plus those bytes, labelled an inferred upper bound, not a measurement. It is
  recorded in the plan's `provenance.load_bound`. When it exceeds what this machine had available at plan time, a
  named warning gives both numbers and where the available figure came from. It never refuses a plan.
- `execute` reads what is available again right before the backend loads, and logs the same comparison.
- The receipt's `load_bound` grades the bound against the run's measured peak RSS (VmHWM, the load included): `held`
  is true, false, or null when no peak was measured. The summary prints it with each number's basis.
