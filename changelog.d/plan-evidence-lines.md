### Plans: what the total rests on, said beside it

The plan's Why section now says what its memory total rests on:
- **Overhead sources:** which overhead terms (the CUDA context, the allocator reserve, the host baseline) were measured
  from receipts on this machine and which are defaults. Each line above it already named its receipt or default.
- **Streamed margin:** a streamed plan prints its admission margin beside the total, `max(2.4 GB, 20% of the plan)`.
  On the planner's own rows, streamed peaks have run above the plan total, and this margin is what covers them, so the
  total alone is not what the process will use (#90).
- **Cited receipts:** a receipt the plan cites names the loss branch its training forwards took, when the receipt
  counted it (a chunked and a stock loss peak differently), and a `SAVED_V1` status.

With no receipts on file, a plan cites no run id at all.
