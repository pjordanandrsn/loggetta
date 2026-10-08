### Single-stream serve plans: the B=1 levers a default server runs, and a decode figure only from a receipt of the same setup

- **What changed.**
  - **The levers.** A serve plan for one sequence names the B=1 levers that a default `serve_paged` runs for the
    model's family, as the installed experts4bit-qlora resolves them: its family-scoped fused stack (fused q/k/v and
    three glue folds), with the read it rests on or the read the family lacks. An experts4bit-qlora without the
    family-scoped default reads as "off unless set". Nothing here keeps its own table of families.
  - **The figure.** The plan's performance section quotes a decode figure only from a serve receipt of exactly the
    same setup: same model, GPU name and driver, setup, and experts4bit-qlora and grouped-nf4-gemm versions, at one
    sequence. The figure is labelled measured, with the receipt's ID and the prompt and new-token counts it was measured at (a decode step grows with the KV position). Anything less shows nothing: loggetta still
    predicts no throughput, and no figure is interpolated or carried between hosts.
  - **The receipt.** `bench/serve_validate.py --concurrency 1` traces the engine's steps (`E4B_PAGED_STEP_TRACE`) and
    records `measured.decode_step_ms_b1`: the median whole-step time of the steps that decoded one row and ran no
    prefill. It also records the decode's device time and the step count, from `loggetta.measure.decode_step_b1`.
- **Why.** On experts4bit-qlora, one sequence's decode speed now depends on the model family as well as the setup:
  the B=1 fused stack is on by default only where it was read (lane P115, experts4bit-qlora #1361 and #1379). A plan
  should say which levers apply, and quote a speed only where one was measured on the same setup.
- **Tests:** `tests/test_single_stream.py`:
  - the trace reader;
  - the levers line with and without the family-scoped default;
  - the exact-match rule, each mismatch showing nothing;
  - a planner round trip.
