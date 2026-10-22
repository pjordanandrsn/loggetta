### Receipts: which loss branch the training forwards took, counted

A receipt now records, in `engaged.chunked_lm_loss`, which loss branch the run's training forwards took. experts4bit-qlora
chooses the chunked LM loss or the stock one per forward under a size gate, and the receipt counts each, as the change
in its counters over this run's training loop: chunked, under the gate (stock), stock otherwise. It also records the
gate's threshold, the chunk size and the head width the gate read. `branch` reads `chunked`, `stock` or `mixed`, so a
row near the gate states its branch from its own counters instead of having it inferred from its peak. Recording never
fails a run that trained: an error is written into the record instead.
