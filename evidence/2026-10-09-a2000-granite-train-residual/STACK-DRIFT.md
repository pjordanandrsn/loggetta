# granite-3.1-3b-a800m: a 32 MiB peak drift between two software stacks (recorded, not attributed)

The same granite-3.1-3b-a800m training setup (`grouped_nf4`, resident, 2 × 512 tokens, r8 bf16 adapters, AdamW,
`E4B_ABSMAX_DQ=0`) was run on the same RTX A2000 on two stacks:

| | 2026-10-04 receipt | 2026-10-09 re-run (G2) | change |
|---|---|---|---|
| allocated peak, training | 3,163.9 MiB | 3,131.5 MiB | −32.4 MiB |
| allocated peak, after load | 2,274.7 MiB | 2,247.1 MiB | −27.6 MiB |
| experts4bit-qlora | 0.44.0 | 0.51.0 | |
| grouped-nf4-gemm | 0.37.0 | 0.44.0 | |
| torch | 2.8.0+cu128 | 2.11.0+cu128 | |
| transformers | 5.18.0 | 5.19.0 | |

Receipts: `../2026-10-04-rtx-a2000/granite-3.1-3b-a800m-instruct-device-grouped_nf4-t1024-20261004T180058Z.json` and
`runs/G2/receipts/granite-3.1-3b-a800m-instruct-device-grouped_nf4-t1024-20261009T160437Z-a178340f.json`.

- **It is mostly static.** 27.6 of the 32.4 MiB is already there after load, in what is loaded and resident, not in
  the training step's transient. The two receipts' engaged records do not differ.
- **It is in the safe direction.** Today's stack peaks lower.
- **The estimate covers both runs:** it is 40.0 MiB over the 2026-10-04 peak and 72.5 MiB over today's, with logits at
  12 B per logit (experts4bit-qlora #1467).
- **It is not attributed.** Four components moved at once, and an attribution would need the old stack rebuilt for an
  allocator replay. For a static drift in the safe direction that the estimate covers, that was judged not worth the
  GPU time.
- **What would reopen it:** a drift the other way (a newer stack peaking higher), or one that takes a peak past the
  estimate.
