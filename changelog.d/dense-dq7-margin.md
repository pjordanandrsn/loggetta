### Dense plans: warn and require streamed admission headroom after DQ7

The VOID DQ7 reading understates device use on every completed streamed arm, by up to 2,395,904,403B. It also understates resident Llama 4096 driver use by 820,943,251B. Warn on dense plans and enforce streamed admission headroom of at least max(2.4 GB, 20% of the full device estimate) even when the caller asks for less. Keep allocator/reserve coefficients and execution opt-in unchanged; receipts continue comparing the original estimate, excluding headroom.
