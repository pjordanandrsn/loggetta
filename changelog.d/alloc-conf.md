### Receipts record the allocator setting

- An execution receipt's `provenance` records `pytorch_cuda_alloc_conf`, the `PYTORCH_CUDA_ALLOC_CONF` the run started
  under (null when unset). It moves the allocator's reserved-but-unallocated bytes, which the planner learns its
  reserve from, so a reader can now see which setting a measured reserve came from. It is recorded only: every
  committed receipt plans identically with the key added.
