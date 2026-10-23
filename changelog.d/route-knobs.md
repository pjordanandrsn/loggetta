### Receipts record the route and kernel switches by value

- `provenance.e4b_env` now covers grouped-nf4-gemm's `GNF4_*` switches beside experts4bit-qlora's `E4B_*`: their names,
  the shared hash, and their values.
- The value list (`measure.E4B_VALUE_KNOBS`) gains the route and kernel switches. These include `E4B_K25_PLAN` and
  `E4B_K25_LUT`, the `E4B_NF4_*` and `E4B_INT4_*` routes, the dense and head GEMVs, the fused glue, and every `GNF4_*`
  switch except the build-cache directory `GNF4_NATIVE_CACHE`. Before, a receipt could not tell `E4B_K25_PLAN=c1` from
  `today`, or a `GNF4_*` switch on from off.
- A listed switch whose value is nonetheless a path is recorded, and hashed, by its name only.
- Every other switch the backends read is in `measure.NAME_ONLY` with its reason: a path, an id, a secret, or
  HTTP-server plumbing. A test scans the installed experts4bit-qlora and grouped-nf4-gemm sources and fails on any name
  in neither table. A backend release that adds a knob fails loggetta's CI until someone decides whether its value is
  recorded.
- A name with a credential word in it (TOKEN, KEY, SECRET, PASSWORD, AUTH, CREDENTIAL, as a whole `_` word; `TOKENS`, a
  count, is not one) never has its value recorded or hashed, even if it were listed. A test keeps such names off the
  list too.
- This is still recorded only: the planner reads none of it and matches no receipt by this hash. Receipts written before
  this change simply lack the new values.
