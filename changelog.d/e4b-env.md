### Receipts record the experts4bit-qlora environment

- An execution receipt's `provenance.e4b_env` records the experts4bit-qlora environment the run started under:
  - the sorted names of every set `E4B_*` variable;
  - the sha256 of the sorted `name=value` lines, so two receipts compare without their values;
  - the values of the switches that change numerics or speed (`measure.E4B_VALUE_KNOBS`, for example
    `E4B_ABSMAX_DQ` and the `E4B_PAGED_*` and `E4B_SERVE_*` levers).
- Any other variable is recorded by name only. A rental, host, port, token, secret, key, password or auth variable adds
  only its name to the hash too, so a receipt never carries a host name or a secret, nor a hash that could guess one.
- Two committed receipts ran different `E4B_ABSMAX_DQ` settings under an identical setup and command line, and differed
  by 39% in seconds per step. Nothing recorded the difference until now.
- It is recorded only: every committed receipt plans identically with the key added.
