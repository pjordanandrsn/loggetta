### Receipts carry no local path

A receipt is meant to be shared, and it used to carry the machine's layout, home directory and username in several
places: the command line, the source files, a local model's directory, an adapter's directory, and notes. Every
execution receipt (train and serve), and `bench/serve_validate.py`'s, now passes through `loggetta.shareable.record`
before it is written:
- A path argument in `provenance.argv` is recorded by its name (`local: plan.json`). A file argument also gets its
  size and sha256 under `provenance.argv_files`. Above 64 MiB it gets its size and the sha256 of its first MiB instead,
  so an NF4 arena of tens of GB is never read whole.
- `provenance.sources.<module>.file` is the package-relative path, written with forward slashes on every OS, plus the
  version and the file's sha256.
- A local model directory, in the receipt and in its embedded plan, is recorded as `local: <name>@<12 hex>`, where the
  hex is the sha256 of its `config.json` plus its safetensors index, or its weight files' names and sizes. A Hub id is
  kept as given.
- Any path left anywhere with a directory in it, as a value, a key, or inside a sentence, is recorded by its name. That
  covers home-relative and Windows drive paths too. URLs, units such as `GB/s`, fractions and one-segment routes such
  as `/health` are untouched.

The planner compares models by the same label on both sides, so a run on a local model directory still informs the
next plan of that same directory. The label hashes the architecture and sharding, not the weights:
- two directories with the same name but a different config or sharding (two `checkpoint-500`s of different models)
  never lend each other measured numbers;
- two fine-tunes of one base sharded alike do compare equal. That suits shape-driven numbers such as memory and step
  time, but not numbers that follow the weights, such as MoE routing;
- a local model with no readable `config.json` gets no fingerprint and matches no stored observation, so it is measured
  as unseen.
A Hub id compares exactly as before.
