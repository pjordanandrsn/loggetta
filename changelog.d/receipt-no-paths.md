### Receipts carry no local path

A receipt is meant to be shared, and it used to carry the machine's layout, home directory and username in several
places: the command line, the source files, a local model's directory, an adapter's directory, and notes. Every
execution receipt (train and serve), and `bench/serve_validate.py`'s, now passes through `loggetta.shareable.record`
before it is written:
- A path argument in `provenance.argv` is recorded by its name (`local: plan.json`). A file argument also gets its
  sha256 under `provenance.argv_files`.
- `provenance.sources.<module>.file` is the package-relative path, plus the version and the file's sha256.
- A local model directory, in the receipt and in its embedded plan, is recorded as `local: <name>`, the rule
  `first_run.json` already applies. A Hub id is kept as given.
- Any absolute path left anywhere, as a value or inside a sentence, is recorded by its name. URLs, units such as
  `GB/s`, and fractions are untouched.

The planner compares models by the same label on both sides, so a run on a local model directory still informs the
next plan of that model. Two different local directories with the same name now compare equal; a Hub id compares
exactly as before.
