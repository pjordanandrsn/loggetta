# Changelog

## 0.7.0 — 2026-10-23

**0.7.0.** A first install now ends trained or with a named refusal. `inspect --torch-index` names the torch build
for the NVIDIA driver before torch is installed, and `scripts/first_run.sh` runs the whole first install. A machine
without a working C compiler, or one loading CUDA's forward-compatibility library, is refused by name before any
download. `execute` runs serve plans; one GPU smoke on an RTX 4090 proved the path. MoE adapters record the quantized
base they were trained on (`backend-adapter/2`, with an experts4bit-qlora that defines it). Receipts carry no local
path, and they record the backend switches and allocator setting they ran under. Two things can change what a script
sees. A training run whose device check took no driver sample now reads `UNMEASURED` and exits 4, where it read `OK`
and exited 0. And receipts name local files and model directories by name only. Floors unchanged.

**External validation.** Thanks to [Thijs van Weezel](https://github.com/thijs-vanweezel) for independently
running Loggetta on a 6 GB RTX 1000 Ada under WSL2. The run showed that a device-memory check could read
`OK` without a single driver sample, and measured a 13.1 GiB peak host RSS against a 6.38 GiB host estimate
that priced only the steady state. Those findings informed this release's `UNMEASURED` status and load-time
host-memory bound. The tester also reported WSL2 installation friction for follow-up documentation.

### Install: the torch build for your driver, named before torch is installed, and a first-run script

`inspect` now names the PyTorch CUDA build that matches the NVIDIA driver, and `inspect --torch-index` prints only that
build's index tag (`cu130` for driver 580 and later, `cu126` for 525 and later; a Blackwell GPU, compute capability
10.0 and up, gets only builds for CUDA 12.8 or later, so `cu129` below driver 580, and needs driver 570), reading
`nvidia-smi` alone, so it works before torch is installed. A driver too old for every supported torch build, or a machine with no NVIDIA GPU, is a
named refusal (exit 3). `scripts/first_run.sh` runs a first install end to end in a new directory: a fresh virtual
environment, the matching torch FIRST, then the package, `inspect`, `plan`, and 10 training steps on OLMoE-1B-7B-0924-Instruct
at a pinned revision. It writes each step's exit status and seconds to `first_run.json` and ends trained (0, or 7 when the device check did
not run), or with a named refusal: no torch build fits (3), the plan was refused (4), an install step failed (5), or the run did not train (6).
A local wheel or directory given as `PACKAGE_SPEC` is recorded in `first_run.json` by its name only, never its path, so
a shared summary carries no home directory.
- A refused plan's summary in `first_run.json` is now the refusal's own first reason, verbatim, read from the refused
  `plan.json`. It was a grep of the printed banner, which kept only "NOT FEASIBLE under the requested constraints."
- A local model directory given as `MODEL` is recorded in `first_run.json` by its name only, never its path, and a
  refusal reason never carries the home directory (it reads `~`).
- A URL or VCS `PACKAGE_SPEC` (`git+https://...`, `name @ https://...`) is recorded as a URL, not as `local`, with any
  user, password, query and fragment dropped, so a token in the URL never reaches `first_run.json`. The URL's path is
  kept as given, since it names what was installed: a user or organization name in a remote path stays. A `file://` URL
  is a local path, recorded by its name only. A URL that does not parse is recorded by its scheme alone.

### Plans: a machine without a working C compiler is refused by name, before any download

The training and serving kernels are Triton kernels. Triton compiles a small helper against `Python.h` the first time
one runs. On a machine with no C compiler (the CUDA runtime images ship none), a first run installed torch, planned,
downloaded the checkpoint, and then died in training with Triton's "Failed to find C compiler" traceback.
- When an NVIDIA GPU is present, the hardware probe now finds the compiler Triton would use (`CC`, else gcc, else
  clang), and tests it the way Triton does: it builds a shared object against this Python's headers. A name on PATH is
  not enough, because a box can have gcc but no Python headers.
- Nothing recorded carries a path: the compiler is recorded by name, and its diagnostics (shown in the refusal) keep
  only the basenames of any absolute paths, such as the temporary file or a home directory in an include chain.
- When no compiler can build, `plan` refuses with the reason and the fix (install a compiler and the Python headers, or
  set `CC`). `inspect` reports it, and `scripts/first_run.sh` ends with its named exit 4.
- A profile saved before this field reads "not probed" and is never refused for it.

### The refusal names CUDA Error 804 correctly

- When torch reports CUDA Error 804 ("forward compatibility was attempted on non supported HW"), the refusal now says
  what happened. The process loaded NVIDIA's CUDA forward-compatibility library (cuda-compat), which this GPU does
  not support. NVIDIA supports it on datacenter GPUs, select server RTX cards and Jetson boards.
- It now gives the remedies that work:
  - an image or environment whose CUDA the driver supports natively;
  - the compat directory taken off `LD_LIBRARY_PATH`;
  - a newer host driver.
- Before, a driver that met the build's CUDA minimum sent the user to check GPU visibility (`CUDA_VISIBLE_DEVICES`,
  `docker run --gpus`), which has nothing to do with this error. Seen on a GeForce RTX 3090 with driver 535 and a
  CUDA 12.8 image.

### `execute` runs a serve plan

- A feasible serve plan now executes through experts4bit-qlora's paged engine, the same build-decode-measure path
  `bench/serve_validate.py` has used for the committed serve receipts: `loggetta execute PLAN --arena ARENA --calib
  CALIB`. The receipt carries the same measured fields: device and driver peaks, load and
  generate seconds, tokens per second, and at one sequence the traced single-stream decode step.
- It needs no development opt-in. A GPU smoke on an RTX 4090 ran the bench path, the plan and execute, and the
  inherited-switch refusal at experts4bit-qlora 0.57.0 (`evidence/2026-10-23-serve-executor-rtx4090`). The dense
  executor keeps its opt-in.
- A receipt's provenance names the loggetta that wrote it: the installed distribution's version, and the module's own
  `__version__` (as `module_version`), which is still there when it runs from a source tree with no installed
  distribution. Before, `sources.loggetta.version` was always null.
- It refuses by name, before anything loads, when the arena or the calibration file is missing, or when the request
  shape does not fit the plan's context. It never bakes an arena; that stays an explicit step.
- It also refuses by name when this environment sets any backend switch (`E4B_*`, `GNF4_*`) that the plan does not.
  This is denied by default, because the backends route kernels on switches no list could keep up with. The plan's
  settings are merged into the process environment, so such a switch would change the run while the receipt describes
  the plan. Only plumbing the engine build never reads may be inherited: the HTTP server's `E4B_TOKEN`, `E4B_HOST` and
  `E4B_PORT`, and the rental launcher's `E4B_RENT_*`. The refusal names the switch and never prints its value.
- Path-valued entries in the engine's own report (an adapter's path, a lever's artifact directory) are recorded by name
  only.
- The receipt names the arena by base name and the sha256 of its first MiB, and the calibration file by base name and
  sha256. Neither appears by path, and the recorded server environment names a local model directory by its name only
  (`local: <name>`), as `first_run.json` does.
- A serve receipt's run id names the context and concurrency (`...-serve-all-vram-c4096x1-...`).
- `bench/serve_validate.py` keeps its command line and receipt format, and now calls the same measuring code.
- The serve executor steps the engine as experts4bit-qlora's server loop does, opening and closing a trace row around
  each step, so a one-sequence run's decode step is traced. `scheduler.run_until_idle()` steps without opening a row,
  so the trace stayed empty. A missing or empty trace is now recorded as a note, never a crash. A GPU smoke on an RTX
  4090 found this; the same gap was in `bench/serve_validate.py` before the executor moved into the package.

### Adapters: the MoE manifest records the quantized base it was trained on (`backend-adapter/2`)

A `backend-adapter/1` manifest records the setup and the base revision, but not what the base was quantized to. A
server rebuilding the base therefore cannot tell an fp32 expert absmax from the double-quantized one training uses by
default. With an experts4bit-qlora that defines `backend-adapter/2`, the MoE executor now writes it. It records:
- the expert absmax form, the attention recipe, the fused q/k/v count and the versions, copied from
  `prepare_qlora_training`'s report;
- the LoRA scale the modules held;
- a `qfp-v1` fingerprint of the quantized base (experts always; attention when the adapter trains it), read where
  training kept the stacks (`expert_residency`: device or host);
- the run id and the time written.

Everything `/2` records is checked right after the model is prepared, BEFORE the first training step, so a run that
could not record it is refused as not run instead of failing after training. The base is fingerprinted then too, and
again at save: if any scope differs, the base changed during training and the manifest is not `/2`. A run recorded as device whose stacks are evicted is refused, never fingerprinted from a
host copy nobody recorded. Trained tensors are never discarded over manifest metadata: if `/2` can only fail at save,
the adapter is written as `/1`, its manifest and the receipt say why (`v2_refused`), and the run reads `SAVED_V1`
(its measurements still count as planning evidence). `load_adapter` fingerprints the rebuilt base and refuses a
mismatch, and reading a `/2` directory goes through experts4bit-qlora's own verifier.
`loggetta inspect --adapter DIR` prints it all. With an older experts4bit-qlora the manifest stays `/1`, as before.
The dense executor writes `/1`.

### Plans: a bound on the load's host memory, warned and graded, never a refusal

The plan's host estimate prices the steady state. Loading also maps the checkpoint's weight files, and their pages count
in the process's RSS: under WSL2 and in containers they count against the memory cap even when the kernel calls them
reclaimable. On one WSL2 run the host estimate was 6.38 GiB and the measured peak RSS 13.1 GiB, against 14.5 GiB
available.
- `plan` reads the checkpoint's weight-file sizes without downloading them: from the Hub API's file metadata, a local
  directory, or the local Hub cache, recording only the repository id or the directory's name, never a home path. When
  none answers, nothing is priced and the plan is unchanged.
- The load-time bound is the host estimate plus those bytes, labelled an inferred bound, not a measurement: the
  quantize transients are not priced, so each receipt grades whether it held. It is recorded in the plan's `provenance.load_bound`. When it exceeds what this machine had available at plan time, a
  named warning gives both numbers and where the available figure came from. It never refuses a plan.
- `execute` reads what is available again right before the backend loads, and logs the same comparison.
- The receipt's `load_bound` grades the bound against the run's measured peak RSS (VmHWM, the load included): `held`
  is true, false, or null when no peak was measured. The summary prints it with each number's basis.

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

### Evidence: the planner's held-out split

- `evidence/planner-split/`: the planner's frozen derivation / held-out split. Every plan/measured pair on file is tagged SEEN (a
  planner constant was set or tuned after it) or UNSEEN. The held-out side has one row, the resident Qwen3-14B row at 4,096 tokens, so every
  family × card is unmeasured and no band is printed yet. The band reads only this file.
- `evidence/planner-split/SPLIT-v2.json`: the planner split's version 2 (`SPLIT.json` stays frozen and unchanged). It adds eight more
  held-out rows: LFM2-8B-A1B and ERNIE-4.5-21B-A3B, each at 1,024 and 2,048 tokens, on an RTX 5090 and an RTX 4090 (experts4bit-qlora
  `bench/ho2/`, pjordanandrsn/experts4bit-qlora#1969 and pjordanandrsn/experts4bit-qlora#1987), all UNSEEN and all COVERED. Four family × card cohorts now hold the 2 rows a band needs; the band itself
  comes in a later change. Every row cites its public receipt copy by sha256 beside the private original. These are device-placement rows only, and
  each row's loss branch is inferred, not counted.

### Receipts: a run whose device check had no driver sample reads UNMEASURED, not OK

A training run's device peak is checked against the plan through `nvidia-smi`'s per-process memory. Under WSL2, and in
some containers, `nvidia-smi` lists no per-process memory, so the check took no sample (`driver_samples` 0). The run
still read `OK`, a status claiming a check that never happened. Such a run now reads `UNMEASURED`:
- The receipt's `measured.driver_unmeasured` says why: no `nvidia-smi` on PATH, every read failed, or reads that never
  listed this process.
- The summary prints "device check NOT RUN".
- `execute` and `train` exit with code 4, which `--help` lists: trained, but the device check took no driver sample.
  `scripts/first_run.sh` maps it to its own exit 7, "trained; the device check did not run", and records
  `device_check` in `first_run.json`. Its exit 4 (the plan was refused) is a different interface.
- The run trained normally, so its adapter is saved. A fallback to a `/1` manifest is still reported.
- Its allocator slack still counts as planning evidence, but it never teaches the CUDA-context term, which needs a
  driver peak.

An integrity failure (`ALARM`) still takes precedence.

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

### Receipts record the experts4bit-qlora environment

- An execution receipt's `provenance.e4b_env` records the experts4bit-qlora environment the run started under:
  - the sorted names of every set `E4B_*` variable;
  - the sha256 of the sorted `name=value` lines, so two receipts compare without their values;
  - the values of the switches that change numerics or speed (`measure.E4B_VALUE_KNOBS`, for example
    `E4B_ABSMAX_DQ` and the `E4B_PAGED_*` and `E4B_SERVE_*` levers).
- Every other variable is recorded by its name only, in the record and in the hash alike. Only the listed switches
  contribute a value, so a receipt never carries a host name, a path, a private model id or a secret, nor a hash that
  could guess one. It is an allowlist, not a name pattern.
- Two committed receipts ran different `E4B_ABSMAX_DQ` settings under an identical setup and command line, and differed
  by 39% in seconds per step. Nothing recorded the difference until now.
- It is recorded only: every committed receipt plans identically with the key added.

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

### Receipts record the allocator setting

- An execution receipt's `provenance` records `pytorch_cuda_alloc_conf`, the `PYTORCH_CUDA_ALLOC_CONF` the run started
  under (null when unset). It moves the allocator's reserved-but-unallocated bytes, which the planner learns its
  reserve from, so a reader can now see which setting a measured reserve came from. It is recorded only: every
  committed receipt plans identically with the key added.

### Receipts: which loss branch the training forwards took, counted

A receipt now records, in `engaged.chunked_lm_loss`, which loss branch the run's training forwards took. experts4bit-qlora
chooses the chunked LM loss or the stock one per forward under a size gate, and the receipt counts each, as the change
in its counters over this run's training loop: chunked, under the gate (stock), stock otherwise. It also records the
gate's threshold, the chunk size and the head width the gate read. `branch` reads `chunked`, `stock` or `mixed`, so a
row near the gate states its branch from its own counters instead of having it inferred from its peak. Recording never
fails a run that trained: an error is written into the record instead.

## 0.6.2 — 2026-10-22

**0.6.2.** Two fixes that make first runs work. When the default torch from PyPI is built for a newer CUDA than
the NVIDIA driver supports, `plan` used to call the machine feasible and `execute` then failed in torch. `plan` now
refuses, with the reason and the install order that fixes it, and `inspect` shows what torch reports. And the Alpaca
demonstration (training without `--dataset`) no longer stops at row 284 on runs longer than about 45 steps. Upgrade if
you install Loggetta on CUDA machines or train the demonstration. Plans on a working install are unchanged apart from
a recorded `torch` entry. Floors unchanged.

### A CUDA plan that the installed torch cannot run is refused, and inspect shows why

`pip install loggetta` can install a torch built for a newer CUDA than the driver supports. torch 2.14.1+cu130 needs
driver 580; on driver 575, `nvidia-smi` lists the GPU but `torch.cuda.is_available()` is False. `inspect` and `plan`
read the GPU from `nvidia-smi` only, so they reported a feasible plan, and `execute` then died in
`torch.cuda.set_device` ("driver too old"). With an NVIDIA GPU discovered, the hardware probe now asks the installed
torch, in a child process, for its version, the CUDA it was built for (`torch.version.cuda`) and whether it can use the
GPU, and keeps torch's own warning. This process still imports no torch. `inspect` prints those facts next to the
driver. `plan` refuses a CUDA plan when torch is installed and says it cannot use the GPU, naming torch, its CUDA, the
driver and the driver that CUDA needs. The remedy it gives: a fresh virtual environment, a matching torch installed
first from its own index, then Loggetta. It also says that installing another torch over the first does nothing, because
pip counts the same version as already satisfied. When the driver already meets the build's minimum, the refusal says so
and shows `CUDA_VISIBLE_DEVICES` and torch's note instead of suggesting a reinstall, since the cause is GPU visibility
(the variable, or a container without GPU access). A torch that could not be asked is a warning, not a refusal. Profiles
saved before this change still load, and plans made from them are unchanged. The README has an install note on driver
versus torch build. (#84)

### Training: the Alpaca demonstration runs past row 284

Training without `--dataset` uses the Alpaca demonstration, and it stopped in data preparation at row 284
("column 'output' must contain non-empty text"). That is the first of the 28 rows in `tatsu-lab/alpaca` with an empty
output, so any demonstration run that needed more than about 23k tokens (some 45 steps at `--seq 512`) failed before
loading weights. The default 20 steps stayed under that, so the quickstart was unaffected. The demonstration now skips
exactly those rows and records how many it skipped on every pass (`rows_skipped_empty_output` in the report and the
data profile). It is pinned to the dataset revision it has had since 2023
(`dce01c9b08f87459cf36a430d809084718273017`), so its token stream is reproducible. A dataset you supply is still
validated row by row, and an empty output there is still refused. A pass that yields no usable row now refuses by
name instead of repeating forever (#86).

## 0.6.1 — 2026-10-21

**0.6.1.** A patch for dense families that experts4bit-qlora describes as MoE with zero MoE layers, gemma4 among
them. The MoE backend now refuses such a model at admission, and the dense backend reads gemma4's per-layer
`head_dim` and refuses its layer differences by name, so `plan` says why gemma4 is not yet plannable instead of
failing. PyPI's description writes its links in a plain-text-safe form. The rest is tests and CI. Floors unchanged.

### Plans: a model with no MoE layers is refused by the MoE backend at admission

experts4bit-qlora describes some dense families (gemma4 among them) as a MoE topology with zero MoE layers. The MoE
backend admitted that description, so it owned the plan, and the plan was refused later at pricing with "no fused expert
stack at the loader's expert path", plus a note that experts still train. The dense backend's own refusal never showed.
The MoE backend now refuses a topology without expert stacks ("no MoE layers (0/N): not a Mixture-of-Experts model").
The dense backend owns the model when it can describe it. When it cannot, as with gemma4 today, the plan lists each
backend's own reason (#78).

### Dense plans: gemma4's per-layer head_dim is read per layer, and its layer differences are refused by name

transformers 5.20 declares gemma4's `head_dim` per layer (`per_layer_config`) and raises on the global attribute. The
dense backend read it globally, so describing `google/gemma-4-E2B-it` failed with `AmbiguousGlobalPerLayerAttributeError`.
It now reads each layer's `head_dim`, classifies that layer's projections at that width, and then refuses with each
difference that pricing, which assumes identical decoder layers, cannot represent: the per-layer head_dim (256 in 28
layers, 512 in 7), layers 15-34 without `k_proj`/`v_proj` (cross-layer KV reuse), and the wider MLP in those layers.
Tested on the model's own `config.json` at a pinned revision. gemma4 is still not plannable; the refusal now says why
(#78).

### Docs: PyPI's description writes its links in a plain-text-safe form

PyPI also serves the description as raw text (its JSON API), and some tools read it that way. Every link in `PYPI.md`
is now plain-text safe. Five of them ended in ")" or "." straight after the URL, and the other nine in the markdown's
own closing parenthesis, so a plain-text autolinker took the punctuation into the URL and requested paths that 404. All
fourteen now read `text: <url>`. Angle brackets end the URL, and markdown still renders them as links. The rendered page
is unchanged; the fix reaches PyPI with the next release.

### Tests: a default config that fails transformers' own validation is skipped by name

The refusal test for unsupported default configs built each type with `AutoConfig.for_model`. transformers 5.20.1's
default `gemma4_assistant` config fails its own validator ("hidden_size_per_layer_input must be 0, got 256"), so the test
failed before any loggetta code ran. Building the default config is now guarded: such a failure skips with transformers'
version and the validator's message, as a type missing from `CONFIG_MAPPING` already did. What loggetta does with a
config that builds is still asserted. Tests only; no runtime change.

### CI: the private-marker guard reports as `private-marker-guard`

The guard's job reports under its own name instead of the generic `guard`, and only pull_request runs carry it;
push runs report as `private-marker-guard (push scan)`. Every branch push is still scanned.

## 0.6.0 — 2026-10-11

**0.6.0.** On an AMD machine, discovery now finds the GPU and `plan` refuses it by name, instead of reporting no
GPU. With experts4bit-qlora 0.53.0 or later, the grouped_nf4 MoE backward is priced once, by the runtime's own
estimate. Dense training's frozen-parameter digest takes a rank-0 parameter. The rest is tests (dense adapter
arithmetic, round-trips and cached generation) and the HO1 held-out evidence. Floors unchanged.

### An AMD GPU is discovered and refused in words, instead of reported as no GPU

GPU discovery read `nvidia-smi` only, so on an AMD machine every plan was refused with "no GPU at index 0". With no
NVIDIA GPU found, the probe now asks amdsmi's Python API, then `rocm-smi --json`, then whether `/dev/kfd` exists, and
records each AMD GPU with `vendor="amd"`, its gfx arch in the new `GPU.arch` field, and memory where reported. The
compute capability stays unknown: ROCm's capability tuple is a gfx number (gfx942 reports `(9, 4)`), and every
backend's rules are written for sm numbers. `plan` then refuses by name: "AMD GPU detected (AMD Instinct MI300X,
gfx942); not yet supported: no AMD card has run the suites". `inspect` shows the arch and says it is not yet
supported. Profiles saved before `GPU.arch` existed still load.

### The grouped_nf4 backward line defers to experts4bit-qlora when it prices the branch itself

experts4bit-qlora#1532 (for #1526) prices the grouped_nf4 MoE backward inside its own `activations` item. Loggetta now
checks whether the installed experts4bit-qlora has that branch (`e4b_prices_gnf4_backward`), and if so adds no line of
its own. For an older release it keeps its line, as before. Totals are never counted twice either way: against the new
estimate, the old line's excess over `activations` is not positive, and `tests/test_gnf4_training_terms.py` checks that.

### Dense training's frozen-parameter digest takes a rank-0 parameter

`dense_train.frozen_digest` hashed each frozen first- and last-layer parameter through a byte view. A byte view of a
rank-0 tensor wider than a byte refuses (`self.dim() cannot be 0 to view Float as Byte`), so a model with a scalar
frozen parameter could not be checked. It now flattens a contiguous copy before the view. For a ranked tensor the bytes
are the same, so every existing digest is unchanged: `tests/test_frozen_digest_rank0.py` pins today's ranked digests (RNG-free fixtures) in
fp32, bf16 and fp16, and checks a rank-0 parameter in each against an independently computed digest.

### Dense adapter arithmetic contracts

Add CPU integration checks between the actual PEFT linear adapter and e4b's native LoRA wrapper: bitwise fp32 outputs, gradients and accumulated AdamW state across projection shapes, ranks, scaling and checkpoint modes. Pin the distinct delta-add rounding of plain PEFT under a bf16 base with fp32 adapters, and the PEFT bitsandbytes wrapper's pre-add rounding with autocast off. No production arithmetic or default changes.

### Dense accumulated training parity

Add CPU fp32 checks that resident and streamed dense training preserve bitwise-equal losses, adapter updates and complete AdamW state after each accumulated update, including uneven supervised-token masks and skipped empty steps. A corrupted streamed host-home mutant fails the same equality check. These tests establish recipe semantics on tiny local Llama and Qwen3 fixtures, with no capacity or timing claim.

### Check dense adapter artifacts across streamed placement and storage dtypes

Exercise real Llama and Qwen3 adapter training, export and public inference reload
with resident or streamed bases and fp32 or bf16 adapters. Require bitwise tensors
and logits, and reject a missing adapter tensor even when its file checksum passes.

### Check cached generation after dense adapter reload

Require bitwise resident/streamed logits, KV cache tensors and greedy generated
tokens after saved Llama and Qwen3 adapter reload. A corrupted cached-prefix control
must fail the same comparison. Older e4b CPU inference retains the explicit #1539
expected-failure marker; fixed runtimes exercise the streamed generation path.

### Tests: the fused q/k/v training projection's term reaches the plan, and its knob turns it off

experts4bit-qlora#1560 turns the fused q/k/v training projection on by default. Its estimate prices the fused
projections' fp32 absmax: 3 B per 64 q/k/v values, 23.6 MB on Qwen3-30B-A3B. Loggetta defers to that estimate.
`tests/test_fused_qkv_term.py` pins four things: the line is in the plan under the default, `E4B_TRAIN_FUSE_QKV=0`
drops it, it appears only where the run fuses (NF4 attention, trained attention, the grouped kernel), and the plan's
`estimate_env` records the knob. The tests skip against an experts4bit-qlora that predates the term. No code changes.

### Registered: HO1, training plans on cards Loggetta was never calibrated on

`bench/HO1-PREREG.md` registers the rules, the row set and a prediction for re-planning committed experts4bit-qlora
training runs on the RTX 5090 (and 4 H100 rows, reported only) with no receipt on file. `bench/ho1_rows.py` fixes the
rows from a pinned experts4bit-qlora commit, and `evidence/2026-10-10-heldout-cards/rows.json` holds them, without plans.
No plan exists yet. No code changes.

### HO1 read: on the RTX 5090, one of 14 setups is under its plan

`evidence/2026-10-10-heldout-cards` re-plans committed experts4bit-qlora training runs, as registered in #59. The
plans use Loggetta 0.5.0 with no receipt on file.
- One RTX 5090 setup is under its plan: Qwen3-30B-A3B at 4096 × 1, fused kernel, fp32 adapters, with driver/plan
  1.023.
- The exact estimate is short on 5 setups, all fused Qwen3-30B.
- Five setups that ran are refused.

Scripts: `bench/ho1_replan.py` and `bench/ho1_report.py`. No code changes.

README and PYPI now say, beside the 0.5.0 in-sample line, that the held-out check found one RTX 5090 setup over its plan.

### HO1 addendum (post hoc): the over-plan arms ran without pad buckets

`evidence/2026-10-10-heldout-cards` gains a post-hoc addendum. HO1_UNDER stays the registered verdict.
- **The over-plan arms:** all 9 padded the LoRA delta without buckets, by an `NF4_QLORA_PAD_BUCKETS=0` override or on
  grouped-nf4-gemm 0.41.0. The planner does not model that path.
- **The detector:** `bench/ho1_rows.py --posthoc` now flags such arms from what they recorded, pinned by
  `tests/test_ho1_unmodelled.py`.
- **The post-hoc read:** no setup is under.
- **Still open:** at a real 4,096 tokens per micro-batch, the allocator estimate is short (1.048 fp32, 1.133 bf16),
  tracked in experts4bit-qlora#1526.
- **The 0.5.0 line:** README and PYPI now say every over-plan run used an unmodelled setting.

### CI: a private-marker guard, as in the runtime and kernel repositories

A workflow refuses private-lane material in tracked files: the same patterns and canary self-test as grouped-nf4-gemm's
guard. The current tree is clean against all of them. No code changes.

## 0.5.0 — 2026-10-10

**0.5.0.** A training plan for a model with no receipts of its own no longer borrows a sibling model's reserve. In
sample, no committed training plan now sits under its measured peak. Plans also record the backend switches their
estimate read, and `execute` warns when the running process differs.

- **Training plans no longer borrow another model's reserve slack.** A model nothing measured on this GPU is priced at
  the card's worst measured training slack, never below 20 %. In sample, granite-3.1-3b-a800m goes from 1.014 of its
  driver peak (under) to 0.898. Near a budget, such a plan may now pick a smaller setup.
- **Plans record `estimate_env`** with experts4bit-qlora 0.52.0 or later, and `execute` warns when the running
  process's switches differ. With an older backend nothing is recorded and the check is skipped.
- **Dense plans:** the activation item takes the larger of its two branches, so an estimate no longer falls when one of
  its terms rises. Four Qwen3-14B plans at 2,048 tokens rise 24.0 MiB; none falls.
- **Project files and docs:** `CITATION.cff`, `SECURITY.md`, `CONTRIBUTING.md`, issue templates, the PyPI project
  links, a limits box, and the claims re-checked against the committed evidence.

**Requirements.** Unchanged: experts4bit-qlora 0.49.0, grouped-nf4-gemm 0.42.0 and PEFT 0.21.2 or later.
experts4bit-qlora 0.52.0 adds the `estimate_env` record.

### Training plans no longer borrow another model's reserve slack (#43)

- **What changed.** A training plan's allocator reserve comes from receipts for this model on this GPU (or this model
  elsewhere through an anchor, as before). A model with none gets a conservative bound: the largest training slack
  measured on this GPU, and never below the 20% default. It is labelled `policy`. A sibling model's number for the same
  setup is no longer used. Serving keeps its cross-model tiers: its slack is 0.06–1.5%, and every serving plan sat
  over its driver peak (#29).
- **Why.** granite-3.1-3b-a800m, with its siblings' receipts on file, was planned with OLMoE's measured 0.222 (its own:
  0.228) and sat under its driver peak at 1.014.
- **What it costs.** In sample (`evidence/2026-10-09-moe-plan-vs-driver-no-borrow`), no plan is under its driver peak.
  granite-3.1 goes from 1.014 to 0.898 and granite-4.0-h-tiny from 0.896 to 0.796. A model nothing measured is priced
  at the card's worst training slack.
- **The rest of granite's gap is attributed** (`evidence/2026-10-09-a2000-granite-train-residual`). Its training peak is
  the loss, where three fp32 logits-sized tensors are live: 12 B per logit, against the 10 B that experts4bit-qlora's
  MoE estimate prices. That is the miss loggetta's dense estimate fixed in #27. The adapter gradients the estimate
  prices are not live at a loss peak and offset most of it: +23.5 MiB on today's stack.
- **Dense plans are unchanged:** byte-identical on 216 cases, with and without all 62 committed receipts on file.
- **Tests.** `test_training_never_borrows_another_models_slack` (fails before, passes after). Two reserve tests'
  receipts now name the model they stand for.

### Plans record the backend switches their estimate read; execute warns when the run differs

- **What changed.** A feasible plan's provenance records `estimate_env`: the environment switches the selected
  backend's estimate read, with their values in the planning process. For experts4bit-qlora that is what its own
  accessor (`recipe.estimate_env`) reports. `execute` compares them with the running process's before anything loads.
  On a difference it logs a warning naming each switch, planned and running, and it records the comparison in the
  receipt's `provenance.estimate_env`.
- **Why.** experts4bit-qlora's training estimate reads `E4B_CHUNKED_LM_LOSS` (experts4bit-qlora#1491), which decides
  whether the loss is priced chunked or whole. `loggetta train` plans and runs in one process. A plan written by
  `loggetta plan --out` and run by `loggetta execute` in another process could price a loss the run does not take, and
  nothing would say so.
- **Unchanged where nothing is recorded.** A backend release without the accessor records nothing and the check is
  skipped. Older plans run as before. loggetta still reads no environment variable itself.
- **Tests.** `tests/test_estimate_env.py`.

### Dense plans: the activation item is the larger of its two branches as returned

- **What changed.** The dense activation item chose its branch with `T × layer_work > loss_bytes`, then returned the
  layer branch scaled by `ACTIVATION_COEFFICIENT` (1.178). The comparison left the coefficient out, so a larger loss
  workspace could switch it to the smaller branch: an estimate that fell as one of its terms rose. It now computes
  both branches as returned and takes the larger.
- **The fallback.** For an experts4bit-qlora without `chunked_loss_bytes`, the fallback's bytes per chunk logit are now
  `CHUNK_LOSS_BYTES_PER_LOGIT` (12, measured in experts4bit-qlora #1504), up from a literal 10.
- **Effect, in sample on the 36-case dense matrix.**
  - With today's experts4bit-qlora, four Qwen3-14B plans at 2,048 tokens rise 24.0 MiB: the cases the old comparison
    under-priced. No plan falls.
  - With experts4bit-qlora #1505 (12 B per chunk logit) as well, every changed plan rises (+178.1 MiB). Qwen3-32B at
    2,048 tokens, which #1505 alone would have lowered by 185.1 MiB, is unchanged.
- **Tests.**
  - `test_a_larger_loss_term_never_lowers_the_activation_estimate`, at Qwen3-32B and 2,048 tokens: it fails before the
    fix and passes after.
  - The DQ9 prior-estimates snapshot is unchanged. Its two affected rows get an explicit activation correction in
    `tests/fixtures/dq9/activation-max-correction.json`, and the test checks that they now take the layer branch.

### Tests: every run an evidence log names has a committed receipt, or a note says why not

`tests/test_evidence_receipts.py` reads every committed log under `evidence/` for the run IDs it names, as loggetta's
`receipt: <path>.json` line or `bench/train_residual.py`'s `run <id> status` line. It requires a committed
`<run_id>.json` under `evidence/`, or a `no receipt: <run_id>` line in the directory's Markdown. `receipts/` is
gitignored, so a receipt has to be force-added. The step was missed twice (#49, #48), and both times a reader found it.
On main before #55 the test fails with exactly those runs. The granite directory now notes its lost run G. No code
changes.

### Docs: the memory headline cites the audit without reserve borrowing

README and PYPI's planner memory check pointed at the audit re-run after the `grouped_nf4` term, where one RTX A2000
training plan sat under its measured peak (1.014×) because it borrowed another model's reserve. Training plans no
longer borrow (#49). The headline now points at that audit (`evidence/2026-10-09-moe-plan-vs-driver-no-borrow`),
where no A2000 training plan is under its measured peak, in sample. No code changes.

### Docs: claims brought back in line with the committed evidence

A claims audit checked about 330 statements in the README, PYPI.md and docs/ against the receipts. This fixes the ones
that drifted:
- RESULTS number drifts and roundings, including the 6.64 → 5.64 replan figure, which comes from the original receipt.
- The prefill-graph pool at 30B is +0.56 GiB with NF4 experts. The +3.3 GiB cited before was the int4 stack. The
  planner's explain string is fixed too.
- ARCHITECTURE:
  - `train.py` reaches the grouped kernel through its arena path.
  - gnf4 0.39.0 first shipped the route interface.
  - There are two backends.
  - The interface list is the main one, not all of it.
  - Training never borrows another model's reserve.
  - Serving reads a second setup field (`placement`).
- SESSION-REPORT is left as dated and gains a correction note.
- README/PYPI say "estimates" for the OLMoE 0.2 GiB figure, and scope the no-plan-under-peak claim to today's planner.

### Docs: granite-3.1's 32 MiB peak drift between two stacks is recorded, not attributed

`evidence/2026-10-09-a2000-granite-train-residual/STACK-DRIFT.md`. The same setup on the same RTX A2000 peaked
3,163.9 MiB on the 2026-10-04 stack and 3,131.5 MiB today. 27.6 of the 32.4 MiB is already present after load, the
direction is safe, and the estimate covers both runs. Four components moved at once (experts4bit-qlora 0.44 → 0.51,
grouped-nf4-gemm 0.37 → 0.44, torch 2.8 → 2.11, transformers 5.18 → 5.19), so it is not attributed. A drift the other
way, or past the estimate, would reopen it. No code changes.

### Docs: project files and package links

Adds `CITATION.cff` (GitHub's "Cite this repository"), `SECURITY.md`, `CONTRIBUTING.md` and issue templates, and fills
the PyPI project links (homepage, changelog, release notes, the research Space on Hugging Face, and the runtime and
kernel packages) and keywords. "New in 0.4.0" now says what the dense readings found in plain words instead of
internal lane codes. The PyPI badge matches the runtime's and the kernels', and the README links the Hugging Face
Space. No code changes.

### Docs: one tagline across the projects

The README's tagline now reads "Large models. Smaller machines.", the same line as the research Space, the ML site and
the profile banner. No code changes.

### Docs: a short limits box near the top

README and PYPI.md gain a three-line limits box under the opening paragraph: plans are estimates, the released training
path is single-GPU MoE on Linux with a supported NVIDIA CUDA GPU, and dense models are planned but not yet supported for
training. It repeats what the pages already say further down. No code changes.

## 0.4.0 — 2026-10-09

**0.4.0.** Dense models can now be planned, and run behind a development flag. MoE training estimates now price the
`grouped_nf4` kernel's backward pass. Serve plans for one sequence say which speed-ups the server runs. And resident
`grouped_nf4` training runs again with experts4bit-qlora 0.49.0 or later.

- **Dense models are development-gated.** Loggetta describes and plans dense decoder models. Their plans are
  estimates checked in sample: out of sample (DQ7, VOID) they missed, and the DQ10 reading is pending. Dense training
  runs only with `--allow-development-executor` until DQ8's 24 GB reading passes. Dense training is not supported yet.
- **The DQ10 reserve policy is opt-in.** It is a registered hypothesis for registered dense runs. Shipped defaults do
  not change.
- **The MoE training estimate prices the `grouped_nf4` backward pass.** OLMoE estimates on the RTX A2000 were about
  0.2 GiB under the measured peak; in sample they are now 90–114 MiB over. Near a budget, a plan may now pick the
  reference kernel or host residency where it picked resident `grouped_nf4` before.
- **Single-stream serve plans** name the B=1 levers a default server runs for the model's family, and quote a decode
  figure only from a receipt of the same setup.
- **Fix: resident `grouped_nf4` training no longer stops before its first step** with experts4bit-qlora 0.49.0 or
  later (#47). Loggetta 0.3.x raised `AbsmaxCompressedError` at its frozen-expert digest; the workaround was
  `E4B_ABSMAX_DQ=0`.

**Requirements.** grouped-nf4-gemm 0.42.0 or later (was 0.41.0). PEFT 0.21.2 or later is now a base dependency. The
experts4bit-qlora floor stays 0.49.0.

### MoE training: the `grouped_nf4` backward pass (#43, #44, #45)

- **What changed.** A `grouped_nf4` training estimate carries a new device line, `grouped_nf4 MoE backward (above the
  activation item)`: how much `boundaries + padded LoRA delta + fused workspaces` exceeds experts4bit-qlora's activation
  item (`boundaries + max(logits, 2 x one layer)`), or nothing when it does not.
  - **Padded LoRA delta:** grouped-nf4-gemm's `_lora_delta_padded` holds `rows x (H + first_out + I)` in the adapter
    dtype. Its rows are data-dependent, so they are a bound in shape, not a measured constant: `min(E, T x top_k) x T`
    for the single padded block, capped where the `auto` route would take the loop (2 GiB padded block), on the
    single-block ladder for fp32 adapters (grouped-nf4-gemm 0.44.0's `NF4_QLORA_SINGLE_LADDER=auto`: at most 1.5625x),
    and `2 x T x top_k` once a call has 16,384 routed rows and pads by buckets.
  - **Fused workspaces:** `5 x T x top_k x first_out` bf16.
  - The expert LoRA gradients that are live there are already priced, in the `adapter gradients` line, and are not
    counted again.
  - grouped-nf4-gemm exposes no sizing helper, so its route constants and ladder are mirrored, and a test pins them
    against the installed package. The kernel's run-time overrides (`NF4_QLORA_*`) are listed as unmodelled.
  - The `grouped-nf4-gemm` floor is now 0.42.0, the first release whose bucketed padding is the default.
- **Why.** loggetta#44 replayed the allocator at OLMoE's `grouped_nf4` training peak on the RTX A2000: the peak is in
  an MoE layer's backward, with 500.9 MiB of padded delta and 167.1 MiB of fused workspaces live, where the estimate
  took the loss branch. Estimates were 195–219 MiB under the allocated peak.
- **In-sample only.** Against the committed OLMoE `grouped_nf4` receipts (#29's A2000 and FP1's RTX 5090, #44's
  replays) the estimate now sits 90–114 MiB over the allocated peak, the cost of a bound. The reference kernel, the
  Qwen3-30B receipts (the loss branch is larger) and the granite receipts do not move; the granite runs stay 56–69 MiB
  under, which this term does not explain. `bench/gnf4_terms_in_sample.py` prints the table.
- **Plans near a budget may change setup.** A `grouped_nf4` training plan is now larger, so where resident
  `grouped_nf4` used to fit just barely, a plan may now pick the reference kernel or host residency.
- **Dense plans are unchanged:** byte-identical on a 36-case matrix, before and after.
- **Tests:** `tests/test_gnf4_training_terms.py`, `tests/test_dense_unchanged_by_moe_terms.py`.
- **Plans against the driver's peak, after this change.** `bench/plan_vs_driver.py --replan` on the committed training
  receipts (`evidence/2026-10-09-moe-plan-vs-driver-after-gnf4`, in-sample): with no receipts on file, every OLMoE plan
  is over its driver peak (driver / plan 0.950–0.987, from 1.001–1.105). One plan is still under: granite-3.1-3b-a800m
  replanned with its siblings' receipts on file, at 1.014, because its reserve line borrows OLMoE's (#43, item 2).

### MoE plans against measured driver peaks (report only, #29)

`bench/plan_vs_driver.py` reads every committed receipt with a plan total and a driver peak, and can replan the
training receipts with today's code (`evidence/2026-10-08-moe-plan-vs-driver`).

- **Serving plans hold:** all 22 are over their driver peak, including SV5–SV7's registered RTX 4090 plans.
- **MoE training plans on the RTX A2000 do not:** today's OLMoE plans sit under the driver peak (driver / plan up to 1.105).
  The allocator estimate is 0.213 GiB under the allocated peak on both placements, and no MoE training residual is
  learned.

No code, coefficient or default changed.

### Single-stream serve plans

- **What changed.**
  - **The levers.** A serve plan for one sequence names the B=1 levers that a default `serve_paged` runs for the
    model's family, as the installed experts4bit-qlora resolves them: its family-scoped fused stack (fused q/k/v and
    three glue folds), with the read it rests on or the read the family lacks. An experts4bit-qlora without the
    family-scoped default reads as "off unless set". Nothing here keeps its own table of families.
  - **The figure.** The plan's performance section quotes a decode figure only from a serve receipt of exactly the
    same setup: same model, GPU name and driver, setup, and experts4bit-qlora and grouped-nf4-gemm versions, at one
    sequence. The figure is labelled measured, with the receipt's ID and the prompt and new-token counts it was
    measured at (a decode step grows with the KV position). Anything less shows nothing: loggetta still predicts no
    throughput, and no figure is interpolated or carried between hosts.
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

### Dense models (development-gated)

- **Dense execution prototype.** Dense training plans dispatch through transformers, PEFT linear LoRA and the
  existing e4b chunked-loss and dense-offload engines. Checkpoint loading reads one safetensors tensor at a time;
  NF4 quantization handles one decoder linear at a time, and streamed codes stay on the host. The selected setup
  is checked against the loaded structure and echoed in a dense ExecutionReceipt. Sampled frozen-layer integrity,
  adapter movement, memory and timing share the measured loop with MoE. Dense adapters use PEFT with checksummed
  export/reload; existing MoE adapters retain their native format. PEFT >=0.21.2 is a base dependency for all installs;
  0.21.2 is the tested floor. Dense execution requires `--allow-development-executor` until a registered capacity
  reading passes (DQ8, 24 GB).
  CPU tiny-model checks cover exact checkpoint reconstruction, resident/streamed loss and gradient equality,
  actual chunked loss and adapter precision on reload. The tiny CUDA/NF4 proof passed; DQ7's capacity reading is
  VOID and reserve calibration remains unlicensed. The e4b floor remains 0.49.0.
- **Dense adapter-training plans.** The dense backend now plans `train` for dense models.
  - **Candidates:** bf16 base resident, NF4 base resident, NF4 base streamed from pinned host memory. Speed order:
    bf16 before NF4, resident before streamed, per experts4bit-qlora's measured DQ lanes.
  - **Estimate:** itemized.
    - *derived:* weights, embeddings/head, LoRA (fp32 r16 α32 on every classified projection by default),
      gradients, AdamW, and experts4bit-qlora's `offload_plan` for streaming.
    - *heuristic:* activations scaled to DQ4's measured slope; the chunked-loss workspace from `chunked_loss_bytes`.
    - Linear biases are counted once, with norms and other parameters.
  - **Checked in-sample:** the activation slope is fitted to DQ4, and against DQ4's measured peaks at Qwen3-32B the
    estimate is +1.9 to +3.1% resident and +3.0 to +8.7% streamed, never under.
  - **Out of sample, it misses** (experts4bit-qlora DQ7, RTX 5090, 2026-10-08; VOID as a lane because one anchor arm
    could not run). Qwen3-14B's allocator estimates were above their peaks (+1.5 to +4.9%); Llama-3.1-8B's fell below
    at 2048 and 4096 tokens (-2.0 to -6.4%). The plan's device total, reserve included, was below the measured driver
    peak on every streamed arm, by up to 2.4 GB, and on resident Llama 4096 by 0.82 GB. None of this is a capacity or
    calibration pass, and dense execution stays behind `--allow-development-executor`.
  - **Refused, in words:** streaming that would free nothing, attention that would change semantics, a sequence past
    the model's positions, int8.
  - `bench/dense_plan_sweep.py` plans 14 pinned configs on stated 16–48 GB cards
    (`evidence/2026-10-08-dense-plan-sweep`; estimator-only, with enforced streamed headroom).
    The unpinned 2026-10-07 table predates #26/#28. Comparing by model name, four feasible cells now refuse:
    Qwen2.5-72B at 32 GB, Qwen3-32B at 16 GB, Gemma-2-9B and Gemma-3-12B at 24 GB. Neither table proves capacity.
- **Reserve slack is learned per backend.** A plan's reserve, including the fallback, comes only from its own backend's
  receipts. The GPU's CUDA context and host baseline stay shared. Receipts that do not name their backend count as
  experts4bit's, so MoE plans are unchanged.
- **A second backend, `dense`, describes dense decoder models without reading weights** (config + a meta-device
  tree). `inspect` and `plan` now reach a dense model through it instead of stopping at the MoE loader's refusal.
  - Every decoder-layer linear is classified by structural role from its shape: attention in/out, MLP in/out. Names
    only break ties, so fused `qkv_proj` / `gate_up_proj` (Phi-3) classify like separate projections; anything left
    over stays frozen and is reported.
  - Parameter counts, tied heads, and attention implementations are reported, including softcapping that SDPA would
    drop (Gemma 2), along with whether experts4bit-qlora's chunked loss covers the class.
  - MoE, pre-quantized and non-decoder models are refused in words.
  - Config, model-build and chunked-loss errors become refusals naming the exception type; refused descriptions
    can be copied or pickled without recursion.
  - Dense training plans and their development executor are described above; other dense workloads remain planned only.
- **Dense pricing corrections** (structural; execution, reserve/context hypotheses and observation licensing do not
  change):
  - **Full-logit loss** is priced at twelve bytes per logit, not ten: the pinned CPU/CUDA census sees three distinct
    fp32 logits-sized tensors live together in the log-softmax backward.
  - **Small frozen weights** that the offload engine keeps on the device below its streaming threshold are counted,
    with their quantization statistics and the trainable LoRA kept separate (#39).
  - **NF4 quantization state** is counted in full: the two private codebooks and the offset, with packed codes and
    nested statistic blocks rounded up for partial blocks (#41).
- **Streamed admission headroom after DQ7.** The VOID DQ7 reading understates device use on every completed streamed
  arm, by up to 2.40 GB, and resident Llama 4096 driver use by 0.82 GB. Dense plans warn, and streamed admission keeps
  at least max(2.4 GB, 20% of the full device estimate) of headroom even when the caller asks for less. Receipts still
  compare the original estimate, without the headroom.
- **The DQ10 memory hypothesis (opt-in).** A scoped reserve policy prices a training reserve and driver overhead
  separately for registered dense runs. Plans, admission and execution carry the frozen hypothesis and the policy's
  SHA-256; an invalid selection or changed pricing is refused before loading. Shipped defaults, allocator pricing,
  streamed headroom and the executor opt-in do not change. Raw diagnostic and holdout observations cannot teach the
  planner an overhead or a capacity calibration.
- **D7 reserve calibration, registered and read.** The proposed D7 dense allocator-reserve calibration is registered
  with six sequence holdouts, model/placement cohorts, byte-level held-out gates and strict observation scopes. Its
  reader has fixed derivation and holdout cohorts, exact integer fractions, both byte-level holdout gates and the
  original 20% reporting; it recomputes DQ7 through the merged reducer and keeps source checksums. Neither licenses a
  planner observation or changes a reserve or execution gate.
- **The docs state the current gates.** The tiny CUDA correctness proof passed; DQ7 stays VOID. The capacity,
  calibration and 24 GB boundary requirements are explicit, the margin is described as enforced policy, and the older
  config-only sweep is labelled historical estimator output.

### Fixes

- **Frozen integrity fails closed.** An empty sampled expert digest now marks a MoE run `ALARM`; checking no frozen
  bytes cannot pass integrity. Normal non-empty frozen expert digests behave as before.
- **Legacy observation licensing with nullable plan metadata.** Receipts with a null plan or null constraints keep
  their existing `licensed_for` scope instead of raising. Raw DQ7/DQ9/DQ10 records and explicit DQ10 policy receipts
  stay excluded from observation import (#37).
- **Resident `grouped_nf4` training no longer stops at the frozen-expert digest (#47, #48).**
  - **What changed.** `_expert_digest` hashes the frozen expert absmax as it is stored. That is the double-quantized
    payload (`<which>_absmax_q`, `_s`, `_off`, `_code`) when experts4bit-qlora compressed it, and the fp32 buffer
    otherwise. Nothing is decompressed. With the fp32 absmax the digest is byte-for-byte the earlier one.
  - **Why.** Since experts4bit-qlora 0.49.0, `enable_fast_train` double-quantizes the absmax by default for resident
    training, and a guard under the old `<which>_absmax` name raises on any use. With loggetta 0.3.x and
    experts4bit-qlora 0.49.0 or later, resident `grouped_nf4` training raised `AbsmaxCompressedError` before step 1.
    The workaround was `E4B_ABSMAX_DQ=0`. Host residency and the reference kernel were not affected.
  - **Evidence.** Reproduced on the released pair (loggetta 0.3.1 + experts4bit-qlora 0.50.0) on an RTX A2000. The
    fixed digest trains the same setup to an OK receipt (`evidence/2026-10-09-a2000-absmax-digest-repro`).
  - **Tests.** `tests/test_absmax_digest.py` builds the model through `prepare_qlora_training`, which calls
    `enable_fast_train`, on a tiny local MoE on the CPU, with experts4bit-qlora's default. It fails before the fix and
    passes after, with experts4bit-qlora 0.49.0 and 0.50.0. The released-backend CI job fails if this test skips.

### Docs

- **The memory headline says what it covers.** README and PYPI's planner memory check (24.54 GiB estimated, 24.34
  measured) is one in-sample case, and they say so. They point to the MoE plan-vs-driver audit, now with the replan
  after the `grouped_nf4` term.

## 0.3.1 — 2026-10-08

**0.3.1.** Documentation only; the package code is identical to 0.3.0. The PyPI page is now shorter than the README and
consistent with it, and its replication link works again: it pointed at an experts4bit-qlora changelog fragment that the
0.49.0 release folded away (#15). The Unsloth comparison is quoted as GPU time, as experts4bit-qlora 0.50.0 quotes it:
Unsloth spends 1.92× e4b's GPU time per step, and 2.80× its wall-clock time on an AMD EPYC 7713 host (#16). It still needs
experts4bit-qlora 0.49.0 or later.

## 0.3.0 — 2026-10-08

**0.3.0.** Train on your own data and keep the adapters. This is the first PyPI release since 0.1.3; 0.2.0 was never
published, and its work ships here. Loggetta now reads your dataset (local files or the Hub), checks and tokenizes it before
any model loads, plans from what it found, trains on assistant tokens with isolated packing and a cosine schedule by default,
and exports a reusable adapter. Upgrade if you want to train on your own data. It needs experts4bit-qlora 0.49.0 or later
(0.48.0 fails under transformers 5.19 and lacks the planner's row and bucket controls).

### Training defaults and planning from the data

- **Isolated packing by default for chat and alpaca data** (`--packing auto|concat|isolated`).
  - Whole examples go into rows (deterministic best fit), positions restart per example, and an example's first token
    and the row padding never train. transformers then keeps each example's attention to itself; the loop passes
    `use_cache=False`, which that isolation requires.
  - Profiles record the packing at the planned seq (rows, fill, truncation), and plans count rows.
  - The per-row attention mask is priced (micro-batch × seq² × 3 B). Its speed cost (no flash attention) is stated,
    not modelled.
  - Models that mix tokens through a recurrent state are refused isolation.
  - Earlier plans keep concatenation.
- **Warmup + cosine learning-rate schedule by default for `train` and `plan`** (`--lr-schedule cosine|constant`,
  `--warmup-steps`).
  - Linear warmup over 3% of the steps, then cosine decay to 10% of the peak.
  - The plan states it, and run reports record the applied rates.
  - Saved plans and direct `Workload` callers keep the constant rate.
- **Assistant-only loss by default for chat and alpaca data (`--loss auto|all|assistant`).**
  - Chat trains each assistant turn's text plus the end-of-turn marker the template closes it with. The marker is
    read from the template.
  - Alpaca trains the response and its EOS.
  - Masks come from character offsets over the rendered conversation. Templates that rewrite text are refused,
    never guessed.
  - Earlier plans and the demonstration keep full-sequence loss.
- **Gradient accumulation averages over trained tokens.** Each micro-batch is weighted by its share of the step's
  trained tokens, so micro-batch × grad-accum splits of the same rows give the same step. Steps with nothing to
  train are skipped and counted.
- **Profiles and receipts record the loss.** Profiles record the loss and its trained-token count, and hash the mask.
  Receipts carry `loss_tokens` and `loss_mask_sha256`.
- **Read the dataset before planning.** Every row of `--dataset` is validated and tokenized before the model is
  described. The plan carries the resulting `data-profile/1`:
  - examples and tokens;
  - length quantiles and a histogram;
  - the tokenizer's identity;
  - a hash of the encoded examples.
- **Plans use the profile.**
  - A plan that would read past the data without `--repeat-data` is refused, with the steps that read it once.
  - `--epochs N` derives the steps.
  - Examples longer than `--seq` are counted: concatenated packing splits them, isolated packing truncates them,
    and the plan states the tokens dropped.
- **`execute` re-tokenizes and checks.** It refuses data or a tokenizer that changed since planning, before any
  weights load. With concatenated packing and full-sequence loss, the packed token stream is identical to 0.2.0's
  for the same seed; chat and Alpaca data now default to isolated packing, whose stream differs.
- Plans without a profile (older plans, the Alpaca demonstration) keep the 0.2.0 behaviour and wire shape.

### Your data and reusable adapters (developed as 0.2.0, never published)

- Accept local JSON/JSONL/CSV/Parquet/TXT files and Hub datasets, with split/config/revision selection.
- Add explicit text, Alpaca, and text-only chat formats, column mapping, seeded shuffle, and opt-in repetition.
- Keep data settings and learning rate in saved plans; preserve old plan defaults and default wire shape.
- Validate and pack data before model loading, with a temporary memory-mapped token cache and token/source hashes.
- Export native runtime adapters and tokenizer files, with exact setup/identity metadata and SHA-256 checksums.
- Add `load_adapter` and strict pre-copy tensor checks; never overwrite existing adapter output.
- Record artifact paths and export failures in run reports, separately from training-step measurements.
- Keep the complete `pip install loggetta` installation, simplify the README, and document the Accelerate comparison.
- Add CPU data, CLI, persistence, and real-LoRA round-trip coverage plus an optional tiny local MoE CUDA test.
- Chat data works with transformers 5, whose `apply_chat_template(tokenize=True)` returns a `BatchEncoding` by
  default; a test now encodes through a real `PreTrainedTokenizerFast`.

Limitations: full-sequence loss only; no automatic held-out evaluation, optimizer resume, PEFT-format export,
first-class dense planning, or server-launch command. CPU tests are not GPU-family validation.

## 0.1.1

- Install the training runtime and low-bit kernel packages by default.

## 0.1.0

- Initial planner, saved execution plans, training orchestration, and measured run reports.
