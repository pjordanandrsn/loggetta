### Install: the torch build for your driver, named before torch is installed, and a first-run script

`inspect` now names the PyTorch CUDA build that matches the NVIDIA driver, and `inspect --torch-index` prints only that
build's index tag (`cu130` for driver 580 and later, `cu126` for 525 and later; a Blackwell GPU, compute capability
10.0 and up, gets only builds for CUDA 12.8 or later, so `cu129` below driver 580, and needs driver 570), reading
`nvidia-smi` alone, so it works before torch is installed. A driver too old for every supported torch build, or a machine with no NVIDIA GPU, is a
named refusal (exit 3). `scripts/first_run.sh` runs a first install end to end in a new directory: a fresh virtual
environment, the matching torch FIRST, then the package, `inspect`, `plan`, and 10 training steps on OLMoE-1B-7B-0924-Instruct
at a pinned revision. It writes each step's exit status and seconds to `first_run.json` and ends trained, or with a named
refusal: no torch build fits (3), the plan was refused (4), an install step failed (5), or the run did not train (6).
A local wheel or directory given as `PACKAGE_SPEC` is recorded in `first_run.json` by its name only, never its path, so
a shared summary carries no home directory.
- A refused plan's summary in `first_run.json` is now the refusal's own first reason, verbatim, read from the refused
  `plan.json`. It was a grep of the printed banner, which kept only "NOT FEASIBLE under the requested constraints."
- A local model directory given as `MODEL` is recorded in `first_run.json` by its name only, never its path, and a
  refusal reason never carries the home directory (it reads `~`).
