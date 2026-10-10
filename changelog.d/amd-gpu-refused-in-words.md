### An AMD GPU is discovered and refused in words, instead of reported as no GPU

GPU discovery read `nvidia-smi` only, so on an AMD machine every plan was refused with "no GPU at index 0". With no
NVIDIA GPU found, the probe now asks amdsmi's Python API, then `rocm-smi --json`, then whether `/dev/kfd` exists, and
records each AMD GPU with `vendor="amd"`, its gfx arch in the new `GPU.arch` field, and memory where reported. The
compute capability stays unknown: ROCm's capability tuple is a gfx number (gfx942 reports `(9, 4)`), and every
backend's rules are written for sm numbers. `plan` then refuses by name: "AMD GPU detected (AMD Instinct MI300X,
gfx942); not yet supported: no AMD card has run the suites". `inspect` shows the arch and says it is not yet
supported. Profiles saved before `GPU.arch` existed still load.
