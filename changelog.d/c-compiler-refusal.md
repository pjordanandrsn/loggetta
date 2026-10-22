### Plans: a machine without a working C compiler is refused by name, before any download

The training and serving kernels are Triton kernels. Triton compiles a small helper against `Python.h` the first time
one runs. On a machine with no C compiler (the CUDA runtime images ship none), a first run installed torch, planned,
downloaded the checkpoint, and then died in training with Triton's "Failed to find C compiler" traceback.
- When an NVIDIA GPU is present, the hardware probe now finds the compiler Triton would use (`CC`, else gcc, else
  clang), and tests it the way Triton does: it builds a shared object against this Python's headers. A name on PATH is
  not enough, because a box can have gcc but no Python headers.
- Only the compiler's name is recorded, never a path.
- When no compiler can build, `plan` refuses with the reason and the fix (install a compiler and the Python headers, or
  set `CC`). `inspect` reports it, and `scripts/first_run.sh` ends with its named exit 4.
- A profile saved before this field reads "not probed" and is never refused for it.
