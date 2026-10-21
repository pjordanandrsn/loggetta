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
versus torch build.
