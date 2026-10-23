### The refusal names CUDA Error 804 correctly

- When torch reports CUDA Error 804 ("forward compatibility was attempted on non supported HW"), the refusal now says
  what happened. The process loaded NVIDIA's CUDA forward-compatibility library (cuda-compat), which NVIDIA supports
  on datacenter GPUs only, and this GPU refuses it.
- It now gives the remedies that work:
  - an image or environment whose CUDA the driver supports natively;
  - the compat directory taken off `LD_LIBRARY_PATH`;
  - a newer host driver.
- Before, a driver that met the build's CUDA minimum sent the user to check GPU visibility (`CUDA_VISIBLE_DEVICES`,
  `docker run --gpus`), which has nothing to do with this error. Seen on a GeForce RTX 3090 with driver 535 and a
  CUDA 12.8 image.
