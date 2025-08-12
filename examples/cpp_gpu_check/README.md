# C++ GPU Check Example

This example builds a tiny C/Objective‑C++ library that reports whether a GPU is available.

- macOS: uses Metal (`MTLCreateSystemDefaultDevice`)
- Linux: checks CUDA runtime via `dlopen("libcuda.so")` (falls back to NVIDIA driver presence)

## Build

```bash
cd kandc/examples/cpp_gpu_check
make
```

Artifacts:

- macOS: `build/libgpucheck.dylib`
- Linux: `build/libgpucheck.so`

## Run locally

```bash
python3 check_gpu.py
```

Example output:

```
GPU available: True
GPU name: Apple M‑series (or CUDA‑capable GPU)
[check_gpu] step 1/5
...
```

## Run with kandc capture

```bash
# from kandc/examples/cpp_gpu_check
kandc capture -- python3 check_gpu.py
```

This streams logs to your terminal and also saves them under `~/.kandc/runs/<app>/<job_id>/`.


