#!/usr/bin/env python3
"""
Naive CUDA matmul kernel (baseline)

Implements C = ReLU(A @ B) using a naïve CUDA kernel (one thread per C[i,j]
with an inner loop over K). Compare with tiled version in
run_kernel_optimized.py.

Run locally:
  python kandc/examples/gpu_kernel_example/run_kernel.py

With kandc capture (records timing):
  kandc capture -- python kandc/examples/gpu_kernel_example/run_kernel.py
"""

import os
import time
from typing import Callable

import torch
from torch.utils.cpp_extension import load_inline

try:
    from kandc import timed_call
except Exception:
    def timed_call(name: str, fn: Callable, *args, **kwargs):  # type: ignore
        return fn(*args, **kwargs)


CUDA_SRC = r"""
#include <torch/extension.h>

// Naive matmul: one thread computes one C[i,j]
__global__ void matmul_naive_kernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
                                    int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // i in [0, M)
    int col = blockIdx.x * blockDim.x + threadIdx.x; // j in [0, N)
    if (row >= M || col >= N) return;
    float acc = 0.f;
    for (int k = 0; k < K; ++k) {
        acc += A[row * K + k] * B[k * N + col];
    }
    // ReLU
    C[row * N + col] = acc > 0.f ? acc : 0.f;
}

void matmul_naive_launch(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda() && C.is_cuda(), "expected CUDA tensors");
    TORCH_CHECK(A.dtype() == torch::kFloat32 && B.dtype() == torch::kFloat32 && C.dtype() == torch::kFloat32,
                "expected float32 tensors");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous() && C.is_contiguous(), "expected contiguous tensors");

    const int M = A.size(0);
    const int K = A.size(1);
    TORCH_CHECK(B.size(0) == K, "A.shape[1] must equal B.shape[0]");
    const int N = B.size(1);
    TORCH_CHECK(C.size(0) == M && C.size(1) == N, "C shape must be (M,N)");

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    matmul_naive_kernel<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "matmul_naive launch failed");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_naive_launch", &matmul_naive_launch, "Launch naive matmul kernel");
}
"""


def _load_ext():
    return load_inline(
        name="matmul_naive_ext",
        cpp_sources="",
        cuda_sources=CUDA_SRC,
        # We define PYBIND11_MODULE in CUDA_SRC, so do not pass functions
        with_cuda=True,
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )


def matmul_relu(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Baseline: run naïve CUDA kernel if CUDA available; otherwise PyTorch CPU.

    A: (M,K) float32, B: (K,N) float32 → C: (M,N) float32
    """
    if A.device.type != B.device.type:
        raise ValueError("A and B must be on the same device")
    if A.device.type != "cuda":
        return torch.relu(A @ B)

    ext = _load_ext()
    A = A.contiguous()
    B = B.contiguous()
    C = torch.empty((A.size(0), B.size(1)), dtype=A.dtype, device=A.device)
    ext.matmul_naive_launch(A, B, C)
    return C


def _benchmark(M: int = 1024, K: int = 1024, N: int = 1024, device: str | None = None) -> None:
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    A = torch.randn(M, K, dtype=torch.float32, device=dev)
    B = torch.randn(K, N, dtype=torch.float32, device=dev)

    # Warmup
    for _ in range(3):
        _ = matmul_relu(A, B)
    if dev.type == "cuda":
        torch.cuda.synchronize()

    # Timed
    t0 = time.perf_counter()
    out = timed_call("matmul_naive", matmul_relu, A, B)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    dt_ms = (time.perf_counter() - t0) * 1000.0

    print(f"device={dev}  MKN=({M},{K},{N})  time={dt_ms:.3f} ms  out_mean={out.mean().item():.4f}")


def main():
    print("=== Naive CUDA matmul: C=ReLU(A@B) ===")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # Small correctness check (tiny sizes)
    for dev in (["cuda"] if torch.cuda.is_available() else ["cpu"]):
        A = torch.randn(8, 16, dtype=torch.float32, device=dev)
        B = torch.randn(16, 8, dtype=torch.float32, device=dev)
        out = matmul_relu(A, B)
        ref = torch.relu(A @ B)
        max_err = (out - ref).abs().max().item()
        print(f"check[{dev}] max_err={max_err:.3e}")

    # Benchmark
    _benchmark(M=1024, K=1024, N=1024)


if __name__ == "__main__":
    main()


