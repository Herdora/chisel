#!/usr/bin/env python3
"""
Optimized tiled CUDA matmul (shared memory tiling)

Implements C = ReLU(A @ B) using a tiled shared-memory kernel.
Compare against the naïve baseline in run_kernel.py.
"""

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

// Tile size for shared memory
#define TILE 32

__global__ void matmul_tiled_kernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
                                    int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y; // i
    int col = blockIdx.x * TILE + threadIdx.x; // j
    float acc = 0.f;

    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc > 0.f ? acc : 0.f; // ReLU
    }
}

void matmul_tiled_launch(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda() && C.is_cuda(), "expected CUDA tensors");
    TORCH_CHECK(A.dtype() == torch::kFloat32 && B.dtype() == torch::kFloat32 && C.dtype() == torch::kFloat32,
                "expected float32 tensors");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous() && C.is_contiguous(), "expected contiguous tensors");

    int M = A.size(0);
    int K = A.size(1);
    TORCH_CHECK(B.size(0) == K, "A.shape[1] must equal B.shape[0]");
    int N = B.size(1);
    TORCH_CHECK(C.size(0) == M && C.size(1) == N, "C shape must be (M,N)");

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    matmul_tiled_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "matmul_tiled launch failed");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_tiled_launch", &matmul_tiled_launch, "Launch tiled matmul kernel");
}
"""


def _load_ext():
    return load_inline(
        name="matmul_tiled_ext",
        cpp_sources="",
        cuda_sources=CUDA_SRC,
        functions=["matmul_tiled_launch"],
        with_cuda=True,
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )


def matmul_relu(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.device.type == B.device.type, "mismatched devices"
    if A.device.type != "cuda":
        return torch.relu(A @ B)
    ext = _load_ext()
    A = A.contiguous()
    B = B.contiguous()
    C = torch.empty((A.size(0), B.size(1)), dtype=A.dtype, device=A.device)
    ext.matmul_tiled_launch(A, B, C)
    return C


def _benchmark(M: int = 1024, K: int = 1024, N: int = 1024) -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A = torch.randn(M, K, dtype=torch.float32, device=dev)
    B = torch.randn(K, N, dtype=torch.float32, device=dev)

    for _ in range(3):
        _ = matmul_relu(A, B)
    if dev.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    out = timed_call("matmul_tiled", matmul_relu, A, B)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    dt_ms = (time.perf_counter() - t0) * 1000.0
    print(f"[tiled] device={dev}  MKN=({M},{K},{N})  time={dt_ms:.3f} ms  out_mean={out.mean().item():.4f}")


def main():
    print("=== Tiled CUDA matmul: C=ReLU(A@B) ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    _benchmark()


if __name__ == "__main__":
    main()


