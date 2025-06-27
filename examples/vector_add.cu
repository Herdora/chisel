// vector_add.cu
#include <cuda_runtime.h>
#include <stdio.h>

extern "C" void launch_vector_add(const float* A, const float* B, float* C, int N);

__global__ void vector_add_kernel(const float* A, const float* B, float* C, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

extern "C" void launch_vector_add(const float* A, const float* B, float* C, int N) {
    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_C, N * sizeof(float));

    cudaMemcpy(d_A, A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    vector_add_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);

    cudaMemcpy(C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
