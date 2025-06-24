// Simple HIP kernel for testing AMD GPU
#include <hip/hip_runtime.h>
#include <stdio.h>

__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    int n = 1024 * 1024;
    size_t size = n * sizeof(float);
    
    // Allocate device memory
    float *d_a, *d_b, *d_c;
    hipMalloc(&d_a, size);
    hipMalloc(&d_b, size);
    hipMalloc(&d_c, size);
    
    // Initialize data (simplified for testing)
    hipMemset(d_a, 1, size);
    hipMemset(d_b, 2, size);
    
    // Launch kernel
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    hipLaunchKernelGGL(vectorAdd, dim3(numBlocks), dim3(blockSize), 0, 0, d_a, d_b, d_c, n);
    
    // Wait for completion
    hipDeviceSynchronize();
    
    // Cleanup
    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_c);
    
    printf("Vector addition completed!\n");
    return 0;
}