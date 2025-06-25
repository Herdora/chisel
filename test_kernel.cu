#include <cuda_runtime.h>
#include <iostream>

__global__ void simple_kernel(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    const int N = 1000000;
    const int size = N * sizeof(float);
    
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    
    // Initialize with some data
    float *h_a = new float[N];
    float *h_b = new float[N];
    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }
    
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    
    // Launch kernel multiple times
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    for (int i = 0; i < 10; i++) {
        simple_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    }
    
    cudaDeviceSynchronize();
    
    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    delete[] h_a;
    delete[] h_b;
    
    std::cout << "Kernel execution completed!" << std::endl;
    return 0;
}