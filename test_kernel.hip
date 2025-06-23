#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>

// Simple vector addition kernel
__global__ void vectorAdd(float* A, float* B, float* C, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    const int N = 1024 * 1024;  // 1M elements
    const size_t size = N * sizeof(float);
    
    // Allocate host memory
    std::vector<float> h_A(N);
    std::vector<float> h_B(N);
    std::vector<float> h_C(N);
    
    // Initialize input vectors
    for (int i = 0; i < N; i++) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i * 2);
    }
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    hipMalloc(&d_A, size);
    hipMalloc(&d_B, size);
    hipMalloc(&d_C, size);
    
    // Copy input data to device
    hipMemcpy(d_A, h_A.data(), size, hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B.data(), size, hipMemcpyHostToDevice);
    
    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Run multiple iterations for better profiling
    for (int iter = 0; iter < 100; iter++) {
        vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    }
    
    hipDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    // Copy result back to host
    hipMemcpy(h_C.data(), d_C, size, hipMemcpyDeviceToHost);
    
    // Verify results
    bool success = true;
    for (int i = 0; i < 10; i++) {
        float expected = h_A[i] + h_B[i];
        if (abs(h_C[i] - expected) > 1e-5) {
            success = false;
            break;
        }
    }
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Vector Addition Test:\n";
    std::cout << "Elements: " << N << "\n";
    std::cout << "Iterations: 100\n";
    std::cout << "Time: " << duration.count() << " microseconds\n";
    std::cout << "Result: " << (success ? "PASS" : "FAIL") << "\n";
    
    // Clean up
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
    
    return success ? 0 : 1;
}