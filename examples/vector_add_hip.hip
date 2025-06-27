// examples/vector_add_hip.hip
#include <hip/hip_runtime.h>

extern "C" void launch_vector_add_hip(const float* A,
                                      const float* B,
                                      float*       C,
                                      int          N);

__global__ void vector_add_kernel(const float* A,
                                  const float* B,
                                  float*       C,
                                  int          N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) C[idx] = A[idx] + B[idx];
}

extern "C"
void launch_vector_add_hip(const float* A,
                           const float* B,
                           float*       C,
                           int          N)
{
    float *dA, *dB, *dC;
    hipMalloc(&dA, N * sizeof(float));
    hipMalloc(&dB, N * sizeof(float));
    hipMalloc(&dC, N * sizeof(float));

    hipMemcpy(dA, A, N * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(dB, B, N * sizeof(float), hipMemcpyHostToDevice);

    int block = 256;
    int grid  = (N + block - 1) / block;
    hipLaunchKernelGGL(vector_add_kernel,
                       dim3(grid), dim3(block),
                       0, 0,
                       dA, dB, dC, N);

    hipMemcpy(C, dC, N * sizeof(float), hipMemcpyDeviceToHost);

    hipFree(dA); hipFree(dB); hipFree(dC);
}
