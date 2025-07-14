// kernel.cu

#include <iostream>

__global__ void add(int *a, int *b, int *c, int n) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < n)
        c[idx] = a[idx] + b[idx];
}

int main() {
    int n = 16;
    int size = n * sizeof(int);

    int *h_a = new int[n];
    int *h_b = new int[n];
    int *h_c = new int[n];

    for (int i = 0; i < n; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    add<<<1, 32>>>(d_a, d_b, d_c, n);

    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++)
        std::cout << h_a[i] << " + " << h_b[i] << " = " << h_c[i] << "\n";

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;

    return 0;
}

