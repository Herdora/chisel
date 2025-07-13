/*
Matrix Multiplication with mfma_f32_16x16x4f32

Problem: Multiply M×K and K×N matrices using MFMA intrinsic
Goal: Understand MFMA API and data flow
*/

// hipcc simple-mm.hip -o simple-mm && ./simple-mm
#include <algorithm>
#include <chrono>
#include <cmath>
#include <hip/hip_runtime.h>
#include <iomanip>
#include <iostream> //

using f4 = float __attribute__((ext_vector_type(4)));

// MFMA intrinsic parameters
constexpr int MFMA_CBSZ = 0; // CBSZ parameter for MFMA
constexpr int MFMA_ABID = 0; // ABID parameter for MFMA
constexpr int MFMA_BLGP = 0; // BLGP parameter for MFMA

// Matrix dimensions
constexpr int MATRIX_ROWS = 256;
constexpr int MATRIX_INNER_DIM = 64;
constexpr int MATRIX_COLS = 256;

__global__ void mfma_matrix_multiply(const float *__restrict__ A,
                                     const float *__restrict__ B,
                                     float *__restrict__ C, int M, int N,
                                     int K) {
  /* ---- thread ↔︎ matrix coordinates -------------------------------- */
  const unsigned lane_row = threadIdx.x;      // 0‥15 → output row
  const unsigned lane_k = threadIdx.y;        // 0‥3  → 4-wide K slice
  const unsigned wave_row0 = blockIdx.y * 16; // tile origin in C
  const unsigned wave_col0 = blockIdx.x * 16;

  f4 acc = {0.f, 0.f, 0.f, 0.f}; // four results / lane

  /* ---- iterate over K, 4 scalars at a time ------------------------- */
  for (int kBase = 0; kBase < K; kBase += 4) {
    /* bounds for the two scalars this lane wants to read */
    const bool a_ok = (wave_row0 + lane_row) < M && (kBase + lane_k) < K;
    const bool b_ok = (wave_col0 + lane_row) < N && (kBase + lane_k) < K;

    const float a_scalar =
        a_ok ? A[(wave_row0 + lane_row) * K + (kBase + lane_k)] : 0.f;

    const float b_scalar =
        b_ok ? B[(kBase + lane_k) * N + (wave_col0 + lane_row)] : 0.f;

    acc = __builtin_amdgcn_mfma_f32_16x16x4f32(a_scalar, b_scalar, acc,
                                               MFMA_CBSZ, MFMA_ABID, MFMA_BLGP);
  }

  /* ---- store four scalars that this lane owns ---------------------- */
  const int col_base = wave_col0 + lane_k * 4; // 0/4/8/12 in tile
  const int global_row = wave_row0 + lane_row;

  if (global_row < M) // row guard
  {
    float *cPtr = &C[global_row * N + col_base];
    if (col_base + 0 < N)
      cPtr[0] = acc.x;
    if (col_base + 1 < N)
      cPtr[1] = acc.y;
    if (col_base + 2 < N)
      cPtr[2] = acc.z;
    if (col_base + 3 < N)
      cPtr[3] = acc.w;
  }
}

__global__ void
matmul_naive(const float *__restrict__ A, const float *__restrict__ B,
             float *__restrict__ C, int M, int N,
             int K) // M=row-count of A/C, N=col-count of B/C, K=shared dim
{
  // Which element of C does this thread own?
  const int col = blockIdx.x * blockDim.x + threadIdx.x; // j
  const int row = blockIdx.y * blockDim.y + threadIdx.y; // i

  if (row >= M || col >= N) // guard-band: don't step off the edge
    return;

  float sum = 0; // dot(A[i, :], B[:, j])
  for (int k = 0; k < K; ++k) {
    sum += A[row * K + k] * // A[i, k]
           B[k * N + col];  // B[k, j]
  }

  C[row * N + col] = sum; // C[i, j]
}

void cpu_matrix_multiply(const float *matrix_a, const float *matrix_b,
                         float *result_matrix, int num_rows, int num_cols,
                         int inner_dim) {
  for (int row_idx = 0; row_idx < num_rows; ++row_idx) {
    for (int col_idx = 0; col_idx < num_cols; ++col_idx) {
      float dot_product = 0.0f;
      for (int k_idx = 0; k_idx < inner_dim; ++k_idx)
        dot_product += matrix_a[row_idx * inner_dim + k_idx] *
                       matrix_b[k_idx * num_cols + col_idx];
      result_matrix[row_idx * num_cols + col_idx] = dot_product;
    }
  }
}

void initialize_test_matrices(float *host_matrix_a, float *host_matrix_b,
                              int num_rows, int num_cols, int inner_dim) {
  for (int i = 0; i < num_rows * inner_dim; ++i) {
    host_matrix_a[i] = (float)(i + 1);
  }
  for (int i = 0; i < inner_dim * num_cols; ++i) {
    host_matrix_b[i] = (float)(inner_dim * num_cols - i);
  }
}

void allocate_and_copy_gpu_matrices(float *host_matrix_a, float *host_matrix_b,
                                    float **device_matrix_a,
                                    float **device_matrix_b, int num_rows,
                                    int num_cols, int inner_dim) {
  hipMalloc(device_matrix_a, num_rows * inner_dim * sizeof(float));
  hipMalloc(device_matrix_b, inner_dim * num_cols * sizeof(float));
  hipMemcpy(*device_matrix_a, host_matrix_a,
            num_rows * inner_dim * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(*device_matrix_b, host_matrix_b,
            inner_dim * num_cols * sizeof(float), hipMemcpyHostToDevice);
}

// Measure kernel execution time
float benchmark_kernel_execution(void (*kernel_function)(const float *,
                                                         const float *, float *,
                                                         int, int, int),
                                 const float *device_matrix_a,
                                 const float *device_matrix_b,
                                 float *device_result, dim3 grid_dim,
                                 dim3 block_dim, const char *kernel_name) {
  hipEvent_t start_event, stop_event;
  hipEventCreate(&start_event);
  hipEventCreate(&stop_event);

  hipEventRecord(start_event);
  hipLaunchKernelGGL(kernel_function, grid_dim, block_dim, 0, 0,
                     device_matrix_a, device_matrix_b, device_result,
                     MATRIX_ROWS, MATRIX_COLS, MATRIX_INNER_DIM);
  hipEventRecord(stop_event);
  hipEventSynchronize(stop_event);

  float execution_time_ms;
  hipEventElapsedTime(&execution_time_ms, start_event, stop_event);

  hipEventDestroy(start_event);
  hipEventDestroy(stop_event);

  return execution_time_ms * 1000.0f; // Convert to microseconds
}

void print_benchmark_results(float mfma_execution_time,
                             float naive_execution_time,
                             float cpu_execution_time) {
  std::cout << std::fixed << std::setprecision(2);
  std::cout << "Matrix dimensions: " << MATRIX_ROWS << "×" << MATRIX_INNER_DIM
            << " × " << MATRIX_INNER_DIM << "×" << MATRIX_COLS << " = "
            << MATRIX_ROWS << "×" << MATRIX_COLS << "\n";
  std::cout << "Timing Results:\n";
  std::cout << "GPU MFMA kernel execution time: " << mfma_execution_time
            << " microseconds\n";
  std::cout << "GPU Naive kernel execution time: " << naive_execution_time
            << " microseconds\n";
  std::cout << "CPU reference execution time: " << cpu_execution_time
            << " microseconds\n";
  std::cout << "MFMA Speedup vs CPU: "
            << cpu_execution_time / mfma_execution_time << "x\n";
  std::cout << "Naive Speedup vs CPU: "
            << cpu_execution_time / naive_execution_time << "x\n\n";
}

void verify_against_cpu(const float *gpu_result,
                        const float *cpu_reference_result, int num_rows,
                        int num_cols, const char *kernel_name) {
  const float abs_tolerance = 1e-6f;
  const float rel_tolerance = 1e-4f;

  for (int i = 0; i < num_rows * num_cols; ++i) {
    const float gpu = gpu_result[i];
    const float cpu = cpu_reference_result[i];
    const float abs_diff = std::abs(gpu - cpu);
    const float tolerance = rel_tolerance * std::abs(cpu) + abs_tolerance;

    if (abs_diff > tolerance) {
      std::cout << kernel_name << " verification against CPU: ✗ Incorrect"
                << std::endl;
      std::cout << "  Mismatch at index " << i << ":" << std::endl;
      std::cout << "    GPU      : " << gpu << std::endl;
      std::cout << "    CPU      : " << cpu << std::endl;
      std::cout << "    Abs Diff : " << abs_diff << std::endl;
      std::cout << "    Tolerance: " << tolerance << std::endl;
      return;
    }
  }
  std::cout << kernel_name << " verification against CPU: ✓ Correct"
            << std::endl;
}

inline dim3 make_mfma_grid(int M, int N) {
  return dim3((N + 15) / 16,  // cols  → blocks-x
              (M + 15) / 16); // rows  → blocks-y
}

int main() {
  constexpr int CPU_BENCHMARK_ITERATIONS = 100;

  float host_matrix_a[MATRIX_ROWS * MATRIX_INNER_DIM],
      host_matrix_b[MATRIX_INNER_DIM * MATRIX_COLS];
  float mfma_result[MATRIX_ROWS * MATRIX_COLS],
      naive_result[MATRIX_ROWS * MATRIX_COLS],
      cpu_reference_result[MATRIX_ROWS * MATRIX_COLS];

  initialize_test_matrices(host_matrix_a, host_matrix_b, MATRIX_ROWS,
                           MATRIX_COLS, MATRIX_INNER_DIM);

  float *device_matrix_a, *device_matrix_b, *device_mfma_result,
      *device_naive_result;
  allocate_and_copy_gpu_matrices(host_matrix_a, host_matrix_b, &device_matrix_a,
                                 &device_matrix_b, MATRIX_ROWS, MATRIX_COLS,
                                 MATRIX_INNER_DIM);
  hipMalloc(&device_mfma_result, MATRIX_ROWS * MATRIX_COLS * sizeof(float));
  hipMalloc(&device_naive_result, MATRIX_ROWS * MATRIX_COLS * sizeof(float));

  auto cpu_start_time = std::chrono::high_resolution_clock::now();
  for (int iteration = 0; iteration < CPU_BENCHMARK_ITERATIONS; ++iteration) {
    cpu_matrix_multiply(host_matrix_a, host_matrix_b, cpu_reference_result,
                        MATRIX_ROWS, MATRIX_COLS, MATRIX_INNER_DIM);
  }
  auto cpu_stop_time = std::chrono::high_resolution_clock::now();
  auto cpu_total_duration =
      std::chrono::duration_cast<std::chrono::microseconds>(cpu_stop_time -
                                                            cpu_start_time);
  float cpu_execution_time =
      cpu_total_duration.count() / CPU_BENCHMARK_ITERATIONS;

  const dim3 mfma_block_dim(16, 4);
  const dim3 mfma_grid_dim = make_mfma_grid(MATRIX_ROWS, MATRIX_COLS);
  float mfma_execution_time = benchmark_kernel_execution(
      mfma_matrix_multiply, device_matrix_a, device_matrix_b,
      device_mfma_result, mfma_grid_dim, mfma_block_dim, "MFMA");
  hipMemcpy(mfma_result, device_mfma_result,
            MATRIX_ROWS * MATRIX_COLS * sizeof(float), hipMemcpyDeviceToHost);
  verify_against_cpu(mfma_result, cpu_reference_result, MATRIX_ROWS,
                     MATRIX_COLS, "MFMA");

  const dim3 naive_block_dim(16, 16);
  const dim3 naive_grid_dim(
      (MATRIX_COLS + naive_block_dim.x - 1) / naive_block_dim.x,
      (MATRIX_ROWS + naive_block_dim.y - 1) / naive_block_dim.y);
  float naive_execution_time = benchmark_kernel_execution(
      matmul_naive, device_matrix_a, device_matrix_b, device_naive_result,
      naive_grid_dim, naive_block_dim, "Naive");
  hipMemcpy(naive_result, device_naive_result,
            MATRIX_ROWS * MATRIX_COLS * sizeof(float), hipMemcpyDeviceToHost);
  verify_against_cpu(naive_result, cpu_reference_result, MATRIX_ROWS,
                     MATRIX_COLS, "Naive");

  print_benchmark_results(mfma_execution_time, naive_execution_time,
                          cpu_execution_time);

  hipFree(device_matrix_a);
  hipFree(device_matrix_b);
  hipFree(device_mfma_result);
  hipFree(device_naive_result);
  return 0;
}

