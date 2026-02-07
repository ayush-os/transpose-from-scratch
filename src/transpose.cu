#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

const int TILE_SIZE = 32;
const int MATRIX_DIM = 4096;

__global__ void transpose_kernel(float *output, const float *input)
{
  __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  tile[threadIdx.y][threadIdx.x] = input[y * MATRIX_DIM + x];

  __syncthreads();

  int xNew = blockIdx.y * blockDim.x + threadIdx.x;
  int yNew = blockIdx.x * blockDim.y + threadIdx.y;

  output[yNew * MATRIX_DIM + xNew] = tile[threadIdx.x][threadIdx.y];
}

void checkCudaError(cudaError_t err, const char *msg) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err)
              << std::endl;
    exit(EXIT_FAILURE);
  }
}

int main() {
  const int MATRIX_DIM = 4096; // 4096 * 4096 = 16,777,216 elements
  const int N_TOTAL = MATRIX_DIM * MATRIX_DIM; // Total elements: 16,777,216
  const size_t bytes = N_TOTAL * sizeof(float);

  const int WARMUP_RUNS = 10;
  const int TIMING_RUNS = 1000;

  std::cout << "--- Matrix Transpose Stable Timing Test (N x N) ---"
            << std::endl;
  std::cout << "Matrix Dimension N: " << MATRIX_DIM << " x " << MATRIX_DIM
            << std::endl;
  std::cout << "Total Array Size: " << N_TOTAL << " elements ("
            << (double)bytes / (1024 * 1024 * 1024) << " GB)" << std::endl;

  std::vector<float> h_input(N_TOTAL);
  std::vector<float> h_output(N_TOTAL);

  for (int i = 0; i < N_TOTAL; ++i) {
    h_input[i] = (float)i;
  }

  float *d_input, *d_output;
  checkCudaError(cudaMalloc(&d_input, bytes), "d_input allocation");
  checkCudaError(cudaMalloc(&d_output, bytes), "d_output allocation");

  checkCudaError(
      cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice),
      "input copy H->D");

  dim3 threadsPerBlock(32, 32, 1);
  dim3 numBlocks(MATRIX_DIM / 32, MATRIX_DIM / 32);

  std::cout << "Grid: " << numBlocks.x << "x" << numBlocks.y << " blocks, "
            << threadsPerBlock.x << "x" << threadsPerBlock.y
            << " threads/block." << std::endl;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  std::cout << "Warming up the GPU and Caches (" << WARMUP_RUNS << " runs)..."
            << std::endl;
  for (int i = 0; i < WARMUP_RUNS; ++i) {
    transpose_kernel<<<numBlocks, threadsPerBlock>>>(d_output, d_input);
  }
  cudaDeviceSynchronize();
  checkCudaError(cudaGetLastError(), "warm-up kernel launch");

  // 2. STABLE TIMING LOOP
  float total_milliseconds = 0;
  std::cout << "Starting Stable Timing Loop (" << TIMING_RUNS << " runs)..."
            << std::endl;

  for (int i = 0; i < TIMING_RUNS; ++i) {
    cudaEventRecord(start);

    // Kernel launch
    transpose_kernel<<<numBlocks, threadsPerBlock>>>(d_output, d_input);

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float milliseconds_i = 0;
    cudaEventElapsedTime(&milliseconds_i, start, stop);
    total_milliseconds += milliseconds_i;
  }

  float average_milliseconds = total_milliseconds / TIMING_RUNS;

  std::cout << "\n--- Timing Results ---" << std::endl;
  std::cout << "Total execution time for " << TIMING_RUNS
            << " stable runs: " << total_milliseconds << " ms" << std::endl;
  std::cout << "**Average kernel execution time:** "
            << average_milliseconds * 1000.0f << " us" << std::endl;

  checkCudaError(
      cudaMemcpy(h_output.data(), d_output, bytes, cudaMemcpyDeviceToHost),
      "output copy D->H");

  const int TEST_ROW_A = 1234;
  const int TEST_COL_A = 4032;

  const int index_A = TEST_ROW_A * MATRIX_DIM + TEST_COL_A;

  const int index_B = TEST_COL_A * MATRIX_DIM + TEST_ROW_A;

  const float expected_value = h_input[index_A];

  if (std::abs(h_output[index_B] - expected_value) < 1e-5) {
    std::cout << "\nVerification Check: **PASSED**" << std::endl;
    std::cout << "  A[" << TEST_ROW_A << "][" << TEST_COL_A << "] ("
              << expected_value << ")"
              << " correctly moved to B[" << TEST_COL_A << "][" << TEST_ROW_A
              << "] (" << h_output[index_B] << ")" << std::endl;
  } else {
    std::cout << "\nVerification Check: **FAILED**" << std::endl;
    std::cout << "  Expected B[" << TEST_COL_A << "][" << TEST_ROW_A
              << "] to be " << expected_value << ", but got "
              << h_output[index_B] << std::endl;
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_input);
  cudaFree(d_output);

  return 0;
}