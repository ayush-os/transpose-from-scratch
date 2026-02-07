#include <math.h>

// --- Baseline Kernel

__global__ void baseline_transpose_kernel(float *output, const float *input)
{
  int threadIdX = blockIdx.x * blockDim.x + threadIdx.x;
  int threadIdY = blockIdx.y * blockDim.y + threadIdx.y;

  int threadIdYMatrix = threadIdY * 4;

  output[threadIdYMatrix][threadIdX] = input[threadIdX][threadIdYMatrix];
  output[threadIdYMatrix+1][threadIdX] = input[threadIdX][threadIdYMatrix+1];
  output[threadIdYMatrix+2][threadIdX] = input[threadIdX][threadIdYMatrix+2];
  output[threadIdYMatrix+3][threadIdX] = input[threadIdX][threadIdYMatrix+3];
}
