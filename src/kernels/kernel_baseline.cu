#include <stdio.h>

// --- Baseline Kernel

__global__ void baseline_transpose_kernel(float *output, const float *input)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < 4096 && y < 4096) {
    output[x * 4096 + y] = input[y * 4096 + x];
  }
}
