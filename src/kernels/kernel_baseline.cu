#include <math.h>

// --- Baseline Kernel

__global__ void baseline_transpose_kernel(float *output, const float *input)
{
  int threadIdX = blockIdx.x * blockDim.x + threadIdx.x;
  int threadIdY = blockIdx.y * blockDim.y + threadIdx.y;

  int threadIdYMatrix = threadIdY * 4;

  output[threadIdYMatrix*4096+threadIdX] = input[threadIdX*4096+threadIdYMatrix];
  output[(threadIdYMatrix+1)*4096+threadIdX] = input[threadIdX*4096+threadIdYMatrix+1];
  output[(threadIdYMatrix+2)*4096+threadIdX] = input[threadIdX*4096+threadIdYMatrix+2];
  output[(threadIdYMatrix+3)*4096+threadIdX] = input[threadIdX*4096+threadIdYMatrix+3];
}
