#include <stdio.h>

// --- Baseline Kernel

const int TILE_SIZE = 32;

__global__ void baseline_transpose_kernel(float *output, const float *input)
{
  __shared__ float tile[TILE_SIZE][TILE_SIZE];

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  tile[threadIdx.y][threadIdx.x] = input[y * 4096 + x];

  __syncthreads();

  int xNew = blockIdx.y * blockDim.x + threadIdx.x;
  int yNew = blockIdx.x * blockDim.y + threadIdx.x;

  output[yNew * 4096 + xNew] = tile[threadIdx.y][threadIdx.x];

  // if (x < 4096 && y < 4096) {
  //   output[x * 4096 + y] = input[y * 4096 + x];
  // }
}
