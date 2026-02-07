#include <stdio.h>

const int TILE_SIZE = 32;
const int MATRIX_DIM = 4096;

__global__ void transpose_kernel(float *output, const float *input)
{
  __shared__ float tile[TILE_SIZE][TILE_SIZE];

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  tile[threadIdx.y][threadIdx.x] = input[y * MATRIX_DIM + x];

  __syncthreads();

  int xNew = blockIdx.y * blockDim.x + threadIdx.x;
  int yNew = blockIdx.x * blockDim.y + threadIdx.y;

  output[yNew * MATRIX_DIM + xNew] = tile[threadIdx.x][threadIdx.y];
}
