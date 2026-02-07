# transpose-from-scratch

Phase 1: base kernel
- each thread handles 1 element in a 4096x4096 matrix
- grid size is (MATRIX_DIM / 32, MATRIX_DIM / 32) and block size is (32, 32)
- no shared memory
- Average kernel execution time: 508.852 us
- time to implement tiling - reads are already coalesced, its the writes that need fixing
- 

Phase 2: tiling/smem
- switching to 32x32 smem array which all threads in the block load into normally
- then i reverse the blockIdx.x/y and the tile[threadIdx.x][threadIdx.y] to do the transpose
- this eliminates the writes uncoalescing, meaning now both reads and writes are coalesced.
- 196.634 us (2.59x speedup from phase 1)

Phase 3: 