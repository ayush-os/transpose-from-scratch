# transpose-from-scratch

Phase 1: base kernel
- each thread handles 1 element in a 4096x4096 matrix
- grid size is (MATRIX_DIM / 32, MATRIX_DIM / 32) and block size is (32, 32)
- no shared memory
- Average kernel execution time: 508.852 us
- time to implement tiling - reads are already coalesced, its the writes that need fixing
- 

Phase 2: tiling/smem