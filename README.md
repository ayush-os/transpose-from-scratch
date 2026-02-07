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
- need to identify the shared memory bank conflict using ncu: 
  transpose_kernel(float *, const float *) (128, 128, 1)x(32, 32, 1), Context 1, Stream 7, Device 0, CC 8.0
    Section: Command line profiler metrics
    -------------------------------------------------------- ----------- ------------
    Metric Name                                              Metric Unit Metric Value
    -------------------------------------------------------- ----------- ------------
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum               16,281,843
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum                   32,301
    -------------------------------------------------------- ----------- ------------
    - clear that its happening pretty much entirely on loads (stores are negligible).

Phase 3: shared memory bank conflict fix via padding
now ncu results:
  transpose_kernel(float *, const float *) (128, 128, 1)x(32, 32, 1), Context 1, Stream 7, Device 0, CC 8.0
    Section: Command line profiler metrics
    -------------------------------------------------------- ----------- ------------
    Metric Name                                              Metric Unit Metric Value
    -------------------------------------------------------- ----------- ------------
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                   15,723
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum                   29,115
    -------------------------------------------------------- ----------- ------------
    ***1,035x reduction in bank conflicts***

now kernel exeuction time - 121.1 us
- 1.62x improvement from phase 2