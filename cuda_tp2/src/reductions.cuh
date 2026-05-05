#pragma once

// =============================================================================
// Sum-reduction kernel variants for Lab 2 Part 3.
//
// Each variant solves the same problem:
//   in : float xw[N*N] (row-major) on device
//   out: float y on device, y = sum_i sigmoid(sum_j xw[i,j])
//
// There are 5 variants, progressively faster:
//
//   0. naive            — Annex-1 of the lab handout, two-pass + sigmoid
//                         (block=256, blocks_per_row=32, warp-divergent reduction)
//   1. no_divergence    — same shape; replace `tid % (2s) == 0` with strided
//                         index so active threads stay contiguous within the warp
//   2. seq_addressing   — reverse the loop (s = block/2 → 1) with `if (tid < s)`
//                         to get sequential-addressing pattern (no bank conflicts)
//   3. first_add        — each thread loads two elements and sums them into
//                         shared memory; halves the number of blocks per row
//   4. warp_shuffle     — single pass per row: each thread cascades many global
//                         loads into a register, then block-level seq reduction,
//                         then warp-level reduction via __shfl_down_sync.
//
// Buffers required by each launcher (allocated once by the caller):
//   d_xw         : N*N floats (input)
//   d_partials   : N*MAX_BLOCKS_PER_ROW floats (scratch, big variants need <=32)
//   d_h          : N floats (hidden activations)
//   d_y_partials : MAX_BLOCKS_PER_ROW floats (scratch for the final sum)
//   d_y          : 1 float (output)
//
// Each `run_*` launcher returns the elapsed GPU time in ms.
// =============================================================================

#include "common.cuh"

constexpr int BLOCK = 256;
constexpr int BLOCKS_PER_ROW_NAIVE = (N + BLOCK - 1) / BLOCK;       // 32
constexpr int BLOCKS_PER_ROW_HALVED = (N + 2 * BLOCK - 1) / (2 * BLOCK); // 16

// =============================================================================
// 0. NAIVE  (matches Annex-1 verbatim)
// =============================================================================
__global__ void k_naive_block(const float* in, float* partials, int n) {
    __shared__ float sdata[BLOCK];
    int row = blockIdx.y;
    int bx  = blockIdx.x;
    int tid = threadIdx.x;
    int idx = bx * blockDim.x + tid;

    sdata[tid] = (idx < n) ? in[row * n + idx] : 0.0f;
    __syncthreads();

    // Naïve interleaved addressing — Annex-1 of the lab handout
    for (int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) partials[row * gridDim.x + bx] = sdata[0];
}

// Second-pass kernels: same naïve reduction shape on the partials, plus sigmoid.
__global__ void k_naive_finalize_row_sigmoid(const float* partials, float* h, int bpr) {
    __shared__ float sdata[BLOCK];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    sdata[tid] = (tid < bpr) ? partials[row * bpr + tid] : 0.0f;
    __syncthreads();
    for (int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) h[row] = 1.0f / (1.0f + __expf(-sdata[0]));
}

__global__ void k_naive_finalize_y(const float* part, float* y, int bpr) {
    __shared__ float sdata[BLOCK];
    int tid = threadIdx.x;
    sdata[tid] = (tid < bpr) ? part[tid] : 0.0f;
    __syncthreads();
    for (int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) y[0] = sdata[0];
}

inline float run_naive(const float* d_xw, float* d_y, float* d_partials,
                       float* d_h, float* d_y_partials) {
    constexpr int bpr = BLOCKS_PER_ROW_NAIVE;
    return time_gpu([&] {
        dim3 grid1(bpr, N);
        k_naive_block<<<grid1, BLOCK>>>(d_xw, d_partials, N);
        k_naive_finalize_row_sigmoid<<<N, BLOCK>>>(d_partials, d_h, bpr);
        dim3 grid3(bpr, 1);
        k_naive_block<<<grid3, BLOCK>>>(d_h, d_y_partials, N);
        k_naive_finalize_y<<<1, BLOCK>>>(d_y_partials, d_y, bpr);
    });
}

// =============================================================================
// 1. NO_DIVERGENCE — strided index, active threads contiguous in warp
// =============================================================================
__global__ void k_nodiv_block(const float* in, float* partials, int n) {
    __shared__ float sdata[BLOCK];
    int row = blockIdx.y;
    int bx  = blockIdx.x;
    int tid = threadIdx.x;
    int idx = bx * blockDim.x + tid;
    sdata[tid] = (idx < n) ? in[row * n + idx] : 0.0f;
    __syncthreads();
    for (int s = 1; s < blockDim.x; s *= 2) {
        int i = 2 * s * tid;            // active threads cluster at start of warp
        if (i < blockDim.x) sdata[i] += sdata[i + s];
        __syncthreads();
    }
    if (tid == 0) partials[row * gridDim.x + bx] = sdata[0];
}

__global__ void k_nodiv_finalize_row_sigmoid(const float* partials, float* h, int bpr) {
    __shared__ float sdata[BLOCK];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    sdata[tid] = (tid < bpr) ? partials[row * bpr + tid] : 0.0f;
    __syncthreads();
    for (int s = 1; s < blockDim.x; s *= 2) {
        int i = 2 * s * tid;
        if (i < blockDim.x) sdata[i] += sdata[i + s];
        __syncthreads();
    }
    if (tid == 0) h[row] = 1.0f / (1.0f + __expf(-sdata[0]));
}

__global__ void k_nodiv_finalize_y(const float* part, float* y, int bpr) {
    __shared__ float sdata[BLOCK];
    int tid = threadIdx.x;
    sdata[tid] = (tid < bpr) ? part[tid] : 0.0f;
    __syncthreads();
    for (int s = 1; s < blockDim.x; s *= 2) {
        int i = 2 * s * tid;
        if (i < blockDim.x) sdata[i] += sdata[i + s];
        __syncthreads();
    }
    if (tid == 0) y[0] = sdata[0];
}

inline float run_nodiv(const float* d_xw, float* d_y, float* d_partials,
                       float* d_h, float* d_y_partials) {
    constexpr int bpr = BLOCKS_PER_ROW_NAIVE;
    return time_gpu([&] {
        dim3 grid1(bpr, N);
        k_nodiv_block<<<grid1, BLOCK>>>(d_xw, d_partials, N);
        k_nodiv_finalize_row_sigmoid<<<N, BLOCK>>>(d_partials, d_h, bpr);
        dim3 grid3(bpr, 1);
        k_nodiv_block<<<grid3, BLOCK>>>(d_h, d_y_partials, N);
        k_nodiv_finalize_y<<<1, BLOCK>>>(d_y_partials, d_y, bpr);
    });
}

// =============================================================================
// 2. SEQUENTIAL ADDRESSING — `if (tid < s)` pattern, no bank conflicts
// =============================================================================
__global__ void k_seq_block(const float* in, float* partials, int n) {
    __shared__ float sdata[BLOCK];
    int row = blockIdx.y;
    int bx  = blockIdx.x;
    int tid = threadIdx.x;
    int idx = bx * blockDim.x + tid;
    sdata[tid] = (idx < n) ? in[row * n + idx] : 0.0f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) partials[row * gridDim.x + bx] = sdata[0];
}

__global__ void k_seq_finalize_row_sigmoid(const float* partials, float* h, int bpr) {
    __shared__ float sdata[BLOCK];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    sdata[tid] = (tid < bpr) ? partials[row * bpr + tid] : 0.0f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) h[row] = 1.0f / (1.0f + __expf(-sdata[0]));
}

__global__ void k_seq_finalize_y(const float* part, float* y, int bpr) {
    __shared__ float sdata[BLOCK];
    int tid = threadIdx.x;
    sdata[tid] = (tid < bpr) ? part[tid] : 0.0f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) y[0] = sdata[0];
}

inline float run_seq(const float* d_xw, float* d_y, float* d_partials,
                     float* d_h, float* d_y_partials) {
    constexpr int bpr = BLOCKS_PER_ROW_NAIVE;
    return time_gpu([&] {
        dim3 grid1(bpr, N);
        k_seq_block<<<grid1, BLOCK>>>(d_xw, d_partials, N);
        k_seq_finalize_row_sigmoid<<<N, BLOCK>>>(d_partials, d_h, bpr);
        dim3 grid3(bpr, 1);
        k_seq_block<<<grid3, BLOCK>>>(d_h, d_y_partials, N);
        k_seq_finalize_y<<<1, BLOCK>>>(d_y_partials, d_y, bpr);
    });
}

// =============================================================================
// 3. FIRST_ADD_DURING_LOAD — each thread loads 2 elements, halving block count
// =============================================================================
__global__ void k_firstadd_block(const float* in, float* partials, int n) {
    __shared__ float sdata[BLOCK];
    int row = blockIdx.y;
    int bx  = blockIdx.x;
    int tid = threadIdx.x;
    int chunk = 2 * blockDim.x;          // each block now covers 2*BLOCK elements
    int base  = bx * chunk;
    int row_off = row * n;
    float v0 = (base + tid              < n) ? in[row_off + base + tid]              : 0.0f;
    float v1 = (base + tid + blockDim.x < n) ? in[row_off + base + tid + blockDim.x] : 0.0f;
    sdata[tid] = v0 + v1;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) partials[row * gridDim.x + bx] = sdata[0];
}

inline float run_firstadd(const float* d_xw, float* d_y, float* d_partials,
                          float* d_h, float* d_y_partials) {
    constexpr int bpr = BLOCKS_PER_ROW_HALVED;        // 16 instead of 32
    return time_gpu([&] {
        dim3 grid1(bpr, N);
        k_firstadd_block<<<grid1, BLOCK>>>(d_xw, d_partials, N);
        k_seq_finalize_row_sigmoid<<<N, BLOCK>>>(d_partials, d_h, bpr);
        dim3 grid3(bpr, 1);
        k_firstadd_block<<<grid3, BLOCK>>>(d_h, d_y_partials, N);
        k_seq_finalize_y<<<1, BLOCK>>>(d_y_partials, d_y, bpr);
    });
}

// =============================================================================
// 4. WARP_SHUFFLE  — single-pass per row, register cascade + warp shuffle
//    This is the "best" kernel: one block reduces a whole row in one launch.
// =============================================================================
__device__ __forceinline__ float warp_reduce_sum(float v) {
    // Sum all 32 lanes of a warp into lane 0 (every lane gets the partial sum;
    // only lane 0 needs the final value).
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffff, v, offset);
    return v;
}

__device__ __forceinline__ float block_reduce_sum(float v) {
    // Reduce one float per thread to one float at threadIdx.x == 0.
    static __shared__ float warp_sums[BLOCK / 32];   // one slot per warp
    int lane    = threadIdx.x & 31;                   // 0..31 within warp
    int warp_id = threadIdx.x >> 5;                   // 0..(BLOCK/32 - 1)

    v = warp_reduce_sum(v);                           // each warp → its lane 0
    if (lane == 0) warp_sums[warp_id] = v;
    __syncthreads();

    // First warp loads the per-warp partials and reduces them
    v = (threadIdx.x < BLOCK / 32) ? warp_sums[lane] : 0.0f;
    if (warp_id == 0) v = warp_reduce_sum(v);
    return v;
}

// One block per row. Each thread cascades n/BLOCK loads into a register,
// then a single block reduction; lane 0 writes h[row] = sigmoid(sum).
__global__ void k_shfl_row_sigmoid(const float* in, float* h, int n) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* row_ptr = in + row * n;

    float acc = 0.0f;
    // Grid-stride within the row: each thread sums n/BLOCK elements (=32 for n=8192)
    for (int j = tid; j < n; j += blockDim.x) {
        acc += row_ptr[j];
    }
    float s = block_reduce_sum(acc);
    if (tid == 0) h[row] = 1.0f / (1.0f + __expf(-s));
}

// One block reduces n elements to a single output (no sigmoid).
__global__ void k_shfl_final(const float* in, float* y, int n) {
    int tid = threadIdx.x;
    float acc = 0.0f;
    for (int j = tid; j < n; j += blockDim.x) acc += in[j];
    float s = block_reduce_sum(acc);
    if (tid == 0) y[0] = s;
}

inline float run_shfl(const float* d_xw, float* d_y, float* /*d_partials*/,
                      float* d_h, float* /*d_y_partials*/) {
    return time_gpu([&] {
        // Pass 1: N blocks, each reduces a row of N elements → h[row] (sigmoided)
        k_shfl_row_sigmoid<<<N, BLOCK>>>(d_xw, d_h, N);
        // Pass 2: 1 block reduces h[N] → y
        k_shfl_final<<<1, BLOCK>>>(d_h, d_y, N);
    });
}
