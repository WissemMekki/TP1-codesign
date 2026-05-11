// TP2 CUDA Part 2 — cuBLAS & TensorCore matrix multiplication
// Benchmarks three implementations for N=8192:
//   0. Naive MatmulXrow  (prof's Part 1 kernel, FP32, no TensorCore)
//   1. cuBLAS Sgemm      (NVIDIA optimised library, FP32, uses TensorCore internally)
//   2. WMMA TensorCore   (our WMMA kernel, FP16 inputs / FP32 accumulation)

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <mma.h>

using namespace nvcuda;

// ============================================================================
// Problem size
// ============================================================================
constexpr int MAT_N = 8192;

// ============================================================================
// Error helpers
// ============================================================================
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            std::fprintf(stderr, "CUDA error %s:%d : %s\n",                    \
                         __FILE__, __LINE__, cudaGetErrorString(_e));           \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

#define CUBLAS_CHECK(call)                                                      \
    do {                                                                        \
        cublasStatus_t _e = (call);                                             \
        if (_e != CUBLAS_STATUS_SUCCESS) {                                      \
            std::fprintf(stderr, "cuBLAS error %s:%d : status=%d\n",           \
                         __FILE__, __LINE__, (int)_e);                          \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

// ============================================================================
// GPU timing
// ============================================================================
template <typename F>
float time_gpu(F&& fn) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    fn();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms;
}

static double median_of(std::vector<float>& v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return (n & 1) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// GFLOPS for an N×N matrix multiply: 2*N^3 FLOPs
static double gflops(double ms, int n) {
    return (2.0 * n * n * n / 1e9) / (ms / 1e3);
}

// ============================================================================
// Spot-check a few elements of d_C against a CPU reference
// Only checks a 32×32 top-left corner to keep CPU time manageable.
// ============================================================================
static bool verify(const std::vector<float>& A, const std::vector<float>& B,
                   const float* d_C, int n, double rel_tol = 0.01) {
    std::vector<float> h_C((size_t)n * n);
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, (size_t)n * n * sizeof(float),
                          cudaMemcpyDeviceToHost));
    const int CHECK = 32;
    for (int i = 0; i < CHECK; ++i) {
        for (int j = 0; j < CHECK; ++j) {
            double ref = 0.0;
            for (int k = 0; k < n; ++k) ref += A[i * n + k] * B[k * n + j];
            double err = std::fabs(h_C[i * n + j] - ref) / (std::fabs(ref) + 1e-6);
            if (err > rel_tol) {
                std::printf("  MISMATCH [%d,%d]: gpu=%.4f cpu=%.4f rel=%.2e\n",
                            i, j, h_C[i * n + j], (float)ref, err);
                return false;
            }
        }
    }
    return true;
}

// ============================================================================
// 0. Naive MatmulXrow  (prof's kernel.cu, unmodified)
// ============================================================================
__global__ void matrixMulXrow(const float* a, const float* b, float* c, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    float tmp = 0.f;
    for (int k = 0; k < n; ++k)
        tmp += a[row * n + k] * b[k * n + col];
    c[row * n + col] = tmp;
}

// ============================================================================
// 2. TensorCore WMMA kernel
//
// Architecture: RTX 4060 Laptop (Ada Lovelace, sm_89) — 4th gen TensorCores.
// Fragment shape: m16n16k16, FP16 inputs, FP32 accumulation.
//
// Strategy: one warp (32 threads) computes one 16×16 output tile of C.
//   Grid  = (N/16, N/16)
//   Block = (32, 1)   — exactly one warp
//
// Inner loop iterates over the K dimension in WMMA_K=16 steps.
// B is stored row-major; we use wmma::row_major for both A and B fragments
// so that load_matrix_sync reads the correct tile from row-major memory.
// ============================================================================
constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;

__global__ void float_to_half(const float* __restrict__ in,
                               half*  __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = __float2half(in[idx]);
}

__global__ void matmul_wmma(const half* __restrict__ A,
                             const half* __restrict__ B,
                             float*      __restrict__ C, int n) {
    // Which 16×16 output tile does this warp own?
    int tileRow = blockIdx.x;   // 0 .. N/16-1
    int tileCol = blockIdx.y;   // 0 .. N/16-1

    wmma::fragment<wmma::matrix_a,    WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b,    WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>                 c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    // Accumulate over K in 16-wide strips
    for (int k = 0; k < n; k += WMMA_K) {
        // Pointer to the top-left corner of each 16×16 tile
        const half* pA = A + tileRow * WMMA_M * n + k;          // row-major stride = n
        const half* pB = B + k * n             + tileCol * WMMA_N; // row-major stride = n

        wmma::load_matrix_sync(a_frag, pA, n);
        wmma::load_matrix_sync(b_frag, pB, n);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // Write the 16×16 result tile (FP32) back to global memory
    float* pC = C + tileRow * WMMA_M * n + tileCol * WMMA_N;
    wmma::store_matrix_sync(pC, c_frag, n, wmma::mem_row_major);
}

// ============================================================================
// main
// ============================================================================
int main() {
    const int n = MAT_N;
    std::printf("== TP2 CUDA Part 2 — cuBLAS & TensorCore (N=%d) ==\n\n", n);
    std::printf("Device: RTX 4060 Laptop (sm_89, Ada Lovelace, 4th-gen TensorCores)\n");
    std::printf("Peak FP32 compute: ~15 TFLOPS  |  Peak FP16 TC: ~130 TFLOPS\n\n");

    // ---- Host data --------------------------------------------------------
    srand(42);
    std::vector<float> h_A((size_t)n * n), h_B((size_t)n * n);
    for (auto& v : h_A) v = (float)rand() / RAND_MAX;
    for (auto& v : h_B) v = (float)rand() / RAND_MAX;

    // ---- Device FP32 buffers ----------------------------------------------
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, (size_t)n * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, (size_t)n * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, (size_t)n * n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), (size_t)n*n*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), (size_t)n*n*sizeof(float), cudaMemcpyHostToDevice));

    // ---- Device FP16 buffers (for TensorCore) ----------------------------
    half *d_A_h, *d_B_h;
    CUDA_CHECK(cudaMalloc(&d_A_h, (size_t)n * n * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B_h, (size_t)n * n * sizeof(half)));
    {
        int threads = 256, blocks = ((size_t)n * n + threads - 1) / threads;
        float_to_half<<<blocks, threads>>>(d_A, d_A_h, n * n);
        float_to_half<<<blocks, threads>>>(d_B, d_B_h, n * n);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    constexpr int RUNS = 5;
    double naive_ms = 1.0;

    std::printf("%-32s %12s %12s %10s   verify\n",
                "Kernel", "median(ms)", "GFLOPS", "speedup");
    std::printf("%-32s %12s %12s %10s   ------\n",
                "------", "----------", "------", "-------");

    // ==========================================================================
    // 0. Naive MatmulXrow
    // ==========================================================================
    {
        const int THREADS = 32;
        dim3 block(THREADS, THREADS);
        dim3 grid(n / THREADS, n / THREADS);

        // warmup
        matrixMulXrow<<<grid, block>>>(d_A, d_B, d_C, n);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> times(RUNS);
        for (int i = 0; i < RUNS; ++i)
            times[i] = time_gpu([&]{ matrixMulXrow<<<grid, block>>>(d_A, d_B, d_C, n); });

        double med = median_of(times);
        naive_ms   = med;
        bool ok    = verify(h_A, h_B, d_C, n, 0.001);
        std::printf("%-32s %12.2f %12.2f %9.2fx   %s\n",
                    "0. Naive MatmulXrow (FP32)",
                    med, gflops(med, n), 1.0, ok ? "PASS" : "FAIL");
    }

    // ==========================================================================
    // 1. cuBLAS Sgemm
    //
    // cuBLAS is column-major. To compute C = A*B with row-major A, B, C:
    //   Observe: C_row = A_row * B_row
    //   In column-major view: C_col^T = B_col^T * A_col^T
    //   => cublasSgemm(OP_N, OP_N, N, N, N, 1, d_B, N, d_A, N, 0, d_C, N)
    //      interprets d_A as A^T_col and d_B as B^T_col, giving C^T_col = C_row.
    // ==========================================================================
    {
        cublasHandle_t handle;
        CUBLAS_CHECK(cublasCreate(&handle));

        const float alpha = 1.f, beta = 0.f;

        // warmup
        CUBLAS_CHECK(cublasSgemm(handle,
                                  CUBLAS_OP_N, CUBLAS_OP_N,
                                  n, n, n,
                                  &alpha,
                                  d_B, n,    // B (swapped: becomes B^T in col-major)
                                  d_A, n,    // A (swapped: becomes A^T in col-major)
                                  &beta,
                                  d_C, n));
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> times(RUNS);
        for (int i = 0; i < RUNS; ++i)
            times[i] = time_gpu([&]{
                CUBLAS_CHECK(cublasSgemm(handle,
                                          CUBLAS_OP_N, CUBLAS_OP_N,
                                          n, n, n,
                                          &alpha, d_B, n, d_A, n,
                                          &beta,  d_C, n));
            });

        double med = median_of(times);
        bool ok    = verify(h_A, h_B, d_C, n, 0.001);
        std::printf("%-32s %12.2f %12.2f %9.2fx   %s\n",
                    "1. cuBLAS Sgemm (FP32)",
                    med, gflops(med, n), naive_ms / med, ok ? "PASS" : "FAIL");

        CUBLAS_CHECK(cublasDestroy(handle));
    }

    // ==========================================================================
    // 2. TensorCore WMMA (FP16 in, FP32 accumulate)
    //
    // Each warp computes one 16×16 tile of C.
    // Grid  = (N/16, N/16) = (512, 512) = 262 144 blocks
    // Block = (32)          = 1 warp per block
    //
    // Note: this naive WMMA implementation has no shared-memory tiling, so
    // it loads each A and B tile from global memory multiple times.
    // cuBLAS adds shared-memory blocking on top of TensorCores and is faster.
    // The goal here is to demonstrate the WMMA API and TensorCore principle.
    // ==========================================================================
    {
        dim3 grid(n / WMMA_M, n / WMMA_N);
        dim3 block(32);   // 1 warp

        // warmup
        matmul_wmma<<<grid, block>>>(d_A_h, d_B_h, d_C, n);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> times(RUNS);
        for (int i = 0; i < RUNS; ++i)
            times[i] = time_gpu([&]{
                matmul_wmma<<<grid, block>>>(d_A_h, d_B_h, d_C, n);
            });

        double med = median_of(times);
        // FP16 inputs: tolerate up to 2% relative error vs FP32 reference
        bool ok    = verify(h_A, h_B, d_C, n, 0.02);
        std::printf("%-32s %12.2f %12.2f %9.2fx   %s\n",
                    "2. WMMA TensorCore (FP16)",
                    med, gflops(med, n), naive_ms / med, ok ? "PASS" : "FAIL~");
    }

    std::printf("\n");
    std::printf("Notes:\n");
    std::printf("  cuBLAS uses FP32 throughout and employs TensorCores internally\n");
    std::printf("  (via TF32 on Ada Lovelace) plus shared-memory blocking — hence\n");
    std::printf("  faster than our naive WMMA despite both using TensorCores.\n");
    std::printf("  WMMA demonstrates the TensorCore API; its speedup over MatmulXrow\n");
    std::printf("  comes from 256 FP16 FMAs per TensorCore clock vs scalar FP32.\n");

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_A_h));
    CUDA_CHECK(cudaFree(d_B_h));
    return 0;
}
