#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <chrono>
#include <cuda_runtime.h>

// ----------------------------------------------------------------------------
// Problem dimensions (Lab 2 Part 3)
// ----------------------------------------------------------------------------
constexpr int N = 8192;          // length of x, # hidden neurons, # cols of W

// ----------------------------------------------------------------------------
// Error checking
// ----------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            std::fprintf(stderr, "CUDA error %s:%d : %s\n",                    \
                         __FILE__, __LINE__, cudaGetErrorString(_e));          \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

// ----------------------------------------------------------------------------
// Host-side data generation
//   x[N]     : input vector, values in [-1, 1]
//   W[N*N]   : weight matrix (row-major), values in [-1/N, 1/N] so that the
//              row sums stay O(1) and sigmoid output is meaningful (not saturated).
//   xw[N*N]  : Step-1 element-wise product, computed on CPU (xw[i,j] = x[j]*W[i,j])
// ----------------------------------------------------------------------------
inline void generate_inputs(std::vector<float>& x, std::vector<float>& W) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dx(-1.0f, 1.0f);
    std::uniform_real_distribution<float> dw(-1.0f / N, 1.0f / N);

    for (int i = 0; i < N; ++i) x[i] = dx(rng);
    for (int i = 0; i < N * N; ++i) W[i] = dw(rng);
}

inline void cpu_step1(const std::vector<float>& x,
                      const std::vector<float>& W,
                      std::vector<float>& xw) {
    // xw[i, j] = x[j] * W[i, j]
    for (int i = 0; i < N; ++i) {
        const float* Wrow = &W[i * N];
        float* xwrow      = &xw[i * N];
        for (int j = 0; j < N; ++j) {
            xwrow[j] = x[j] * Wrow[j];
        }
    }
}

// ----------------------------------------------------------------------------
// CPU reference for the GPU step (sum reduction + sigmoid + final sum)
//   Done in double precision to be the ground truth that float kernels are
//   compared against.
// ----------------------------------------------------------------------------
inline double cpu_reference(const std::vector<float>& xw, std::vector<double>& h_out) {
    h_out.resize(N);
    for (int i = 0; i < N; ++i) {
        double s = 0.0;
        const float* row = &xw[i * N];
        for (int j = 0; j < N; ++j) s += row[j];
        h_out[i] = 1.0 / (1.0 + std::exp(-s));
    }
    double y = 0.0;
    for (int i = 0; i < N; ++i) y += h_out[i];
    return y;
}

// ----------------------------------------------------------------------------
// GPU timing helper — measures elapsed time of a callable in ms via cudaEvent.
//   Usage:  float ms = time_gpu([&]{ kernel<<<g,b>>>(...); });
// ----------------------------------------------------------------------------
template <typename F>
inline float time_gpu(F&& fn) {
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

inline double median_of(std::vector<float>& v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return (n % 2) ? v[n/2] : 0.5 * (v[n/2 - 1] + v[n/2]);
}

// Effective compute throughput: roughly 1 add per element of xw plus 1 sigmoid
// per row plus N adds for the final reduction. For the report we use the
// elementwise-add count (N*N) as the GFLOPS reference — this is the
// memory-bound part that the reduction kernel is responsible for.
inline double gflops_for(double ms) {
    constexpr double FLOPS = double(N) * double(N);   // ~67M adds
    return (FLOPS / 1e9) / (ms / 1e3);
}

inline bool verify(double y_gpu, double y_cpu, double rel_tol = 1e-2) {
    double err = std::fabs(y_gpu - y_cpu) / std::fabs(y_cpu);
    if (err > rel_tol) {
        std::printf("  VERIFY FAIL : y_gpu=%.6f y_cpu=%.6f rel_err=%.3e (> %.0e)\n",
                    y_gpu, y_cpu, err, rel_tol);
        return false;
    }
    return true;
}
