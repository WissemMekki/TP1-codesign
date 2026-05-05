// Benchmark all 5 reduction kernel variants against the same d_xw input.
// For each: warm-up + 5 timed runs, median, GFLOPS, speedup, verification.
// Output: stdout table + ../results.md

#include "reductions.cuh"
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <vector>

struct Variant {
    const char* name;
    float (*fn)(const float*, float*, float*, float*, float*);
};

int main() {
    std::printf("== Lab 2 Part 3 — Sum Reduction Benchmark ==\n");
    std::printf("N = %d, %d hidden neurons, %d total adds\n\n", N, N, N * N);

    // -------------------- Host-side prep --------------------
    std::vector<float> x(N);
    std::vector<float> W(N * N);
    std::vector<float> xw(N * N);
    std::printf("[host] generating x and W ...\n");
    generate_inputs(x, W);

    std::printf("[host] step 1: element-wise xw = x .* W ... ");
    auto t0 = std::chrono::high_resolution_clock::now();
    cpu_step1(x, W, xw);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_step1_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("%.1f ms\n", cpu_step1_ms);

    std::printf("[host] computing CPU reference y ... ");
    std::vector<double> h_ref;
    t0 = std::chrono::high_resolution_clock::now();
    double y_cpu = cpu_reference(xw, h_ref);
    t1 = std::chrono::high_resolution_clock::now();
    double cpu_ref_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("y_cpu=%.6f (%.1f ms)\n\n", y_cpu, cpu_ref_ms);

    // -------------------- Device buffers --------------------
    float *d_xw = nullptr, *d_partials = nullptr, *d_h = nullptr,
          *d_y_partials = nullptr, *d_y = nullptr;
    size_t bytes_xw       = size_t(N) * N * sizeof(float);
    size_t bytes_partials = size_t(N) * BLOCKS_PER_ROW_NAIVE * sizeof(float);
    size_t bytes_h        = size_t(N) * sizeof(float);
    size_t bytes_y_part   = size_t(BLOCKS_PER_ROW_NAIVE) * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_xw,         bytes_xw));
    CUDA_CHECK(cudaMalloc(&d_partials,   bytes_partials));
    CUDA_CHECK(cudaMalloc(&d_h,          bytes_h));
    CUDA_CHECK(cudaMalloc(&d_y_partials, bytes_y_part));
    CUDA_CHECK(cudaMalloc(&d_y,          sizeof(float)));

    std::printf("[device] uploading xw (%.1f MB) ...\n", bytes_xw / 1e6);
    CUDA_CHECK(cudaMemcpy(d_xw, xw.data(), bytes_xw, cudaMemcpyHostToDevice));

    // -------------------- Variants --------------------
    Variant variants[] = {
        { "0. naive (Annex-1)        ", run_naive    },
        { "1. no-divergence          ", run_nodiv    },
        { "2. seq-addressing         ", run_seq      },
        { "3. first-add-during-load  ", run_firstadd },
        { "4. warp-shuffle (cascade) ", run_shfl     },
    };

    constexpr int RUNS = 5;
    std::printf("\n%-30s %12s %12s %10s   verify\n",
                "Kernel", "median (ms)", "GFLOPS", "speedup");
    std::printf("------------------------------------------------------------------------------\n");

    double naive_ms = 0.0;
    std::vector<std::pair<std::string, double>> results;

    for (const auto& v : variants) {
        // Warm-up
        v.fn(d_xw, d_y, d_partials, d_h, d_y_partials);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> times(RUNS);
        for (int i = 0; i < RUNS; ++i) {
            times[i] = v.fn(d_xw, d_y, d_partials, d_h, d_y_partials);
        }
        double med = median_of(times);

        // Verify against CPU reference
        float y_gpu = 0.f;
        CUDA_CHECK(cudaMemcpy(&y_gpu, d_y, sizeof(float), cudaMemcpyDeviceToHost));
        bool ok = verify(y_gpu, y_cpu, 5e-2);

        if (naive_ms == 0.0) naive_ms = med;
        double speedup = naive_ms / med;

        std::printf("%-30s %12.3f %12.2f %9.2fx   %s\n",
                    v.name, med, gflops_for(med), speedup, ok ? "PASS" : "FAIL");
        results.emplace_back(v.name, med);
    }

    // -------------------- Write results.md --------------------
    {
        std::ofstream md("results.md");
        md << "# TP2 CUDA Part 3 — Sum Reduction Results\n\n";
        md << "Hardware: NVIDIA GeForce RTX 4060 Laptop GPU (sm_89, 24 SMs).\n";
        md << "Input: N=" << N << ", element-wise CPU step then GPU sum-reduction"
           << " of " << N << " rows of " << N << " floats, sigmoid, then final sum.\n\n";
        md << "| Kernel | Time (ms) | GFLOPS | Speedup vs Naïve |\n";
        md << "|---|---:|---:|---:|\n";
        for (const auto& r : results) {
            md << "| " << r.first << " | "
               << r.second << " | "
               << gflops_for(r.second) << " | "
               << (results[0].second / r.second) << "x |\n";
        }
        md.close();
        std::printf("\n[host] wrote results.md\n");
    }

    CUDA_CHECK(cudaFree(d_xw));
    CUDA_CHECK(cudaFree(d_partials));
    CUDA_CHECK(cudaFree(d_h));
    CUDA_CHECK(cudaFree(d_y_partials));
    CUDA_CHECK(cudaFree(d_y));
    return 0;
}
