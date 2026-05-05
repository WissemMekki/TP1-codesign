// Lab 2 Part 3-1 — single-hidden-layer NN with the NAÏVE sum reduction.
// Uses the kernel exactly as given in Annex-1 of the lab handout.
//
// Pipeline:
//   CPU  : generate x[N] and W[N,N], compute xw = x .* W
//   D2H  : upload xw to device
//   GPU  : two-pass naïve reduction over each row + sigmoid → h[N]
//          two-pass naïve reduction over h[N]              → y
//   H2D  : download y, compare to CPU reference

#include "reductions.cuh"
#include <cstdio>
#include <vector>

int main() {
    std::printf("== Part 3-1 : NN with Naïve Sum-Reduction (Annex-1) ==\n");
    std::printf("N = %d\n\n", N);

    std::vector<float> x(N), W(N * N), xw(N * N);
    generate_inputs(x, W);
    cpu_step1(x, W, xw);

    std::vector<double> h_ref;
    double y_cpu = cpu_reference(xw, h_ref);

    float *d_xw, *d_partials, *d_h, *d_y_partials, *d_y;
    CUDA_CHECK(cudaMalloc(&d_xw,         size_t(N) * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_partials,   size_t(N) * BLOCKS_PER_ROW_NAIVE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_h,          size_t(N) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y_partials, size_t(BLOCKS_PER_ROW_NAIVE) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y,          sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_xw, xw.data(), size_t(N) * N * sizeof(float),
                          cudaMemcpyHostToDevice));

    // Warm-up + 5 timed runs, take median
    run_naive(d_xw, d_y, d_partials, d_h, d_y_partials);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> times(5);
    for (int i = 0; i < 5; ++i)
        times[i] = run_naive(d_xw, d_y, d_partials, d_h, d_y_partials);
    double med = median_of(times);

    float y_gpu;
    CUDA_CHECK(cudaMemcpy(&y_gpu, d_y, sizeof(float), cudaMemcpyDeviceToHost));

    std::printf("GPU time (median of 5)  : %.3f ms\n", med);
    std::printf("Performance             : %.2f GFLOPS\n", gflops_for(med));
    std::printf("y_gpu = %.6f   y_cpu = %.6f   %s\n",
                y_gpu, y_cpu, verify(y_gpu, y_cpu, 5e-2) ? "[PASS]" : "[FAIL]");

    cudaFree(d_xw);
    cudaFree(d_partials);
    cudaFree(d_h);
    cudaFree(d_y_partials);
    cudaFree(d_y);
    return 0;
}
