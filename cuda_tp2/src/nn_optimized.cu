// Lab 2 Part 3-2 — single-hidden-layer NN with the BEST sum reduction.
// Uses k_shfl_row_sigmoid + k_shfl_final from reductions.cuh:
//   - 1 block per row (no second-pass partials)
//   - Each thread cascades 32 global loads into a register
//   - Block-level reduction = warp reductions via __shfl_down_sync
//   - Sigmoid applied in the same kernel before writing h[row]

#include "reductions.cuh"
#include <cstdio>
#include <vector>

int main() {
    std::printf("== Part 3-2 : NN with Warp-Shuffle Sum-Reduction (best) ==\n");
    std::printf("N = %d\n\n", N);

    std::vector<float> x(N), W(N * N), xw(N * N);
    generate_inputs(x, W);
    cpu_step1(x, W, xw);

    std::vector<double> h_ref;
    double y_cpu = cpu_reference(xw, h_ref);

    float *d_xw, *d_h, *d_y;
    CUDA_CHECK(cudaMalloc(&d_xw, size_t(N) * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_h,  size_t(N) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y,  sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_xw, xw.data(), size_t(N) * N * sizeof(float),
                          cudaMemcpyHostToDevice));

    // run_shfl ignores the partials/y_partials arguments
    run_shfl(d_xw, d_y, nullptr, d_h, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> times(5);
    for (int i = 0; i < 5; ++i)
        times[i] = run_shfl(d_xw, d_y, nullptr, d_h, nullptr);
    double med = median_of(times);

    float y_gpu;
    CUDA_CHECK(cudaMemcpy(&y_gpu, d_y, sizeof(float), cudaMemcpyDeviceToHost));

    std::printf("GPU time (median of 5)  : %.3f ms\n", med);
    std::printf("Performance             : %.2f GFLOPS\n", gflops_for(med));
    std::printf("y_gpu = %.6f   y_cpu = %.6f   %s\n",
                y_gpu, y_cpu, verify(y_gpu, y_cpu, 5e-2) ? "[PASS]" : "[FAIL]");

    cudaFree(d_xw);
    cudaFree(d_h);
    cudaFree(d_y);
    return 0;
}
