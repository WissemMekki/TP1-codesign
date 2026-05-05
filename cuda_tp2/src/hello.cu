#include <cstdio>
#include <cuda_runtime.h>

__global__ void hello() {
    printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

int main() {
    int dev_count = 0;
    cudaGetDeviceCount(&dev_count);
    printf("CUDA devices found: %d\n", dev_count);

    for (int i = 0; i < dev_count; ++i) {
        cudaDeviceProp p{};
        cudaGetDeviceProperties(&p, i);
        printf("  [%d] %s  (sm_%d%d, %d SMs, %.1f GB)\n",
               i, p.name, p.major, p.minor, p.multiProcessorCount,
               p.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    }

    hello<<<2, 4>>>();
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("OK\n");
    return 0;
}
