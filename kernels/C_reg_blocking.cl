// Optimization 2: Local Memory Tiling + 1D Register Blocking
// Each thread computes WPT elements of C using register arrays.
// Reuses loaded B values across WPT accumulators.

#define TS 32
#define WPT 8
#define RTS (TS / WPT)  // 4 - reduced tile size for row dimension

__kernel void mmul(
    const int N,
    __global float* A,
    __global float* B,
    __global float* C)
{
    const int j = get_global_id(0);       // column, 0..N-1
    const int i = get_global_id(1);       // row (reduced), 0..N/WPT-1
    const int lj = get_local_id(0);       // 0..TS-1 = 0..31
    const int li = get_local_id(1);       // 0..RTS-1 = 0..3

    // Local memory tiles
    __local float Asub[TS][TS];
    __local float Bsub[TS][TS];

    // Register array for accumulation
    float acc[WPT];
    for (int w = 0; w < WPT; w++)
        acc[w] = 0.0f;

    const int numTiles = N / TS;

    for (int t = 0; t < numTiles; t++) {
        // Cooperative load: each thread loads WPT elements of each tile
        #pragma unroll
        for (int w = 0; w < WPT; w++) {
            Asub[li * WPT + w][lj] = A[(i * WPT + w) * N + (t * TS + lj)];
            Bsub[li * WPT + w][lj] = B[(t * TS + li * WPT + w) * N + j];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Accumulate: reuse each Bsub value across WPT accumulators
        #pragma unroll
        for (int k = 0; k < TS; k++) {
            float Btmp = Bsub[k][lj];
            #pragma unroll
            for (int w = 0; w < WPT; w++)
                acc[w] += Asub[li * WPT + w][k] * Btmp;
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store WPT results
    #pragma unroll
    for (int w = 0; w < WPT; w++)
        C[(i * WPT + w) * N + j] = acc[w];
}
