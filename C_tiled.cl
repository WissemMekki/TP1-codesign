// Optimization 1: Local Memory Tiling
// Each workgroup loads TS x TS tiles of A and B into local memory,
// reducing global memory traffic by a factor of TS.

#define TS 32

__kernel void mmul(
    const int N,
    __global float* A,
    __global float* B,
    __global float* C)
{
    // Column and row indices
    const int j = get_global_id(0);
    const int i = get_global_id(1);
    const int lj = get_local_id(0);
    const int li = get_local_id(1);

    // Local memory tiles
    __local float Asub[TS][TS];
    __local float Bsub[TS][TS];

    float acc = 0.0f;
    const int numTiles = N / TS;

    for (int t = 0; t < numTiles; t++) {
        // Cooperatively load tiles into local memory
        Asub[li][lj] = A[i * N + (t * TS + lj)];
        Bsub[li][lj] = B[(t * TS + li) * N + j];

        barrier(CLK_LOCAL_MEM_FENCE);

        // Accumulate dot product from local tiles
        #pragma unroll
        for (int k = 0; k < TS; k++)
            acc += Asub[li][k] * Bsub[k][lj];

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    C[i * N + j] = acc;
}
