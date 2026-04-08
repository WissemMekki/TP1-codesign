// Optimization 3: 2D Register Blocking with bank conflict avoidance
// Each thread computes a WPTM x WPTN (8x8) sub-tile of C.
// Asub stored transposed [TSM][TSK+PAD] for coalesced global loads
// and bank conflict-free local reads.

#define TSM 64      // Tile size in M dimension
#define TSN 64      // Tile size in N dimension
#define TSK 16      // Tile size in K dimension
#define WPTM 8      // Work per thread in M
#define WPTN 8      // Work per thread in N
#define RTSM (TSM / WPTM)  // 8
#define RTSN (TSN / WPTN)  // 8
#define LPTA ((TSK * TSM) / (RTSM * RTSN))  // 16
#define LPTB ((TSK * TSN) / (RTSM * RTSN))  // 16
#define PAD 3       // Padding to avoid local memory bank conflicts

__kernel void mmul(
    const int N,
    __global float* A,
    __global float* B,
    __global float* C)
{
    const int tidn = get_local_id(0);   // 0..RTSN-1
    const int tidm = get_local_id(1);   // 0..RTSM-1
    const int offsetN = get_group_id(0) * TSN;
    const int offsetM = get_group_id(1) * TSM;

    // Padded local memory to avoid bank conflicts
    __local float Asub[TSM][TSK + PAD];
    __local float Bsub[TSK][TSN + PAD];

    // 64 accumulators per thread (8x8 sub-tile of C)
    float acc[WPTM][WPTN];
    #pragma unroll
    for (int wm = 0; wm < WPTM; wm++)
        #pragma unroll
        for (int wn = 0; wn < WPTN; wn++)
            acc[wm][wn] = 0.0f;

    // Register caches
    float Areg[WPTM];
    float Breg[WPTN];

    const int tid = tidm * RTSN + tidn;
    const int numTiles = N / TSK;

    for (int t = 0; t < numTiles; t++) {
        // Load A: Asub[m_local][k_local] = A[row * N + k]
        // Adjacent threads → adjacent k_local → coalesced global reads
        #pragma unroll
        for (int la = 0; la < LPTA; la++) {
            int idx = la * RTSM * RTSN + tid;
            int m_local = idx / TSK;
            int k_local = idx % TSK;
            Asub[m_local][k_local] = A[(offsetM + m_local) * N + (t * TSK + k_local)];
        }

        // Load B: Bsub[k_local][n_local] = B[k * N + col]
        // Adjacent threads → adjacent n_local → coalesced global reads
        #pragma unroll
        for (int lb = 0; lb < LPTB; lb++) {
            int idx = lb * RTSM * RTSN + tid;
            int k_local = idx / TSN;
            int n_local = idx % TSN;
            Bsub[k_local][n_local] = B[(t * TSK + k_local) * N + (offsetN + n_local)];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Outer product accumulation with register caching
        #pragma unroll
        for (int k = 0; k < TSK; k++) {
            #pragma unroll
            for (int wm = 0; wm < WPTM; wm++)
                Areg[wm] = Asub[tidm * WPTM + wm][k];
            #pragma unroll
            for (int wn = 0; wn < WPTN; wn++)
                Breg[wn] = Bsub[k][tidn * WPTN + wn];

            #pragma unroll
            for (int wm = 0; wm < WPTM; wm++)
                #pragma unroll
                for (int wn = 0; wn < WPTN; wn++)
                    acc[wm][wn] += Areg[wm] * Breg[wn];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store results
    #pragma unroll
    for (int wm = 0; wm < WPTM; wm++)
        #pragma unroll
        for (int wn = 0; wn < WPTN; wn++)
            C[(offsetM + tidm * WPTM + wm) * N + (offsetN + tidn * WPTN + wn)] = acc[wm][wn];
}
