# Part B — Adaptive Split Search

## Problem

Splitting matrix multiplication across 2 GPUs (NVIDIA + Intel iGPU) requires choosing
how many rows each device computes. The optimal split depends on hardware — e.g. ~37%
Intel on RTX 4060 vs ~50% on RTX 3050 — and shifts under contention (shared memory bus).

## Approach

### Step 1: Theoretical split from solo benchmarks

Benchmark each GPU individually on the full N×N matrix:
- NVIDIA (uncoalesced kernel) → P_nv GFLOPS
- Intel iGPU (optimized kernel) → P_in GFLOPS

Theoretical split: `R_intel = N × P_in / (P_nv + P_in)`

This assumes linear scaling and no contention — a good starting point but not optimal.

### Step 2: Adaptive search under contention

Starting from the theoretical split, iteratively adjust using measured kernel times
from parallel execution. The goal: **equalize NVIDIA and Intel kernel times** so
neither device idles while the other finishes.

Each iteration:
1. Run both GPUs in parallel at current split, measure individual kernel times
   (T_nv, T_in) via OpenCL profiling events
2. Compute effective throughput under contention:
   - `rate_nv = R_nv / T_nv` (rows/sec for NVIDIA)
   - `rate_in = R_in / T_in` (rows/sec for Intel)
3. Estimate target split via weighted barycenter:
   - `target_R_in = N × rate_in / (rate_nv + rate_in)`
4. Jump to target (snapped to multiple of 64)
5. Stop when `|T_nv - T_in|` < convergence threshold

### Why barycenter, not midpoint

At split (nv=5120, in=3072), if NVIDIA takes 3.0s and Intel takes 2.7s:
- Simple midpoint between current splits = 4096 → overshoots
- Weighted barycenter = 8192 × (3072/2.7) / (5120/3.0 + 3072/2.7) = 3275

The midpoint ignores that Intel processes fewer rows/sec. The barycenter
accounts for each device's actual throughput, giving a much better estimate.

## Two implementations

Both are algebraically equivalent but express the jump differently.

### Method 1: Direct target (`calibrate.py`)

Computes the target split directly from effective throughputs:

```
rate_nv = R_nv / T_nv
rate_in = R_in / T_in
target_R_in = N × rate_in / (rate_nv + rate_in)
jump = target_R_in - R_in
```

Compact — one formula gives the new split.

### Method 2: Weighted average with explicit steps (`calibrate_weighted_average.py`)

Breaks the jump into intuitive steps:

```
1. Equalizing time (weighted average of T_nv and T_in):
   eq_time = (T_in × rate_in + T_nv × rate_nv) / (rate_in + rate_nv)

2. Time gap between the bottleneck GPU and the equalizing time:
   if T_nv > T_in:  deltaT = T_nv - eq_time   (NVIDIA is bottleneck)
   else:            deltaT = T_in - eq_time    (Intel is bottleneck)

3. Convert time gap to rows using the receiving device's throughput:
   if NVIDIA is bottleneck:  added_rows = rate_in × deltaT  (give Intel more)
   else:                     removed_rows = rate_nv × deltaT (give Intel fewer)
```

The reasoning: if NVIDIA takes 6s and Intel takes 4s, a naive target of 5s each
would overshoot — adding rows that cost Intel 1s doesn't free up 1s from NVIDIA
(NVIDIA processes faster per row). The weighted average (eq_time ≈ 5.31s) accounts
for this asymmetry: Intel needs to absorb more rows because each row costs it more
time than it saves NVIDIA.

### Convergence

Both methods converge in 1-3 iterations. Oscillation detection stops the search
if the same split is revisited, picking the best wall time seen. The result is
cached to `split_config.txt` so subsequent runs skip calibration entirely.

## Row alignment constraints

- NVIDIA (uncoalesced kernel, local_size 16×16): rows must be multiple of 16
- Intel (optimized kernel, TSM=64): rows must be multiple of 64
- All splits snapped to multiple of 64 (satisfies both)

