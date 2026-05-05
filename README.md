# TP1 CODESIGN — OpenCL Matrix Multiplication

HC & CODESIGN GL3 — INSAT

---

## Complete Context for codesign tp1 Compte Rendu

---

## Hardware Setup

| Device | Role | Platform |
|--------|------|----------|
| NVIDIA RTX 4060 Laptop / RTX 3050 | Dedicated GPU | NVIDIA CUDA (idx 0) |
| Intel UHD / Iris Xe (integrated) | iGPU | Intel OpenCL Graphics (idx 1) |
| Intel i7-13620H | CPU | Intel OpenCL (idx 2) |

---

## Part A — Kernel Optimization

**Baseline:** Prof's coalesced kernel `C_elem_ji.cl` — `j=get_global_id(0)`, `i=get_global_id(1)`. Called "coalesced" because adjacent work-items (differing in `j`) write to adjacent memory addresses `C[i*N+j]`.

---

## Optimization 1: Local Memory Tiling (`C_tiled.cl`) — 1.39x

**Problem with baseline:** Every work-item reads `A` and `B` directly from global memory (VRAM). Global memory latency is high (~400-800 cycles). For a tile shared by 32×32=1024 work-items, the same element of `A` is loaded 32 separate times.

**Technique:** Divide the `K` dimension (inner loop) into tiles of size `TS=32`. Each work-group cooperatively loads a `TS×TS` block of `A` and `B` into `__local` memory (fast, on-chip, ~4 cycles). All work-items in the group then read from local memory for the inner loop.

**Why it helps:** Each element loaded from global memory is reused `TS=32` times within the work-group. Reduces global memory traffic by factor `TS`. Local memory bandwidth is ~100× faster than global.

**Parameters:** `TS=32`, work-group `(32×32)=1024` work-items, each computes 1 element of `C`.

---

## Optimization 2: 1D Register Blocking (`C_reg_blocking.cl`) — 4.42x

**Problem with tiling:** Each work-item still only computes 1 output element. Local memory accesses, while fast, still have latency. We waste compute throughput.

**Technique:** Each work-item computes `WPT=8` elements of `C` in a vertical strip instead of 1. It maintains 8 accumulators (`acc[8]`) in registers — the fastest storage on the GPU (0 cycle latency). Work-group size shrinks to `32×4=128` work-items, global size becomes `(N, N/8)`.

**Why it helps:**
- Registers are faster than any memory including local
- 8× more output per work-item means 8× fewer work-items launched — less scheduling overhead
- The 8 accumulators hold partial sums across the entire `K` loop — no memory traffic for intermediate results

Jump from `1.39x` → `4.42x` is a `3.2×` jump over tiling because register ops are effectively free compared to even local memory reads.

---

## Optimization 3: 2D Register Blocking + Bank Conflict Padding (`C_optimized.cl`) — 6.75x

**Technique:** Each work-item computes `WPTM×WPTN = 8×8 = 64` elements of `C` (a 2D tile). Tile dimensions: `TSM=TSN=64` (output tile), `TSK=16` (K tile depth).

### Key design decisions:

#### A) Asub stored transposed `[TSM][TSK+PAD]`

The `A` sub-tile is loaded in transposed layout. This allows adjacent work-items to load adjacent `K` values from `A` (coalesced global load), while still allowing efficient row-access during the outer product computation.

#### B) Outer product accumulation:

```
for k in 0..TSK:
    load Areg[8] from Asub (8 values for my 8 M-rows, from local)
    load Breg[8] from Bsub (8 values for my 8 N-cols, from local)
    for wm in 0..8:
        for wn in 0..8:
            acc[wm][wn] += Areg[wm] * Breg[wn]  ← 64 FMAs, pure registers
```

Each outer product uses 8+8=16 local memory reads to do 64 multiply-accumulates. Arithmetic intensity = 4 FMAs per local memory read.

#### C) PAD=3 to avoid bank conflicts

Local memory is divided into 32 banks. If multiple work-items access the same bank simultaneously → serialized (bank conflict). With `Bsub[TSK][TSN]` where `TSN=64`, all work-items in a warp reading the same row hit the same bank repeatedly. Adding `PAD=3` makes the row stride 67 (not 64=multiple of 32) → accesses spread across different banks.

**Impact of PAD alone:** `3.37M` → `5.43M` MFLOPS (+61% from padding alone). This was discovered experimentally.

### TSK=16 vs 32:

- `TSK` controls how many K-steps per tile iteration = compute intensity per global load
- RTX 4060 has high bandwidth (192 GB/s) → `TSK=16` is already enough to hide latency
- RTX 3050 has lower bandwidth (112 GB/s) → `TSK=32` might help (more reuse per load)
- We tested: `TSK=16` wins on 4060 (`TSK=32` drops to 0.70x), 3050 may differ

---

## Part A Results (RTX 4060, N=2048, COUNT=20)

| Kernel | MFLOPS | Speedup |
|--------|--------|---------|
| Coalesced (baseline) | ~810,000 | 1.00x |
| Tiled TS=32 | ~1,125,000 | 1.39x |
| 1D Reg Blocking WPT=8 | ~3,581,000 | 4.42x |
| 2D Reg Blocking + PAD=3 | ~5,476,000 | 6.75x |

---

## Part B — Multi-Device Execution

**Setup:**
- **NVIDIA** runs `C_elem_ij.cl` (UNCOALESCED — `i=get_global_id(0)`, intentionally the worst kernel as per TP instructions)
- **Intel iGPU** runs `C_optimized.cl` (our best Part A kernel)
- `N=8192`, `COUNT=5`
- **Speedup** = Performance(2 devices) / Performance(NVIDIA alone UNCOALESCED)

**Row split strategy:** Both devices compute their assigned rows of `C` in parallel. Each device gets its rows of `A` and all of `B`. Total time = `max(T_nvidia, T_intel)`.

---

## Step 1: Theoretical Split

**Benchmark each GPU individually on full 8192×8192:**
- `P_nv` = NVIDIA uncoalesced solo GFLOPS
- `P_in` = Intel optimized solo GFLOPS

$$R_{\text{intel}} = N \times \frac{P_{\text{in}}}{P_{\text{nv}} + P_{\text{in}}}$$

**Intuition:** If Intel delivers 36% of total GFLOPS, it should get 36% of rows — both finish in equal time under ideal conditions.

**On RTX 4060:** `P_nv≈220`, `P_in≈122` → `R_intel=2944` (36%)

**On RTX 3050:** `P_nv≈127`, `P_in≈723` → `R_intel=6976` (85%)

The RTX 3050 result (85% to Intel) explains the 6.72x speedup: Intel iGPU (Iris Xe, ~723 GFLOPS) massively outperforms RTX 3050 uncoalesced (~127 GFLOPS). The speedup isn't just from parallelism — it's from putting 85% of the work on the much faster (and better-kerneled) device.

**Row alignment:** All splits snapped to multiples of 64 — satisfies both NVIDIA (local 16×16) and Intel (TSM=64).

---

## Step 2: Why the Theoretical Split Is a Good Starting Point

We verified empirically that solo throughput ≈ concurrent throughput (< 2% difference). This is because:
- **NVIDIA** reads/writes only its own GDDR6 VRAM during compute — no shared bus
- **Intel iGPU** shares system RAM with CPU but uses zero-copy buffers (no PCIe traffic)
- The two devices don't compete for memory during computation — they're fully independent

Therefore the theoretical formula (based on solo benchmarks) predicts the concurrent optimal split well. The adaptive search typically converges in 1-2 iterations from the theoretical starting point.

---

## Step 3: Adaptive Search — Weighted Barycenter

**Why we refine beyond theoretical:** Under contention (both running simultaneously), effective throughputs can shift slightly. Also, the optimal split is where `T_nvidia ≈ T_intel` (neither device idles).

### Algorithm:

1. Start at theoretical split
2. Run both GPUs in parallel, measure individual kernel times `T_nv` and `T_in` via OpenCL profiling events
3. Compute effective concurrent throughput:
   - `rate_nv = R_nv / T_nv` (rows/sec under contention)
   - `rate_in = R_in / T_in`
4. Compute equalizing time (weighted barycenter):
   $$\text{eq\_time} = \frac{T_{\text{in}} \times \text{rate}_{\text{in}} + T_{\text{nv}} \times \text{rate}_{\text{nv}}}{\text{rate}_{\text{in}} + \text{rate}_{\text{nv}}}$$
5. Compute rows to shift:
   ```
   if T_nv > T_in:  deltaT = T_nv - eq_time
                    added_rows = rate_in × deltaT  (give Intel more)
   else:            deltaT = T_in - eq_time
                    removed_rows = rate_nv × deltaT (give Intel fewer)
   ```
6. Stop when `|T_nv - T_in| < 50ms` (convergence threshold) or oscillation detected

### Why barycenter, not midpoint:

**Example:** Intel 4s, NVIDIA 6s, `rate_nv=1700` rows/s, `rate_in=900` rows/s

- **Midpoint target** = 5s each → give Intel `900 × 1 = 900` more rows → Intel becomes 5s but NVIDIA only drops to 5.47s (still bottleneck)
- **Barycenter:** `eq_time = (4×900 + 6×1700)/(900+1700) = 5.31s` → Intel needs `900 × 1.31 = 1179` rows → both reach ~5.31s simultaneously

The barycenter accounts for the asymmetry: Intel processes fewer rows/sec, so adding 1 second of Intel work requires more rows than removing 1 second of NVIDIA work.

The result is cached to `split_config.txt` (hardware-specific, gitignored). On subsequent runs, `multi_device.py` reads it directly — no calibration overhead.

---

## Part B Results

| Machine | NVIDIA alone | 2 GPUs | Speedup |
|---------|--------------|--------|---------|
| RTX 4060 + Intel UHD | ~220 GFLOPS | ~335 GFLOPS | ~1.52x |
| RTX 3050 + Intel Iris Xe | ~127 GFLOPS | ~843 GFLOPS | ~6.72x |

---

## Key memory architecture notes for the report:

- **NVIDIA:** data lives in GDDR6 VRAM (192 GB/s), PCIe transfer only at start/end
- **Intel iGPU:** zero-copy buffers — data stays in system RAM, iGPU reads directly via internal memory controller, no PCIe traffic at all
- **Enqueue Intel first** + `queue.flush()` ensures both GPUs start simultaneously (wall timer starts before first enqueue, so submission latency matters)