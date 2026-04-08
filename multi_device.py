"""
Part B: Running kernel on multiple OpenCL devices
Split matrix multiplication (N=8192) across NVIDIA GPU + Intel integrated GPU.
"""
import pyopencl as cl
import numpy as np
from time import time

AVAL = 3.257
BVAL = 5.723
COUNT = 5
N = 8192

# ============================================================
# Setup: find both GPUs
# ============================================================
platforms = cl.get_platforms()
nvidia_dev = intel_dev = None
for p in platforms:
    for d in p.get_devices():
        name = d.name.lower()
        if 'nvidia' in name:
            nvidia_dev = d
        elif 'intel' in name and 'uhd' in name:
            intel_dev = d

assert nvidia_dev, "NVIDIA GPU not found"
assert intel_dev, "Intel integrated GPU not found"
print(f"NVIDIA device: {nvidia_dev.name}")
print(f"Intel device:  {intel_dev.name}")

# ============================================================
# Phase 1: Individual GPU benchmarks (TP question 1)
# ============================================================
print(f"\n{'='*65}")
print(f"  Phase 1: Individual GPU benchmarks (N={N}, COUNT={COUNT})")
print(f"{'='*65}")

def benchmark_device(device, kernel_file, N, global_size, local_size):
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)
    size = N * N
    h_A = np.empty(size, dtype=np.float32); h_A.fill(AVAL)
    h_B = np.empty(size, dtype=np.float32); h_B.fill(BVAL)
    h_C = np.empty(size, dtype=np.float32)
    d_a = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_A)
    d_b = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_B)
    d_c = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, size=h_C.nbytes)
    src = open(kernel_file).read()
    prg = cl.Program(ctx, src).build()
    mmul = prg.mmul
    mmul.set_scalar_arg_dtypes([np.int32, None, None, None])
    # Warmup
    mmul(queue, global_size, local_size, np.int32(N), d_a, d_b, d_c); queue.finish()
    # Timed
    t0 = time()
    for _ in range(COUNT):
        mmul(queue, global_size, local_size, np.int32(N), d_a, d_b, d_c); queue.finish()
    elapsed = time() - t0
    gflops = 2.0 * COUNT * N**3 / (1e9 * elapsed)
    cl.enqueue_copy(queue, h_C, d_c)
    max_err = float(np.max(np.abs(h_C - float(N) * AVAL * BVAL)))
    return gflops, elapsed, max_err

print("\nBenchmarking NVIDIA with UNCOALESCED kernel...")
gf_nvidia, _, err_nv = benchmark_device(nvidia_dev, 'C_elem_ij.cl', N, (N, N), (16, 16))

print("Benchmarking Intel iGPU with OPTIMIZED kernel (2D Reg Blocking)...")
gf_intel, _, err_in = benchmark_device(intel_dev, 'C_optimized.cl', N, (N//8, N//8), (8, 8))

print(f"\n{'GPU':<25} {'Method':<30} {'GFLOPS/sec':>12}")
print(f"{'-'*70}")
print(f"{'NVIDIA RTX 4060':<25} {'UNCOALESCED':<30} {gf_nvidia:>12.2f}")
print(f"{'Intel UHD Graphics':<25} {'2D Reg Blocking (best A)':<30} {gf_intel:>12.2f}")

# ============================================================
# Phase 2: Compute split ratio (TP question 2)
# ============================================================
print(f"\n{'='*65}")
print(f"  Phase 2: Split ratio computation")
print(f"{'='*65}")

R_nvidia_raw = N * gf_nvidia / (gf_nvidia + gf_intel)
R_nvidia = int(round(R_nvidia_raw / 64)) * 64
R_nvidia = max(64, min(N - 64, R_nvidia))
R_intel = N - R_nvidia

print(f"\nIndividual performance:")
print(f"  NVIDIA (UNCOALESCED):       {gf_nvidia:.2f} GFLOPS/sec")
print(f"  Intel (2D Reg Blocking):    {gf_intel:.2f} GFLOPS/sec")
print(f"  Ratio: NVIDIA {gf_nvidia/(gf_nvidia+gf_intel)*100:.1f}% / Intel {gf_intel/(gf_nvidia+gf_intel)*100:.1f}%")
print(f"\nSplit: NVIDIA gets {R_nvidia} rows ({R_nvidia*100//N}%), Intel gets {R_intel} rows ({R_intel*100//N}%)")
print(f"  NVIDIA: rows 0..{R_nvidia-1} using UNCOALESCED kernel")
print(f"  Intel:  rows {R_nvidia}..{N-1} using 2D Reg Blocking kernel")

# ============================================================
# Phase 3: Multi-device parallel execution (TP question 3)
# ============================================================
print(f"\n{'='*65}")
print(f"  Phase 3: Multi-device parallel execution")
print(f"{'='*65}")

ctx_nv = cl.Context([nvidia_dev])
ctx_in = cl.Context([intel_dev])
queue_nv = cl.CommandQueue(ctx_nv)
queue_in = cl.CommandQueue(ctx_in)

# Host arrays
h_A = np.empty(N * N, dtype=np.float32); h_A.fill(AVAL)
h_B = np.empty(N * N, dtype=np.float32); h_B.fill(BVAL)
h_C = np.empty(N * N, dtype=np.float32)

# Build kernels
prg_nv = cl.Program(ctx_nv, open('C_elem_ij.cl').read()).build()
mmul_nv = prg_nv.mmul; mmul_nv.set_scalar_arg_dtypes([np.int32, None, None, None])
prg_in = cl.Program(ctx_in, open('C_optimized.cl').read()).build()
mmul_in = prg_in.mmul; mmul_in.set_scalar_arg_dtypes([np.int32, None, None, None])

# --- NVIDIA alone: full matrix with UNCOALESCED ---
print(f"\nBenchmarking NVIDIA alone (full {N}x{N}, UNCOALESCED)...")
d_a_full = cl.Buffer(ctx_nv, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_A)
d_b_nv = cl.Buffer(ctx_nv, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_B)
d_c_full = cl.Buffer(ctx_nv, cl.mem_flags.WRITE_ONLY, size=h_C.nbytes)
mmul_nv(queue_nv, (N, N), (16, 16), np.int32(N), d_a_full, d_b_nv, d_c_full); queue_nv.finish()

nvidia_alone_times = []
for _ in range(COUNT):
    t0 = time()
    mmul_nv(queue_nv, (N, N), (16, 16), np.int32(N), d_a_full, d_b_nv, d_c_full)
    queue_nv.finish()
    nvidia_alone_times.append(time() - t0)
avg_nvidia_alone = np.mean(nvidia_alone_times)
gflops_nvidia_alone = 2.0 * N**3 / (1e9 * avg_nvidia_alone)

# --- Fix 3: Contention-aware split sweep ---
print(f"\nSweeping split ratios to find optimal under contention...")

# Helper to create Intel zero-copy buffer (Fix 1)
def make_intel_buffer(ctx, queue, flags, host_array):
    buf = cl.Buffer(ctx, flags | cl.mem_flags.ALLOC_HOST_PTR, size=host_array.nbytes)
    ptr, _ = cl.enqueue_map_buffer(queue, buf, cl.map_flags.WRITE_INVALIDATE_REGION,
                                    0, host_array.shape, host_array.dtype)
    ptr[:] = host_array
    del ptr
    return buf

# Intel B buffer (shared across all splits, zero-copy)
d_b_in = make_intel_buffer(ctx_in, queue_in, cl.mem_flags.READ_ONLY, h_B)

best_speedup = 0
best_R_in = 0
best_gflops = 0

for R_in_try in [256, 512, 768, 1024, 1536, 2048, 2560, 2944, 3072, 3584, 4096]:
    if R_in_try % 64 != 0 or R_in_try >= N:
        continue
    R_nv_try = N - R_in_try

    h_A_nv = h_A[:R_nv_try * N].copy()
    h_A_in = h_A[R_nv_try * N:].copy()
    h_C_nv = np.empty(R_nv_try * N, dtype=np.float32)
    h_C_in = np.empty(R_in_try * N, dtype=np.float32)

    # NVIDIA buffers (dedicated VRAM, COPY_HOST_PTR is fine)
    d_a_nv = cl.Buffer(ctx_nv, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_A_nv)
    d_c_nv = cl.Buffer(ctx_nv, cl.mem_flags.WRITE_ONLY, size=h_C_nv.nbytes)

    # Intel buffers (zero-copy via ALLOC_HOST_PTR — Fix 1)
    d_a_in = make_intel_buffer(ctx_in, queue_in, cl.mem_flags.READ_ONLY, h_A_in)
    d_c_in = cl.Buffer(ctx_in, cl.mem_flags.READ_WRITE | cl.mem_flags.ALLOC_HOST_PTR, size=h_C_in.nbytes)

    # Warmup
    mmul_nv(queue_nv, (R_nv_try, N), (16, 16), np.int32(N), d_a_nv, d_b_nv, d_c_nv); queue_nv.finish()
    mmul_in(queue_in, (N // 8, R_in_try // 8), (8, 8), np.int32(N), d_a_in, d_b_in, d_c_in); queue_in.finish()

    times = []
    for _ in range(3):
        t0 = time()
        # Fix 2: enqueue slower device first, flush immediately
        mmul_in(queue_in, (N // 8, R_in_try // 8), (8, 8), np.int32(N), d_a_in, d_b_in, d_c_in)
        queue_in.flush()
        mmul_nv(queue_nv, (R_nv_try, N), (16, 16), np.int32(N), d_a_nv, d_b_nv, d_c_nv)
        queue_nv.flush()
        queue_nv.finish()
        queue_in.finish()
        times.append(time() - t0)

    avg_t = np.mean(times)
    gf = 2.0 * N**3 / (1e9 * avg_t)
    sp = gf / gflops_nvidia_alone
    print(f"  NVIDIA:{R_nv_try} + Intel:{R_in_try}  -> {avg_t:.3f}s  {gf:.1f} GFLOPS  {sp:.2f}x")

    if sp > best_speedup:
        best_speedup = sp
        best_R_in = R_in_try
        best_gflops = gf

# --- Final run with best split ---
R_nvidia = N - best_R_in
R_intel = best_R_in
print(f"\nBest split: NVIDIA:{R_nvidia} + Intel:{R_intel} rows")

h_A_nv = h_A[:R_nvidia * N].copy()
h_A_in = h_A[R_nvidia * N:].copy()
h_C_nv = np.empty(R_nvidia * N, dtype=np.float32)
h_C_in = np.empty(R_intel * N, dtype=np.float32)

d_a_nv = cl.Buffer(ctx_nv, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_A_nv)
d_c_nv = cl.Buffer(ctx_nv, cl.mem_flags.WRITE_ONLY, size=h_C_nv.nbytes)
d_a_in = make_intel_buffer(ctx_in, queue_in, cl.mem_flags.READ_ONLY, h_A_in)
d_c_in = cl.Buffer(ctx_in, cl.mem_flags.READ_WRITE | cl.mem_flags.ALLOC_HOST_PTR, size=h_C_in.nbytes)

# Warmup
mmul_nv(queue_nv, (R_nvidia, N), (16, 16), np.int32(N), d_a_nv, d_b_nv, d_c_nv); queue_nv.finish()
mmul_in(queue_in, (N // 8, R_intel // 8), (8, 8), np.int32(N), d_a_in, d_b_in, d_c_in); queue_in.finish()

multi_times = []
for _ in range(COUNT):
    t0 = time()
    mmul_in(queue_in, (N // 8, R_intel // 8), (8, 8), np.int32(N), d_a_in, d_b_in, d_c_in)
    queue_in.flush()
    mmul_nv(queue_nv, (R_nvidia, N), (16, 16), np.int32(N), d_a_nv, d_b_nv, d_c_nv)
    queue_nv.flush()
    queue_nv.finish()
    queue_in.finish()
    multi_times.append(time() - t0)

avg_multi = np.mean(multi_times)
gflops_multi = 2.0 * N**3 / (1e9 * avg_multi)
speedup = gflops_multi / gflops_nvidia_alone

# Correctness
cl.enqueue_copy(queue_nv, h_C_nv, d_c_nv); queue_nv.finish()
cl.enqueue_copy(queue_in, h_C_in, d_c_in); queue_in.finish()
h_C[:R_nvidia * N] = h_C_nv
h_C[R_nvidia * N:] = h_C_in
expected = float(N) * AVAL * BVAL
max_err = float(np.max(np.abs(h_C - expected)))

# ============================================================
# Results
# ============================================================
print(f"\n{'='*65}")
print(f"  Results")
print(f"{'='*65}")
print(f"\n  NVIDIA alone (UNCOALESCED, {N}x{N}):")
print(f"    Avg time: {avg_nvidia_alone:.4f}s  ->  {gflops_nvidia_alone:.2f} GFLOPS/sec")
print(f"\n  2 GPUs parallel (NVIDIA:{R_nvidia} rows + Intel:{R_intel} rows):")
print(f"    Avg time: {avg_multi:.4f}s  ->  {gflops_multi:.2f} GFLOPS/sec")
print(f"\n  Speedup = {gflops_multi:.2f} / {gflops_nvidia_alone:.2f} = {speedup:.2f}x")
print(f"  Max error: {max_err:.2f}")
print()
