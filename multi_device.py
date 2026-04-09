"""
Part B: Running kernel on multiple OpenCL devices.
Uses the calibrated split from calibrate.py to run matrix multiplication
across NVIDIA GPU (uncoalesced) + Intel iGPU (optimized) in parallel.
"""
import pyopencl as cl
import numpy as np
from time import time
from calibrate import find_devices, get_split, make_zero_copy_buffer

AVAL = 3.257
BVAL = 5.723
COUNT = 5
N = 8192

# Devices
nvidia_dev, intel_dev = find_devices()
print(f"NVIDIA device: {nvidia_dev.name}")
print(f"Intel device:  {intel_dev.name}")

# Get optimal split (cached or freshly calibrated)
R_intel = get_split()
R_nvidia = N - R_intel

# Contexts and queues
ctx_nv = cl.Context([nvidia_dev])
ctx_in = cl.Context([intel_dev])
queue_nv = cl.CommandQueue(ctx_nv)
queue_in = cl.CommandQueue(ctx_in)

# Host arrays
h_A = np.empty(N * N, dtype=np.float32); h_A.fill(AVAL)
h_B = np.empty(N * N, dtype=np.float32); h_B.fill(BVAL)
h_C = np.empty(N * N, dtype=np.float32)

# Build kernels
prg_nv = cl.Program(ctx_nv, open('kernels/C_elem_ij.cl').read()).build()
mmul_nv = prg_nv.mmul
mmul_nv.set_scalar_arg_dtypes([np.int32, None, None, None])

prg_in = cl.Program(ctx_in, open('kernels/C_optimized.cl').read()).build()
mmul_in = prg_in.mmul
mmul_in.set_scalar_arg_dtypes([np.int32, None, None, None])

# ============================================================
# Benchmark: NVIDIA alone (full matrix, uncoalesced)
# ============================================================
print(f"\nBenchmarking NVIDIA alone (full {N}x{N}, UNCOALESCED)...")
d_a_full = cl.Buffer(ctx_nv, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_A)
d_b_nv = cl.Buffer(ctx_nv, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_B)
d_c_full = cl.Buffer(ctx_nv, cl.mem_flags.WRITE_ONLY, size=h_C.nbytes)

# Warmup
mmul_nv(queue_nv, (N, N), (16, 16), np.int32(N), d_a_full, d_b_nv, d_c_full)
queue_nv.finish()

nvidia_alone_times = []
for _ in range(COUNT):
    t0 = time()
    mmul_nv(queue_nv, (N, N), (16, 16), np.int32(N), d_a_full, d_b_nv, d_c_full)
    queue_nv.finish()
    nvidia_alone_times.append(time() - t0)
avg_nvidia_alone = np.mean(nvidia_alone_times)
gflops_nvidia_alone = 2.0 * N**3 / (1e9 * avg_nvidia_alone)

# ============================================================
# Multi-device parallel execution
# ============================================================
print(f"\nRunning 2 GPUs in parallel (NVIDIA:{R_nvidia} + Intel:{R_intel} rows)...")

h_A_nv = h_A[:R_nvidia * N].copy()
h_A_in = h_A[R_nvidia * N:].copy()
h_C_nv = np.empty(R_nvidia * N, dtype=np.float32)
h_C_in = np.empty(R_intel * N, dtype=np.float32)

# NVIDIA buffers (dedicated VRAM)
d_a_nv = cl.Buffer(ctx_nv, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_A_nv)
d_c_nv = cl.Buffer(ctx_nv, cl.mem_flags.WRITE_ONLY, size=h_C_nv.nbytes)

# Intel buffers (zero-copy, shares system RAM)
d_b_in = make_zero_copy_buffer(ctx_in, queue_in, cl.mem_flags.READ_ONLY, h_B)
d_a_in = make_zero_copy_buffer(ctx_in, queue_in, cl.mem_flags.READ_ONLY, h_A_in)
d_c_in = cl.Buffer(ctx_in, cl.mem_flags.READ_WRITE | cl.mem_flags.ALLOC_HOST_PTR,
                    size=h_C_in.nbytes)

# Warmup
mmul_nv(queue_nv, (R_nvidia, N), (16, 16), np.int32(N), d_a_nv, d_b_nv, d_c_nv)
queue_nv.finish()
mmul_in(queue_in, (N // 8, R_intel // 8), (8, 8), np.int32(N), d_a_in, d_b_in, d_c_in)
queue_in.finish()

multi_times = []
for _ in range(COUNT):
    t0 = time()
    # Enqueue Intel first to maximize overlap
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

# Correctness check
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
