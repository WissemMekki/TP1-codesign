"""
Calibrates the optimal row split between NVIDIA and Intel iGPU for Part B.
Benchmarks both GPUs individually, then uses adaptive search to find the
split where both devices finish at roughly the same time under contention.
Writes result to split_config.txt.
"""
import os
import pyopencl as cl
import numpy as np
from time import time

AVAL = 3.257
BVAL = 5.723
N = 8192
CALIBRATION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'split_config.txt')

# Search parameters
SMALL_STEP = 128        # minimum meaningful row shift (2 x 64)
MAX_ITERATIONS = 8
BENCH_REPS = 3          # repetitions per split test
CONVERGENCE_MS = 50     # stop if GPU times differ by less than this


def find_devices():
    nvidia_dev = intel_dev = None
    for p in cl.get_platforms():
        for d in p.get_devices():
            name = d.name.lower()
            if 'nvidia' in name:
                nvidia_dev = d
            elif 'intel' in name and ('uhd' in name or 'iris' in name):
                intel_dev = d
    assert nvidia_dev, "NVIDIA GPU not found"
    assert intel_dev, "Intel integrated GPU not found"
    return nvidia_dev, intel_dev


def benchmark_device(device, kernel_file, global_size, local_size, count=5):
    """Benchmark a single device on full NxN matrix multiplication."""
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
    mmul(queue, global_size, local_size, np.int32(N), d_a, d_b, d_c)
    queue.finish()
    # Timed runs
    t0 = time()
    for _ in range(count):
        mmul(queue, global_size, local_size, np.int32(N), d_a, d_b, d_c)
        queue.finish()
    elapsed = time() - t0
    return 2.0 * count * N**3 / (1e9 * elapsed)


def make_zero_copy_buffer(ctx, queue, flags, host_array):
    """Zero-copy buffer for iGPU (shares system RAM)."""
    buf = cl.Buffer(ctx, flags | cl.mem_flags.ALLOC_HOST_PTR, size=host_array.nbytes)
    ptr, _ = cl.enqueue_map_buffer(queue, buf, cl.map_flags.WRITE_INVALIDATE_REGION,
                                    0, host_array.shape, host_array.dtype)
    ptr[:] = host_array
    del ptr
    return buf


def snap_to_64(x):
    """Round to nearest multiple of 64, clamped within [64, N-64]."""
    x = int(round(x / 64)) * 64
    return max(64, min(N - 64, x))


def test_split(R_in, ctx_nv, ctx_in, queue_nv, queue_in, mmul_nv, mmul_in,
               h_A, h_B, d_b_nv, d_b_in):
    """
    Test a split with R_in rows for Intel, rest for NVIDIA.
    Returns (wall_time, T_nvidia, T_intel) averaged over BENCH_REPS runs.
    Uses OpenCL profiling events for individual kernel times.
    """
    R_nv = N - R_in

    h_A_nv = h_A[:R_nv * N].copy()
    h_A_in = h_A[R_nv * N:].copy()

    # NVIDIA buffers (dedicated VRAM)
    d_a_nv = cl.Buffer(ctx_nv, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_A_nv)
    d_c_nv = cl.Buffer(ctx_nv, cl.mem_flags.WRITE_ONLY, size=R_nv * N * 4)

    # Intel buffers (zero-copy, shares system RAM)
    d_a_in = make_zero_copy_buffer(ctx_in, queue_in, cl.mem_flags.READ_ONLY, h_A_in)
    d_c_in = cl.Buffer(ctx_in, cl.mem_flags.READ_WRITE | cl.mem_flags.ALLOC_HOST_PTR,
                        size=R_in * N * 4)

    # Warmup
    mmul_nv(queue_nv, (R_nv, N), (16, 16), np.int32(N), d_a_nv, d_b_nv, d_c_nv)
    queue_nv.finish()
    mmul_in(queue_in, (N // 8, R_in // 8), (8, 8), np.int32(N), d_a_in, d_b_in, d_c_in)
    queue_in.finish()

    wall_times, nv_times, in_times = [], [], []
    for _ in range(BENCH_REPS):
        t0 = time()
        # Enqueue Intel first to maximize overlap
        ev_in = mmul_in(queue_in, (N // 8, R_in // 8), (8, 8),
                        np.int32(N), d_a_in, d_b_in, d_c_in)
        queue_in.flush()
        ev_nv = mmul_nv(queue_nv, (R_nv, N), (16, 16),
                        np.int32(N), d_a_nv, d_b_nv, d_c_nv)
        queue_nv.flush()
        queue_nv.finish()
        queue_in.finish()
        wall_times.append(time() - t0)
        nv_times.append((ev_nv.profile.end - ev_nv.profile.start) * 1e-9)
        in_times.append((ev_in.profile.end - ev_in.profile.start) * 1e-9)

    return np.mean(wall_times), np.mean(nv_times), np.mean(in_times)


def find_optimal_split(R_in_start, nvidia_dev, intel_dev, h_A, h_B):
    """
    Adaptive search starting from theoretical split.
    Uses weighted barycenter of effective throughputs to estimate jumps.
    """
    ctx_nv = cl.Context([nvidia_dev])
    ctx_in = cl.Context([intel_dev])
    queue_nv = cl.CommandQueue(ctx_nv, properties=cl.command_queue_properties.PROFILING_ENABLE)
    queue_in = cl.CommandQueue(ctx_in, properties=cl.command_queue_properties.PROFILING_ENABLE)

    prg_nv = cl.Program(ctx_nv, open('kernels/C_elem_ij.cl').read()).build()
    mmul_nv = prg_nv.mmul
    mmul_nv.set_scalar_arg_dtypes([np.int32, None, None, None])

    prg_in = cl.Program(ctx_in, open('kernels/C_optimized.cl').read()).build()
    mmul_in = prg_in.mmul
    mmul_in.set_scalar_arg_dtypes([np.int32, None, None, None])

    # Shared B buffers
    d_b_nv = cl.Buffer(ctx_nv, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_B)
    d_b_in = make_zero_copy_buffer(ctx_in, queue_in, cl.mem_flags.READ_ONLY, h_B)

    shared = (ctx_nv, ctx_in, queue_nv, queue_in, mmul_nv, mmul_in, h_A, h_B, d_b_nv, d_b_in)

    R_in = R_in_start
    best_R_in = R_in
    best_wall = float('inf')
    visited = set()

    print(f"\n{'Iter':<6} {'NVIDIA rows':<13} {'Intel rows':<12} {'Wall(s)':<9} "
          f"{'T_nv(s)':<9} {'T_in(s)':<9} {'Jump':<8}")
    print("-" * 68)

    for i in range(MAX_ITERATIONS):
        R_nv = N - R_in
        wall, T_nv, T_in = test_split(R_in, *shared)

        if wall < best_wall:
            best_wall = wall
            best_R_in = R_in

        visited.add(R_in)

        # Estimate jump via weighted barycenter of effective throughputs
        rate_nv = R_nv / T_nv
        rate_in = R_in / T_in
        target_R_in = N * rate_in / (rate_nv + rate_in)
        jump = snap_to_64(target_R_in) - R_in

        imbalance_ms = abs(T_nv - T_in) * 1000

        print(f"{i+1:<6} {R_nv:<13} {R_in:<12} {wall:<9.3f} "
              f"{T_nv:<9.3f} {T_in:<9.3f} {jump:>+6}")

        # Converged: GPUs finish within threshold of each other
        if imbalance_ms < CONVERGENCE_MS:
            print(f"  Converged (imbalance {imbalance_ms:.0f}ms < {CONVERGENCE_MS}ms)")
            break

        # Nudge if jump too small
        if abs(jump) < SMALL_STEP:
            jump = SMALL_STEP if T_nv > T_in else -SMALL_STEP

        next_R_in = snap_to_64(R_in + jump)

        # Oscillation: already tested this split, stop and use best seen
        if next_R_in in visited:
            print(f"  Oscillating — picking best wall time")
            break

        R_in = next_R_in

    return best_R_in


def calibrate():
    """Full calibration: solo benchmarks → theoretical split → adaptive search → save."""
    nvidia_dev, intel_dev = find_devices()
    print(f"NVIDIA: {nvidia_dev.name}")
    print(f"Intel:  {intel_dev.name}")

    # Solo benchmarks
    print(f"\nBenchmarking NVIDIA (uncoalesced, full {N}x{N})...")
    P_nv = benchmark_device(nvidia_dev, 'kernels/C_elem_ij.cl', (N, N), (16, 16))
    print(f"  {P_nv:.2f} GFLOPS")

    print(f"Benchmarking Intel iGPU (optimized, full {N}x{N})...")
    P_in = benchmark_device(intel_dev, 'kernels/C_optimized.cl', (N // 8, N // 8), (8, 8))
    print(f"  {P_in:.2f} GFLOPS")

    # Theoretical split based on solo performance
    R_in_theory = snap_to_64(N * P_in / (P_nv + P_in))
    R_nv_theory = N - R_in_theory
    print(f"\nTheoretical split: NVIDIA {R_nv_theory} rows ({R_nv_theory*100//N}%) "
          f"/ Intel {R_in_theory} rows ({R_in_theory*100//N}%)")

    # Adaptive search under contention
    print(f"\nAdaptive search (starting from theoretical split)...")
    h_A = np.empty(N * N, dtype=np.float32); h_A.fill(AVAL)
    h_B = np.empty(N * N, dtype=np.float32); h_B.fill(BVAL)
    R_in_optimal = find_optimal_split(R_in_theory, nvidia_dev, intel_dev, h_A, h_B)

    print(f"\nOptimal split: NVIDIA {N - R_in_optimal} / Intel {R_in_optimal} rows")

    # Save to file
    with open(CALIBRATION_FILE, 'w') as f:
        f.write(str(R_in_optimal))
    print(f"Saved to {CALIBRATION_FILE}")

    return R_in_optimal


def get_split():
    """Read cached split from file, or run calibration if not found."""
    if os.path.exists(CALIBRATION_FILE):
        with open(CALIBRATION_FILE) as f:
            R_in = int(f.read().strip())
        print(f"Loaded cached split: Intel {R_in} rows / NVIDIA {N - R_in} rows")
        return R_in
    print("No cached split found, running calibration...")
    return calibrate()


if __name__ == '__main__':
    calibrate()
