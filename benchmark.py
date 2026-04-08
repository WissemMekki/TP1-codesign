import os
os.environ['PYOPENCL_CTX'] = '0'  # NVIDIA GPU

import pyopencl as cl
import numpy as np
from time import time

# Constants
AVAL = 3.257
BVAL = 5.723
COUNT = 20

def run_kernel(context, queue, kernel_file, N, global_size, local_size):
    size = N * N
    h_A = np.empty(size).astype(np.float32); h_A.fill(AVAL)
    h_B = np.empty(size).astype(np.float32); h_B.fill(BVAL)
    h_C = np.empty(size).astype(np.float32)

    d_a = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_A)
    d_b = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_B)
    d_c = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, size=h_C.nbytes)

    source = open(kernel_file).read()
    program = cl.Program(context, source).build()
    mmul = program.mmul
    mmul.set_scalar_arg_dtypes([np.int32, None, None, None])

    # Warmup
    mmul(queue, global_size, local_size, np.int32(N), d_a, d_b, d_c)
    queue.finish()

    # Timed runs
    start = time()
    for _ in range(COUNT):
        mmul(queue, global_size, local_size, np.int32(N), d_a, d_b, d_c)
        queue.finish()
    elapsed = time() - start

    mflops = 2.0 * COUNT * N * N * N / (1000000.0 * elapsed)

    # Correctness check
    cl.enqueue_copy(queue, h_C, d_c)
    expected = float(N) * AVAL * BVAL
    max_err = float(np.max(np.abs(h_C - expected)))

    return mflops, elapsed, max_err

# Kernel configurations
kernels = [
    {
        'name': 'Baseline Coalesced',
        'file': 'C_elem_ji.cl',
        'global_size': lambda N: (N, N),
        'local_size': (16, 16),
    },
    {
        'name': 'Tiled (TS=32)',
        'file': 'C_tiled.cl',
        'global_size': lambda N: (N, N),
        'local_size': (32, 32),
    },
    {
        'name': '1D Reg Blocking (WPT=8)',
        'file': 'C_reg_blocking.cl',
        'global_size': lambda N: (N, N // 8),
        'local_size': (32, 4),
    },
    {
        'name': '2D Reg Blocking (best)',
        'file': 'C_optimized.cl',
        'global_size': lambda N: (N // 8, N // 8),
        'local_size': (8, 8),
    },
]

context = cl.create_some_context()
queue = cl.CommandQueue(context)

for N in [2048]:
    print(f"\n{'=' * 72}")
    print(f"  Matrix Multiplication {N} x {N}  (COUNT={COUNT})")
    print(f"{'=' * 72}")
    print(f"{'Kernel':<30} {'MFLOPS':>12} {'Speedup':>10} {'MaxErr':>10}")
    print(f"{'-' * 72}")

    baseline_mflops = None
    for cfg in kernels:
        gs = cfg['global_size'](N)
        ls = cfg['local_size']
        try:
            mflops, elapsed, err = run_kernel(context, queue, cfg['file'], N, gs, ls)
            if baseline_mflops is None:
                baseline_mflops = mflops
            speedup = mflops / baseline_mflops
            print(f"{cfg['name']:<30} {mflops:>12.1f} {speedup:>9.2f}x {err:>10.2f}")
        except Exception as e:
            print(f"{cfg['name']:<30} {'ERROR':>12} {'---':>10} {str(e)[:30]}")

print()
