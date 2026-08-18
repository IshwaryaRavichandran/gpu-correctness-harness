"""
bench.py — throughput benchmark + regression guard.
Timing uses CUDA events inside the .so (not Python wall-clock).
Exits non-zero if any kernel falls below its minimum GB/s threshold.
"""

import sys
import numpy as np
from src.cuda_wrapper import timed_vector_add, timed_matrix_mul, timed_reduce_sum, timed_softmax

WARMUP_RUNS = 5
BENCH_RUNS  = 20

MIN_THROUGHPUT_GBS = {
    "vector_add": 16.0,
    "reduce_sum":  8.0,
}

def _mean_ms(fn, *args):
    return sum(fn(*args)[1] for _ in range(BENCH_RUNS)) / BENCH_RUNS

def _bw(n_bytes, ms):
    return (n_bytes / 1e9) / (ms / 1e3)

def _print(name, n_bytes, ms, threshold):
    bw = _bw(n_bytes, ms)
    flag = f"  !! BELOW {threshold} GB/s" if threshold and bw < threshold else ""
    print(f"  {name:<22} {ms:7.3f} ms    {bw:6.1f} GB/s{flag}")
    return bw

if __name__ == "__main__":
    print(f"\nKernel Benchmark  (warmup={WARMUP_RUNS}, runs={BENCH_RUNS})\n")
    print(f"  {'Kernel':<22} {'Avg ms':>8}    {'Bandwidth'}")
    print(f"  {'-'*52}")

    failed = []

    N = 1 << 24
    a = np.random.rand(N).astype(np.float32)
    b = np.random.rand(N).astype(np.float32)
    [timed_vector_add(a, b) for _ in range(WARMUP_RUNS)]
    ms = _mean_ms(timed_vector_add, a, b)
    bw = _print("vector_add (16M)", 3*N*4, ms, MIN_THROUGHPUT_GBS["vector_add"])
    if bw < MIN_THROUGHPUT_GBS["vector_add"]: failed.append("vector_add")

    x = np.random.rand(N).astype(np.float32)
    [timed_reduce_sum(x) for _ in range(WARMUP_RUNS)]
    ms = _mean_ms(timed_reduce_sum, x)
    bw = _print("reduce_sum (16M)", N*4, ms, MIN_THROUGHPUT_GBS["reduce_sum"])
    if bw < MIN_THROUGHPUT_GBS["reduce_sum"]: failed.append("reduce_sum")

    M = 512
    A = np.random.rand(M, M).astype(np.float32)
    B = np.random.rand(M, M).astype(np.float32)
    [timed_matrix_mul(A, B) for _ in range(WARMUP_RUNS)]
    ms = _mean_ms(timed_matrix_mul, A, B)
    _print("matrix_mul (512x512)", 3*M*M*4, ms, None)

    s = np.random.rand(1024).astype(np.float32)
    [timed_softmax(s) for _ in range(WARMUP_RUNS)]
    ms = _mean_ms(timed_softmax, s)
    _print("softmax (1024)", 2*1024*4, ms, None)

    print()
    if failed:
        print(f"REGRESSION: {', '.join(failed)} below threshold.")
        sys.exit(1)
    print("All kernels within throughput thresholds.")
