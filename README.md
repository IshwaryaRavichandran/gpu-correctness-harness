# GPU Correctness Harness

CUDA kernels don't throw exceptions when they're wrong. A misaligned tile boundary, a warp reading half the reduction, exp() overflowing to inf—the output is just silently incorrect. This harness surfaces those failure modes.

Kernels compile via nvcc, load into Python through ctypes, and get checked against NumPy baselines. Timing measurements use CUDA events inside the compiled library, not wall-clock.

## Architecture

```
+-----------------------------------------------------------------------+
|                          PyTest Framework                             |
|      (Boundary Analysis, Softmax Overflow, Edge-Case Vectors)         |
+-----------------------------------------------------------------------+
                                  |
                        (ctypes C-ABI Wrapper)
                                  v
+-----------------------------------------------------------------------+
|                  C++/CUDA Shared Library (.so)                        |
|  +---------------------------------------------------------------+    |
|  | CUDA Event Timer (Start → Launch Kernel → Stop)               |    |
|  +---------------------------------------------------------------+    |
|  | Kernels: vector_add | matrix_mul | reduction | softmax        |    |
|  +---------------------------------------------------------------+    |
+-----------------------------------------------------------------------+
                                  |
                          (Device Execution)
                                  v
+-----------------------------------------------------------------------+
|                      NVIDIA GPU (T4)                                  |
|         Warp-Shuffle Intrinsic (__shfl_down_sync) Execution           |
+-----------------------------------------------------------------------+
```

## Kernel Coverage

### Vector Add
N=256 and N=257 are not the same test. One fills a block exactly. The other has a partial block where off-by-one in ceiling division silently drops the last element.

### Matrix Multiply
A tiled GEMM parameterized over tile sizes [16, 32, 64, 128, 257]. Everything passes at powers of two. 257 is where index math breaks.

### Reduction
Shared memory tree down to 32 lanes, then __shfl_down_sync for the final warp. No bank conflicts, no unnecessary syncs.

### Softmax
Two-pass stable algorithm. exp(1000) is inf in FP32. The naive kernel fails this test.

## Getting Started

### Requirements

* CUDA Toolkit 11.x or higher
* Python 3.10 or higher
* Make build system
* NVIDIA GPU (tested on T4)

### Build & Test

Clean build:

```bash
make clean && make
```

Run the full test suite:

```bash
pytest tests/ -v
```

Run performance benchmarks:

```bash
python benchmarks/bench.py
```

## Test Results

All 23 tests passing on NVIDIA T4:

```
tests/test_correctness.py::TestVectorAdd::test_equivalence_random            PASSED
tests/test_correctness.py::TestVectorAdd::test_single_element                PASSED
tests/test_correctness.py::TestVectorAdd::test_non_power_of_two_length       PASSED
tests/test_correctness.py::TestVectorAdd::test_exact_block_boundary          PASSED
tests/test_correctness.py::TestVectorAdd::test_mismatched_shapes_raises      PASSED
tests/test_correctness.py::TestVectorAdd::test_wrong_dtype_raises            PASSED
tests/test_correctness.py::TestMatrixMul::test_matrix_mul_equivalence[16]    PASSED
tests/test_correctness.py::TestMatrixMul::test_matrix_mul_equivalence[32]    PASSED
tests/test_correctness.py::TestMatrixMul::test_matrix_mul_equivalence[64]    PASSED
tests/test_correctness.py::TestMatrixMul::test_matrix_mul_equivalence[128]   PASSED
tests/test_correctness.py::TestMatrixMul::test_matrix_mul_equivalence[257]   PASSED
tests/test_correctness.py::TestReduction::test_sum_random                    PASSED
tests/test_correctness.py::TestReduction::test_sum_single_element            PASSED
tests/test_correctness.py::TestReduction::test_sum_exact_block_boundary      PASSED
tests/test_correctness.py::TestReduction::test_sum_non_power_of_two          PASSED
tests/test_correctness.py::TestReduction::test_sum_multi_block               PASSED
tests/test_correctness.py::TestReduction::test_all_zeros                     PASSED
tests/test_correctness.py::TestSoftmax::test_equivalence_random              PASSED
tests/test_correctness.py::TestSoftmax::test_output_sums_to_one             PASSED
tests/test_correctness.py::TestSoftmax::test_numerical_stability_large_values PASSED
tests/test_correctness.py::TestSoftmax::test_uniform_input_gives_uniform_output PASSED
tests/test_correctness.py::TestSoftmax::test_single_element                  PASSED
tests/test_correctness.py::TestSoftmax::test_oversized_input_raises          PASSED

23 passed in 0.45s
```

## Performance Benchmarks

CUDA event timing with warmup=5, 20 runs per kernel. NVIDIA T4 peak bandwidth is approximately 320 GB/s.

| Kernel | Input Size | Avg Time | Bandwidth |
|--------|-----------|----------|-----------|
| vector_add | 16M elements | 0.765 ms | 263.2 GB/s |
| reduce_sum | 16M elements | 0.629 ms | 106.7 GB/s |
| matrix_mul | 512x512 | 0.342 ms | 9.2 GB/s |
| softmax | 1024 | 0.018 ms | 0.9 GB/s |

### Performance Notes

Vector add achieves 263 GB/s, which is 82% of T4 peak. This is expected for a memory-bound kernel.

Reduce sum achieves 107 GB/s. The two-phase design adds overhead. Specialized libraries like CUB achieve closer to 250 GB/s with fused algorithms.

Matrix multiply bandwidth is intentionally low. This is a naive GEMM with poor arithmetic intensity. The relevant metric here is correctness at edge cases like N=257, not peak performance.

## Roadmap

* FP16 and BF16 kernel variants
* Online softmax algorithm for sequences beyond N=1024
* NVTX markers for Nsight timeline profiling
* CI integration with GPU runner

## Key Takeaways

This harness validates CUDA kernel correctness across:
* Non-power-of-two dimensions
* Boundary conditions and partial blocks
* Numerical stability under extreme values
* Warp-level synchronization and memory safety
* Real-world edge case dimensions (257 is not random)

The goal is to catch silent failures before they reach production, validating both the kernel logic and the infrastructure that wraps it.
