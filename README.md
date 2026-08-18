# gpu-correctness-harness

# gpu-correctness-harness

CUDA kernels don't throw exceptions when they're wrong. A misaligned 
tile boundary, a warp reading half the reduction, exp() overflowing 
to inf, the output is just silently incorrect. This harness surfaces 
those failure modes.

Kernels compile via `nvcc`, load into Python through `ctypes`, and 
get checked against NumPy baselines. Timing is CUDA events inside 
the `.so`,  not wall-clock.

---

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

---

## Kernels

## Kernels

**vector_add** : N=256 and N=257 are not the same test. One fills 
a block exactly. The other has a partial block where off-by-one in 
ceiling division silently drops the last element.

**matrix_mul** : tiled GEMM parameterized over [16, 32, 64, 128, 257]. 
Everything passes at powers of two. 257 is where index math breaks.

**reduction** : shared memory tree down to 32 lanes, then 
`__shfl_down_sync` for the final warp. No bank conflicts, no 
unnecessary syncs.

**softmax** : two-pass stable. exp(1000) is inf in FP32. 
`test_numerical_stability_large_values` is the test that fails 
against a naive implementation.
---

## Run it

```bash
make clean && make
pytest tests/ -v
python benchmarks/bench.py
```

Requires CUDA Toolkit 11.x+ and Python 3.10+.

---

## Results — NVIDIA T4

**23/23 tests passing:**

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

**Throughput** (CUDA events, warmup=5, runs=20, T4 peak ~320 GB/s):

```
  Kernel                 Avg ms    Bandwidth
  ----------------------------------------------------
  vector_add (16M)        0.765 ms    263.2 GB/s
  reduce_sum (16M)        0.629 ms    106.7 GB/s
  matrix_mul (512x512)    0.342 ms      9.2 GB/s
  softmax (1024)          0.018 ms      0.9 GB/s
```

`vector_add` at 263 GB/s is 82% of T4 peak, expected for a memory-bound kernel. `reduce_sum` at 107 GB/s reflects two-phase design cost; on-device reduction (CUB) gets closer to 250 GB/s. `matrix_mul` bandwidth is low by design, naive GEMM has poor arithmetic intensity; the relevant number here is correctness at N=257.

---

## Next

FP16/BF16 variants, online softmax past N=1024, nvtx markers 
for Nsight timelines, CI with a GPU runner.
