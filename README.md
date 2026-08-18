# gpu-correctness-harness

A hands-on GPU validation project built to bridge my background in test infrastructure and hardware design into CUDA systems engineering.

The harness compiles CUDA kernels via `nvcc`, loads them into Python through `ctypes`, and validates GPU output against NumPy CPU baselines, catching correctness failures at thread boundaries, tile edges, and numerical edge cases. Benchmark timing uses CUDA events inside the `.so`, not Python wall-clock.

---

## What I was trying to learn

I wanted to understand how GPU validation actually works at the systems level — not just writing kernels, but designing a harness that catches the bugs kernels are likely to have. Boundary-value analysis, dtype contract enforcement, numerical stability, and throughput regression are the same problems I've worked on in software test infrastructure, just one layer closer to the metal.

Building this taught me where that intuition transfers directly and where GPU-specific knowledge (warp execution, shared memory layout, FP32 overflow behavior) changes the game.

---

## Project layout

```
gpu-correctness-harness/
├── kernels/
│   ├── vector_add.cu       # element-wise FP32 add, 1D grid, 256 threads/block
│   ├── matrix_mul.cu       # square GEMM with 16x16 shared-memory tiling
│   ├── reduction.cu        # parallel sum — shared memory tree + warp-shuffle (__shfl_down_sync)
│   └── softmax.cu          # numerically stable softmax, 2-pass shared-memory reduction
├── src/
│   └── cuda_wrapper.py     # ctypes loader, dtype/shape validation, timed entry points
├── tests/
│   └── test_correctness.py # 23 tests: correctness, boundary values, contract enforcement
├── benchmarks/
│   └── bench.py            # CUDA-event throughput benchmark + regression guard
└── Makefile
```

---

## Kernels and what they test

**vector_add** — the baseline. Tests at `N=1`, `N=1000` (non-power-of-two), and `N=256` (exact block boundary). The boundary case is the one that matters: off-by-one in ceiling division silently drops the last element with no error message.

**matrix_mul** — naive tiled GEMM, parameterized over `[16, 32, 64, 128, 257]`. The 257 case deliberately crosses the 256-thread tile boundary. Index math that looks correct at power-of-two sizes often breaks here.

**reduction** — parallel sum using shared-memory tree reduction + `__shfl_down_sync` for the final warp. The warp-shuffle avoids unnecessary `__syncthreads` calls and shared memory bank conflicts in the last 32 lanes. Writing this kernel and its tests surfaced a real bug: the tree was stopping at `s > 32` instead of `s >= 32`, leaving 64 values where only 32 were being read by the warp — every result came back exactly half the correct answer. The N=256 boundary test caught it on the first run.

**softmax** — two-pass numerically stable implementation. Pass 1 computes `max(x)` via shared-memory reduction. Pass 2 computes `exp(x[i] - max)` and normalizes. The test `test_numerical_stability_large_values` passes `[1000.0, 1001.0, 1002.0]` — inputs that cause `exp(1000)` to overflow to `inf` in FP32, making naive softmax return NaN. Testing this also uncovered a second bug: launching with `threads = N` for small N breaks tree reduction when N isn't a power of 2 (with `N=3`, `blockDim.x/2 = 1`, so `smem[2]` is never compared and the kernel reports the wrong max). Fixed by always launching 256 threads and using strided loops — inactive threads contribute `-FLT_MAX` and `0.0`, which are identity values for max and sum.

---

## Build and run

```bash
# Requires CUDA Toolkit 11.x+ and Python 3.10+
make clean && make
pytest tests/ -v
python benchmarks/bench.py
```

---

## Results — NVIDIA T4, Google Colab

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

**Throughput benchmark** (CUDA events, warmup=5, runs=20, T4 peak ~320 GB/s):

```
  Kernel                 Avg ms    Bandwidth
  ----------------------------------------------------
  vector_add (16M)        0.765 ms    263.2 GB/s
  reduce_sum (16M)        0.629 ms    106.7 GB/s
  matrix_mul (512x512)    0.342 ms      9.2 GB/s
  softmax (1024)          0.018 ms      0.9 GB/s

All kernels within throughput thresholds.
```

`vector_add` hits 82% of T4 peak bandwidth — expected for a purely memory-bound kernel. `reduce_sum` at 107 GB/s reflects the cost of the two-phase design; a CUB-backed reduction keeps partial sums on-device and gets closer to 250 GB/s. `matrix_mul` bandwidth is intentionally low — naive GEMM has poor arithmetic intensity and that's fine here; the point of this kernel is correctness at N=257, not peak FLOPS.

---

## What's next

- FP16 kernel variants with tolerance-aware comparison against FP32 reference
- Online softmax for sequences longer than 1024 (multi-block, single-pass)
- `nvtx` range markers for Nsight Systems timeline visibility
- Parallel reduction kernel (warp-level max) and convolution correctness tests
- GitHub Actions CI with a GPU-enabled runner
