"""
test_correctness.py

Validates CUDA kernel outputs against NumPy CPU baselines.
Boundary cases first — happy-path tests are a sanity check;
the cases that catch real bugs are at thread/tile edges and
the inputs that expose numerical instability.
"""

import numpy as np
import pytest
from src.cuda_wrapper import vector_add, matrix_mul, reduce_sum, softmax


class TestVectorAdd:

    def test_equivalence_random(self):
        rng = np.random.default_rng(42)
        a = rng.random(10_000, dtype=np.float32)
        b = rng.random(10_000, dtype=np.float32)
        assert np.allclose(vector_add(a, b), a + b, atol=1e-5)

    def test_single_element(self):
        a = np.array([3.14], dtype=np.float32)
        b = np.array([2.71], dtype=np.float32)
        assert np.allclose(vector_add(a, b), a + b, atol=1e-5)

    def test_non_power_of_two_length(self):
        a = np.ones(1000, dtype=np.float32)
        b = np.ones(1000, dtype=np.float32)
        assert np.allclose(vector_add(a, b), np.full(1000, 2.0, dtype=np.float32), atol=1e-5)

    def test_exact_block_boundary(self):
        a = np.ones(256, dtype=np.float32)
        b = np.ones(256, dtype=np.float32)
        assert np.allclose(vector_add(a, b), np.full(256, 2.0, dtype=np.float32), atol=1e-5)

    def test_mismatched_shapes_raises(self):
        with pytest.raises(ValueError, match="Shape mismatch"):
            vector_add(np.ones(10, dtype=np.float32), np.ones(20, dtype=np.float32))

    def test_wrong_dtype_raises(self):
        with pytest.raises(ValueError, match="float32"):
            vector_add(np.ones(10, dtype=np.float64), np.ones(10, dtype=np.float64))


class TestMatrixMul:

    @pytest.mark.parametrize("N", [16, 32, 64, 128, 257])
    def test_matrix_mul_equivalence(self, N):
        rng = np.random.default_rng(7)
        a = rng.random((N, N), dtype=np.float32)
        b = rng.random((N, N), dtype=np.float32)
        ref = (a @ b).astype(np.float32)
        assert np.allclose(matrix_mul(a, b), ref, atol=1e-3), f"MatMul mismatch at N={N}"


class TestReduction:

    def test_sum_random(self):
        rng = np.random.default_rng(0)
        x = rng.random(50_000, dtype=np.float32)
        assert abs(reduce_sum(x) - float(x.sum())) / float(x.sum()) < 1e-4

    def test_sum_single_element(self):
        assert abs(reduce_sum(np.array([42.0], dtype=np.float32)) - 42.0) < 1e-5

    def test_sum_exact_block_boundary(self):
        assert abs(reduce_sum(np.ones(256, dtype=np.float32)) - 256.0) < 1e-4

    def test_sum_non_power_of_two(self):
        assert abs(reduce_sum(np.ones(1001, dtype=np.float32)) - 1001.0) < 1e-4

    def test_sum_multi_block(self):
        assert abs(reduce_sum(np.ones(100_000, dtype=np.float32)) - 100_000.0) < 1.0

    def test_all_zeros(self):
        assert reduce_sum(np.zeros(512, dtype=np.float32)) == 0.0


class TestSoftmax:

    def _ref(self, x):
        e = np.exp(x - x.max())
        return (e / e.sum()).astype(np.float32)

    def test_equivalence_random(self):
        rng = np.random.default_rng(3)
        x = rng.random(256, dtype=np.float32)
        assert np.allclose(softmax(x), self._ref(x), atol=1e-5)

    def test_output_sums_to_one(self):
        x = np.random.default_rng(5).random(512, dtype=np.float32)
        assert abs(softmax(x).sum() - 1.0) < 1e-5

    def test_numerical_stability_large_values(self):
        """Naive softmax produces NaN here. Stable version must pass."""
        x = np.array([1000.0, 1001.0, 1002.0], dtype=np.float32)
        out = softmax(x)
        assert not np.any(np.isnan(out)), "NaN — missing max-subtraction in kernel"
        assert not np.any(np.isinf(out)), "Inf — overflow in exp()"
        assert np.allclose(out, self._ref(x), atol=1e-5)

    def test_uniform_input_gives_uniform_output(self):
        N = 128
        out = softmax(np.ones(N, dtype=np.float32))
        assert np.allclose(out, np.full(N, 1.0 / N, dtype=np.float32), atol=1e-6)

    def test_single_element(self):
        assert abs(softmax(np.array([5.0], dtype=np.float32))[0] - 1.0) < 1e-6

    def test_oversized_input_raises(self):
        with pytest.raises(ValueError, match="1024"):
            softmax(np.ones(2000, dtype=np.float32))
