import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.cuda_wrapper import CUDACorrectnessWrapper

LIB_PATH = os.path.join(os.path.dirname(__file__), "..", "build", "libcuda_correctness.so")

@pytest.fixture(scope="module")
def cuda_driver():
    if not os.path.exists(LIB_PATH):
        pytest.skip(f"Compiled library not found at {LIB_PATH}. Run 'make' first.")
    return CUDACorrectnessWrapper(lib_path=LIB_PATH)

class TestVectorAdd:
    def test_equivalence_random(self, cuda_driver):
        rng = np.random.default_rng(seed=0)
        a = rng.random(1024, dtype=np.float32)
        b = rng.random(1024, dtype=np.float32)

        result = cuda_driver.vector_add(a, b)
        expected = a + b

        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)

    def test_single_element(self, cuda_driver):
        a = np.array([3.0], dtype=np.float32)
        b = np.array([4.0], dtype=np.float32)

        result = cuda_driver.vector_add(a, b)

        np.testing.assert_allclose(result, a + b, rtol=1e-5, atol=1e-6)

    def test_non_power_of_two_length(self, cuda_driver):
        rng = np.random.default_rng(seed=1)
        n = 1000
        a = rng.random(n, dtype=np.float32)
        b = rng.random(n, dtype=np.float32)

        result = cuda_driver.vector_add(a, b)

        np.testing.assert_allclose(result, a + b, rtol=1e-5, atol=1e-6)

    def test_exact_block_boundary(self, cuda_driver):
        for n in (255, 256, 257):
            a = np.arange(n, dtype=np.float32)
            b = np.arange(n, dtype=np.float32)[::-1].copy()

            result = cuda_driver.vector_add(a, b)

            np.testing.assert_allclose(result, a + b, rtol=1e-5, atol=1e-6)

    def test_mismatched_shapes_raises(self, cuda_driver):
        a = np.zeros(4, dtype=np.float32)
        b = np.zeros(5, dtype=np.float32)

        with pytest.raises(AssertionError):
            cuda_driver.vector_add(a, b)

    def test_wrong_dtype_raises(self, cuda_driver):
        a = np.zeros(4, dtype=np.float64)
        b = np.zeros(4, dtype=np.float64)

        with pytest.raises(AssertionError):
            cuda_driver.vector_add(a, b)

class TestMatrixMul:
    @pytest.mark.parametrize("dim", [16, 32, 64, 128, 257])
    def test_matrix_mul_equivalence(self, cuda_driver, dim):
        a = np.random.randn(dim, dim).astype(np.float32)
        b = np.random.randn(dim, dim).astype(np.float32)

        expected = a @ b
        actual = cuda_driver.matrix_mul(a, b)

        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)
