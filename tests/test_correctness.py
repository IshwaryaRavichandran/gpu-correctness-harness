import pytest
import numpy as np
import sys
import os

# Add src folder directly to Python's import path
sys.path.append(os.path.abspath("gpu-correctness-harness/src"))
from cuda_wrapper import CUDACorrectnessWrapper

@pytest.fixture(scope="module")
def cuda_driver():
    return CUDACorrectnessWrapper()

@pytest.mark.parametrize("dim", [16, 32, 64, 128, 257])
def test_matrix_mul_equivalence(cuda_driver, dim):
    """Validates tiled GPU GEMM against CPU NumPy golden reference."""
    a = np.random.randn(dim, dim).astype(np.float32)
    b = np.random.randn(dim, dim).astype(np.float32)

    expected = a @ b  # CPU Reference
    actual = cuda_driver.matrix_mul(a, b)

    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)
