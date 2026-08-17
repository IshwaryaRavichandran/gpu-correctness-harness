import os
import ctypes
import numpy as np

class CUDACorrectnessWrapper:
    def __init__(self, lib_path: str = "build/libcuda_correctness.so"):
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"Compiled shared object not found at {lib_path}. Run 'make' first.")

        self.lib = ctypes.CDLL(lib_path)

        self.lib.launch_vector_add.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),
            ctypes.c_int
        ]
        self.lib.launch_vector_add.restype = ctypes.c_int

        self.lib.launch_matrix_mul.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),
            ctypes.c_int
        ]
        self.lib.launch_matrix_mul.restype = ctypes.c_int

    def vector_add(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        assert a.shape == b.shape, "Input arrays must have identical shapes"
        assert a.dtype == np.float32 and b.dtype == np.float32, "Arrays must be float32"

        n = a.shape[0]
        c = np.empty(n, dtype=np.float32)

        status = self.lib.launch_vector_add(a, b, c, n)
        if status != 0:
            raise RuntimeError(f"CUDA execution failed with status code {status}")

        return c

    def matrix_mul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        assert a.shape == b.shape, "Matrices must be square and identical shape"
        assert a.dtype == np.float32 and b.dtype == np.float32

        n = a.shape[0]
        c = np.empty((n, n), dtype=np.float32)

        status = self.lib.launch_matrix_mul(a, b, c, n)
        if status != 0:
            raise RuntimeError(f"CUDA matrix multiplication failed with error code {status}")

        return c
