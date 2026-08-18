"""
cuda_wrapper.py

Loads libkernels.so and exposes typed Python interfaces for each kernel.
All shape/dtype validation happens here before anything touches device memory.
"""

import ctypes
import os
import numpy as np

_LIB_PATH = os.path.join(os.path.dirname(__file__), "..", "build", "libkernels.so")

def _load_lib():
    if not os.path.exists(_LIB_PATH):
        raise FileNotFoundError(f"Shared library not found at {_LIB_PATH}. Run `make` first.")
    return ctypes.CDLL(_LIB_PATH)

_lib = _load_lib()
_f32p = ctypes.POINTER(ctypes.c_float)

_lib.vector_add.argtypes        = [_f32p, _f32p, _f32p, ctypes.c_int]
_lib.vector_add.restype         = None
_lib.timed_vector_add.argtypes  = [_f32p, _f32p, _f32p, ctypes.c_int]
_lib.timed_vector_add.restype   = ctypes.c_float
_lib.matrix_mul.argtypes        = [_f32p, _f32p, _f32p, ctypes.c_int]
_lib.matrix_mul.restype         = None
_lib.timed_matrix_mul.argtypes  = [_f32p, _f32p, _f32p, ctypes.c_int]
_lib.timed_matrix_mul.restype   = ctypes.c_float
_lib.parallel_reduce_sum.argtypes = [_f32p, _f32p, ctypes.c_int]
_lib.parallel_reduce_sum.restype  = None
_lib.timed_reduce_sum.argtypes  = [_f32p, _f32p, ctypes.c_int]
_lib.timed_reduce_sum.restype   = ctypes.c_float
_lib.cuda_softmax.argtypes      = [_f32p, _f32p, ctypes.c_int]
_lib.cuda_softmax.restype       = None
_lib.timed_softmax.argtypes     = [_f32p, _f32p, ctypes.c_int]
_lib.timed_softmax.restype      = ctypes.c_float

def _as_f32(arr, name):
    if arr.dtype != np.float32:
        raise ValueError(f"{name} must be float32, got {arr.dtype}")
    return np.ascontiguousarray(arr)

def _ptr(arr):
    return arr.ctypes.data_as(_f32p)

def vector_add(a, b):
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    a, b = _as_f32(a, "a"), _as_f32(b, "b")
    c = np.empty_like(a)
    _lib.vector_add(_ptr(a), _ptr(b), _ptr(c), ctypes.c_int(a.size))
    return c

def timed_vector_add(a, b):
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    a, b = _as_f32(a, "a"), _as_f32(b, "b")
    c = np.empty_like(a)
    ms = _lib.timed_vector_add(_ptr(a), _ptr(b), _ptr(c), ctypes.c_int(a.size))
    return c, float(ms)

def matrix_mul(a, b):
    if a.ndim != 2 or a.shape != b.shape or a.shape[0] != a.shape[1]:
        raise ValueError(f"Expected square matrices of equal size, got {a.shape} and {b.shape}")
    a, b = _as_f32(a, "a"), _as_f32(b, "b")
    c = np.empty_like(a)
    _lib.matrix_mul(_ptr(a), _ptr(b), _ptr(c), ctypes.c_int(a.shape[0]))
    return c

def timed_matrix_mul(a, b):
    if a.ndim != 2 or a.shape != b.shape or a.shape[0] != a.shape[1]:
        raise ValueError(f"Expected square matrices of equal size, got {a.shape} and {b.shape}")
    a, b = _as_f32(a, "a"), _as_f32(b, "b")
    c = np.empty_like(a)
    ms = _lib.timed_matrix_mul(_ptr(a), _ptr(b), _ptr(c), ctypes.c_int(a.shape[0]))
    return c, float(ms)

def reduce_sum(x):
    x = _as_f32(x.ravel(), "x")
    result = np.zeros(1, dtype=np.float32)
    _lib.parallel_reduce_sum(_ptr(x), _ptr(result), ctypes.c_int(x.size))
    return float(result[0])

def timed_reduce_sum(x):
    x = _as_f32(x.ravel(), "x")
    result = np.zeros(1, dtype=np.float32)
    ms = _lib.timed_reduce_sum(_ptr(x), _ptr(result), ctypes.c_int(x.size))
    return float(result[0]), float(ms)

def softmax(x):
    if x.ndim != 1:
        raise ValueError(f"Expected 1D input, got shape {x.shape}")
    if x.size > 1024:
        raise ValueError(f"Kernel supports N <= 1024 (got {x.size}).")
    x = _as_f32(x, "x")
    out = np.empty_like(x)
    _lib.cuda_softmax(_ptr(x), _ptr(out), ctypes.c_int(x.size))
    return out

def timed_softmax(x):
    if x.ndim != 1:
        raise ValueError(f"Expected 1D input, got shape {x.shape}")
    if x.size > 1024:
        raise ValueError(f"Kernel supports N <= 1024 (got {x.size}).")
    x = _as_f32(x, "x")
    out = np.empty_like(x)
    ms = _lib.timed_softmax(_ptr(x), _ptr(out), ctypes.c_int(x.size))
    return out, float(ms)
