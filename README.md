# GPU Kernel Correctness & Boundary Suite

A low-level CUDA validation harness written in **Python**, **C++**, and **PyTest**. This suite executes compiled CUDA kernels via Python `ctypes` bindings and verifies GPU mathematical accuracy against CPU NumPy golden reference models.

---

## 🛠️ Repository Architecture

```text
gpu-correctness-harness/
├── .gitignore
├── Makefile
├── README.md
├── requirements.txt
├── kernels/
│   ├── matrix_mul.cu
│   └── vector_add.cu
├── src/
│   ├── __init__.py
│   └── cuda_wrapper.py
└── tests/
    └── test_correctness.py
```

---

## 🚀 Quickstart & Build Instructions

### Prerequisites
* NVIDIA GPU + CUDA Toolkit (`nvcc`)
* Python 3.10+
* `pytest` and `numpy`

### 1. Compile CUDA Shared Object
```bash
make clean && make
```

### 2. Run Validation Suite
```bash
pytest tests/ -v
```

---

## 🧪 Testing & Validation Results

```text
============================= test session starts ==============================
platform linux -- Python 3.12.13, pytest-8.4.2, pluggy-1.6.0
rootdir: /content/gpu-correctness-harness
collected 11 items

tests/test_correctness.py::TestVectorAdd::test_equivalence_random PASSED [  9%]
tests/test_correctness.py::TestVectorAdd::test_single_element PASSED     [ 18%]
tests/test_correctness.py::TestVectorAdd::test_non_power_of_two_length PASSED [ 27%]
tests/test_correctness.py::TestVectorAdd::test_exact_block_boundary PASSED [ 36%]
tests/test_correctness.py::TestVectorAdd::test_mismatched_shapes_raises PASSED [ 45%]
tests/test_correctness.py::TestVectorAdd::test_wrong_dtype_raises PASSED [ 54%]
tests/test_correctness.py::TestMatrixMul::test_matrix_mul_equivalence[16] PASSED [ 63%]
tests/test_correctness.py::TestMatrixMul::test_matrix_mul_equivalence[32] PASSED [ 72%]
tests/test_correctness.py::TestMatrixMul::test_matrix_mul_equivalence[64] PASSED [ 81%]
tests/test_correctness.py::TestMatrixMul::test_matrix_mul_equivalence[128] PASSED [ 90%]
tests/test_correctness.py::TestMatrixMul::test_matrix_mul_equivalence[257] PASSED [100%]

============================== 11 passed in 0.42s ==============================
```
