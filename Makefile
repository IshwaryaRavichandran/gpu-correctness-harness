NVCC = nvcc
CFLAGS = -Xcompiler -fPIC -shared -O3

all: build/libcuda_correctness.so

build/libcuda_correctness.so: kernels/vector_add.cu kernels/matrix_mul.cu
	mkdir -p build
	$(NVCC) $(CFLAGS) kernels/vector_add.cu kernels/matrix_mul.cu -o build/libcuda_correctness.so

clean:
	rm -rf build/
