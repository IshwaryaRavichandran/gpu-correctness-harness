NVCC      = nvcc
NVCCFLAGS = -O2 -shared -Xcompiler -fPIC --std=c++14

BUILD_DIR = build
LIB       = $(BUILD_DIR)/libkernels.so

SRCS = kernels/vector_add.cu \
       kernels/matrix_mul.cu \
       kernels/reduction.cu  \
       kernels/softmax.cu

.PHONY: all clean

all: $(BUILD_DIR) $(LIB)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(LIB): $(SRCS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

clean:
	rm -rf $(BUILD_DIR)
