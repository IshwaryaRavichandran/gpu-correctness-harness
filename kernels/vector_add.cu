#include <cuda_runtime.h>
#include <stdio.h>

extern "C" {

__global__ void vector_add_kernel(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int launch_vector_add(const float* h_A, const float* h_B, float* h_C, int N) {
    size_t bytes = N * sizeof(float);
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;

    if (cudaMalloc(&d_A, bytes) != cudaSuccess) return -1;
    if (cudaMalloc(&d_B, bytes) != cudaSuccess) return -2;
    if (cudaMalloc(&d_C, bytes) != cudaSuccess) return -3;

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    vector_add_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

}
