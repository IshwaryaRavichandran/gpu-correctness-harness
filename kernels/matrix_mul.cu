
#include <cuda_runtime.h>
#define TILE 16

__global__ void matrix_mul_kernel(const float* A, const float* B, float* C, int N) {
    __shared__ float tA[TILE][TILE];
    __shared__ float tB[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;
    for (int t = 0; t < (N + TILE - 1) / TILE; ++t) {
        tA[threadIdx.y][threadIdx.x] = (row < N && t*TILE+threadIdx.x < N)
            ? A[row*N + t*TILE+threadIdx.x] : 0.0f;
        tB[threadIdx.y][threadIdx.x] = (col < N && t*TILE+threadIdx.y < N)
            ? B[(t*TILE+threadIdx.y)*N + col] : 0.0f;
        __syncthreads();
        for (int k = 0; k < TILE; ++k) sum += tA[threadIdx.y][k] * tB[k][threadIdx.x];
        __syncthreads();
    }
    if (row < N && col < N) C[row*N + col] = sum;
}

extern "C" void matrix_mul(const float* h_A, const float* h_B, float* h_C, int N) {
    float *d_A, *d_B, *d_C;
    size_t sz = N * N * sizeof(float);
    cudaMalloc(&d_A, sz); cudaMalloc(&d_B, sz); cudaMalloc(&d_C, sz);
    cudaMemcpy(d_A, h_A, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sz, cudaMemcpyHostToDevice);
    dim3 threads(TILE, TILE);
    dim3 blocks((N+TILE-1)/TILE, (N+TILE-1)/TILE);
    matrix_mul_kernel<<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C, d_C, sz, cudaMemcpyDeviceToHost);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

extern "C" float timed_matrix_mul(const float* h_A, const float* h_B, float* h_C, int N) {
    float *d_A, *d_B, *d_C;
    size_t sz = N * N * sizeof(float);
    cudaMalloc(&d_A, sz); cudaMalloc(&d_B, sz); cudaMalloc(&d_C, sz);
    cudaMemcpy(d_A, h_A, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sz, cudaMemcpyHostToDevice);
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    dim3 threads(TILE, TILE);
    dim3 blocks((N+TILE-1)/TILE, (N+TILE-1)/TILE);
    cudaEventRecord(start);
    matrix_mul_kernel<<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaMemcpy(h_C, d_C, sz, cudaMemcpyDeviceToHost);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return ms;
}
