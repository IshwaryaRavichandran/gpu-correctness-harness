#include <cuda_runtime.h>
#include <stdio.h>

#define TILE_WIDTH 16

extern "C" {

__global__ void matrix_mul_tiled_kernel(const float* A, const float* B, float* C, int Width) {
    __shared__ float s_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float s_B[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float Pvalue = 0.0f;

    for (int ph = 0; ph < (Width + TILE_WIDTH - 1) / TILE_WIDTH; ++ph) {
        if (Row < Width && (ph * TILE_WIDTH + tx) < Width)
            s_A[ty][tx] = A[Row * Width + ph * TILE_WIDTH + tx];
        else
            s_A[ty][tx] = 0.0f;

        if (Col < Width && (ph * TILE_WIDTH + ty) < Width)
            s_B[ty][tx] = B[(ph * TILE_WIDTH + ty) * Width + Col];
        else
            s_B[ty][tx] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += s_A[ty][k] * s_B[k][tx];
        }

        __syncthreads();
    }

    if (Row < Width && Col < Width) {
        C[Row * Width + Col] = Pvalue;
    }
}

int launch_matrix_mul(const float* h_A, const float* h_B, float* h_C, int N) {
    size_t bytes = N * N * sizeof(float);
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;

    if (cudaMalloc(&d_A, bytes) != cudaSuccess) return -1;
    if (cudaMalloc(&d_B, bytes) != cudaSuccess) return -2;
    if (cudaMalloc(&d_C, bytes) != cudaSuccess) return -3;

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((N + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH);

    matrix_mul_tiled_kernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}

} // extern "C"
