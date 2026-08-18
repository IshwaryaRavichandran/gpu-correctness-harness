
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void block_reduce_sum_kernel(const float* __restrict__ input,
                                         float* __restrict__ partial_sums,
                                         int N) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (gid < N) ? input[gid] : 0.0f;
    __syncthreads();

    // BUG WAS HERE: s > 32 left 64 elements; warp only read 32 → half the answer
    // FIX: s >= 32 reduces to exactly 32 before handing off to warp shuffle
    for (int s = blockDim.x / 2; s >= 32; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid < 32) {
        float val = sdata[tid];
        val = warp_reduce_sum(val);
        if (tid == 0) partial_sums[blockIdx.x] = val;
    }
}

extern "C" void parallel_reduce_sum(const float* h_input, float* h_result, int N) {
    float *d_input, *d_partial;
    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cudaMalloc(&d_input,   N      * sizeof(float));
    cudaMalloc(&d_partial, blocks * sizeof(float));
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    block_reduce_sum_kernel<<<blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_input, d_partial, N);
    cudaDeviceSynchronize();
    float* h_partial = new float[blocks];
    cudaMemcpy(h_partial, d_partial, blocks * sizeof(float), cudaMemcpyDeviceToHost);
    float total = 0.0f;
    for (int i = 0; i < blocks; i++) total += h_partial[i];
    *h_result = total;
    delete[] h_partial;
    cudaFree(d_input); cudaFree(d_partial);
}

extern "C" float timed_reduce_sum(const float* h_input, float* h_result, int N) {
    float *d_input, *d_partial;
    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cudaMalloc(&d_input,   N      * sizeof(float));
    cudaMalloc(&d_partial, blocks * sizeof(float));
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    block_reduce_sum_kernel<<<blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_input, d_partial, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    float* h_partial = new float[blocks];
    cudaMemcpy(h_partial, d_partial, blocks * sizeof(float), cudaMemcpyDeviceToHost);
    float total = 0.0f;
    for (int i = 0; i < blocks; i++) total += h_partial[i];
    *h_result = total;
    delete[] h_partial;
    cudaFree(d_input); cudaFree(d_partial);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return ms;
}
