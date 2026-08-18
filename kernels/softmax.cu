
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

__global__ void softmax_kernel(const float* __restrict__ input,
                                float*       __restrict__ output,
                                int N) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;

    // Pass 1: parallel max reduction
    float local_max = -FLT_MAX;
    for (int i = tid; i < N; i += blockDim.x)
        local_max = fmaxf(local_max, input[i]);
    smem[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        __syncthreads();
    }
    float row_max = smem[0];
    __syncthreads();

    // Pass 2: exp(x - max) and sum
    float local_sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        output[i] = expf(input[i] - row_max);
        local_sum += output[i];
    }
    smem[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float row_sum = smem[0];
    __syncthreads();

    for (int i = tid; i < N; i += blockDim.x)
        output[i] /= row_sum;
}

extern "C" void cuda_softmax(const float* h_input, float* h_output, int N) {
    float *d_input, *d_output;
    cudaMalloc(&d_input,  N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    // BUG WAS HERE: threads = N for small N → non-power-of-2 block size breaks
    // tree reduction. e.g. N=3 → blockDim.x/2=1, smem[2] never compared.
    // FIX: always use 256 threads. Strided loops handle any N;
    // inactive threads contribute -FLT_MAX (max) and 0.0 (sum) — safe identity values.
    int threads = 256;
    softmax_kernel<<<1, threads, threads * sizeof(float)>>>(d_input, d_output, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_input); cudaFree(d_output);
}

extern "C" float timed_softmax(const float* h_input, float* h_output, int N) {
    float *d_input, *d_output;
    cudaMalloc(&d_input,  N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    int threads = 256;
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    softmax_kernel<<<1, threads, threads * sizeof(float)>>>(d_input, d_output, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_input); cudaFree(d_output);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return ms;
}
