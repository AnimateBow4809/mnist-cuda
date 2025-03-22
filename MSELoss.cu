#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "LossFunction.cuh"

#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while (0)

__global__ void mse_forward_kernel(const float* pred, const float* target, float* loss, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int batch = blockIdx.y;
    __shared__ float cache[256];
    int tid = threadIdx.x;

    float diff = 0.0f;
    if (idx < size) {
        diff = pred[idx + batch * size] - target[idx + batch * size];
        cache[tid] = diff * diff;
    }
    else {
        cache[tid] = 0.0f;
    }
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            cache[tid] += cache[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(&loss[batch], cache[0] / size);
    }
}

__global__ void mse_backward_kernel(const float* pred, const float* target, float* grad, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int batch = blockIdx.y;
    if (idx < size) {
        grad[batch * size + idx] = (2.0f / size) * (pred[batch * size + idx] - target[batch * size + idx]);
    }
}

float* MSELoss::forward(const float* predictions, const float* targets, int size, int batch) {
    float* d_loss;
    CUDA_CHECK(cudaMalloc(&d_loss, batch * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_loss, 0, batch * sizeof(float)));

    int threads = 256;
    dim3 blocks((size + threads - 1) / threads, batch);
    mse_forward_kernel << <blocks, threads >> > (predictions, targets, d_loss, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return d_loss;
}

void MSELoss::backward(const float* predictions, const float* targets, float* grad, int size, int batch) {
    int threads = 256;
    dim3 blocks((size + threads - 1) / threads, batch);
    mse_backward_kernel << <blocks, threads >> > (predictions, targets, grad, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}