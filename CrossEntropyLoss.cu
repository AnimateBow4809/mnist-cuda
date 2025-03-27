#include "lossFunction.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include "iostream"


#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while (0)


// Kernel for Cross-Entropy Loss forward pass
__global__ void crossEntropyForwardKernel(const float* predictions, const float* targets, float* loss, int size, int batch) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch) return;

    float sum_loss = 0.0f;
    for (int i = 0; i < size; i++) {
        int index = idx * size + i;
        sum_loss += -targets[index] * logf(fmaxf(predictions[index], 1e-8f)); // Avoid log(0)
    }
    loss[idx] = sum_loss / size;
}

// Kernel for Cross-Entropy Loss backward pass
__global__ void crossEntropyBackwardKernel(const float* predictions, const float* targets, float* grad, int size, int batch) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * size) return;

    grad[idx] = (predictions[idx] - targets[idx]) / size;
}

// Forward pass
float* CrossEntropyLoss::forward(const float* predictions, const float* targets, int size, int batch) {
    float* d_loss;
    CUDA_CHECK(cudaMalloc(&d_loss, batch * sizeof(float)));
    int threads = 256;
    int blocks = (batch + threads - 1) / threads;
    crossEntropyForwardKernel << <blocks, threads >> > (predictions, targets, d_loss, size, batch);
    cudaDeviceSynchronize();

    return d_loss;
}

// Backward pass
void CrossEntropyLoss::backward(const float* predictions, const float* targets, float* grad, int size, int batch) {
    int threads = 256;
    int blocks = (batch * size + threads - 1) / threads;
    crossEntropyBackwardKernel << <blocks, threads >> > (predictions, targets, grad, size, batch);
    cudaDeviceSynchronize();
}
