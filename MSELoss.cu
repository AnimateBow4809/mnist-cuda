#include "iostream"
#include "LossFunction.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void mse_forward_kernel(const float* pred, const float* target, float* loss, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int batch = blockIdx.y;

    __shared__ float cache[256];  // Shared memory for parallel reduction (assumes max 256 threads per block)
    int tid = threadIdx.x;
    cache[tid] = 0.0f;

    if (idx < size) {
        float diff = pred[idx + batch * size] - target[idx + batch * size];
        cache[tid] = diff * diff;  // Squared error
    }
    __syncthreads();

    // Parallel reduction to sum squared errors
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            cache[tid] += cache[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(&loss[batch], cache[0] / size);  // Average over size
    }
}

__global__ void mse_backward_kernel(const float* pred, const float* target, float* grad, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int batch = blockIdx.y;

    if (idx < size) {
        grad[batch*size+idx] = 2.0f * (pred[batch*size+idx] - target[batch*size+idx]) / size; // Gradient of MSE
    }
}

float* MSELoss::forward(const float* predictions, const float* targets, int size, int batch) {
    float* d_loss;
    cudaMalloc(&d_loss, batch * sizeof(float));
    cudaMemset(d_loss, 0, batch * sizeof(float));  // Initialize loss to 0

    int threads = 256;
    dim3 blocks((size + threads - 1) / threads, batch);
    mse_forward_kernel << <blocks, threads >> > (predictions, targets, d_loss, size);

   // float* loss_host = new float[batch];
   // cudaMemcpy(loss_host, d_loss, batch * sizeof(float), cudaMemcpyDeviceToHost);
   // cudaFree(d_loss);

    return d_loss;  // Make sure caller deletes this memory
}
void MSELoss::backward(const float* predictions, const float* targets, float* grad, int size, int batch) {
    cudaMemset(grad, 0, batch * size * sizeof(float));  // Ensure gradients start at 0
    int threads = 256;
    dim3 blocks((size + threads - 1) / threads, batch);
    mse_backward_kernel << <blocks, threads >> > (predictions, targets, grad, size);
}
