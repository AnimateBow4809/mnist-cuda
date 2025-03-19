#include "Utils.cuh"

__global__ void clipGradients(float* gradients, int size, float clip_value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        gradients[idx] = fminf(fmaxf(gradients[idx], -clip_value), clip_value);
    }
}

__global__ void clipGradientsL2(float* grads, int size, float clip_threshold) {
    // Compute L2 norm of the gradients
    __shared__ float norm;
    if (threadIdx.x == 0) norm = 0.0f;
    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Sum squared gradients in parallel
    float val = (i < size) ? grads[i] * grads[i] : 0.0f;
    atomicAdd(&norm, val);
    __syncthreads();

    // Compute square root of the norm
    if (threadIdx.x == 0) norm = sqrtf(norm);
    __syncthreads();

    // Apply gradient clipping if norm exceeds threshold
    if (norm > clip_threshold && i < size) {
        grads[i] *= clip_threshold / norm;
    }
}
