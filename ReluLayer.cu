#include "ReluLayer.cuh"

// CUDA Kernel: Forward Pass (ReLU)
__global__ void reluForwardKernel(float* input, float* output, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        output[idx] = fmaxf(input[idx], 0.0f);  // ReLU: max(0, x)
    }
}

// CUDA Kernel: Backward Pass (ReLU Derivative)
__global__ void reluBackwardKernel(float* input, float* grad_output, float* grad_input, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        grad_input[idx] = (input[idx] > 0) ? grad_output[idx] : 0.0f;  // dReLU/dx
    }
}

// Constructor: Allocate memory
ReLULayer::ReLULayer(int batch, int channels, int height, int width)
    : batch(batch), channels(channels), height(height), width(width) {

    num_elements = batch * channels * height * width;

    cudaMalloc(&d_output, num_elements * sizeof(float));
    cudaMalloc(&d_input_grad, num_elements * sizeof(float));
}

// Destructor: Free memory
ReLULayer::~ReLULayer() {
    cudaFree(d_output);
    cudaFree(d_input_grad);
}

// Forward Pass
void ReLULayer::forward(float* d_input) {
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;

    reluForwardKernel << <blocks, threads >> > (d_input, d_output, num_elements);
}

// Backward Pass
void ReLULayer::backward(float* d_input, float* d_output_grad,float lr) {
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;

    reluBackwardKernel << <blocks, threads >> > (d_input, d_output_grad, d_input_grad, num_elements);
}

float* ReLULayer::getOutput(int* outputSize) {
    if (outputSize)
    {
        *outputSize = num_elements * sizeof(float);
    }
    return d_output;
}

float* ReLULayer::getInputGrad(int* inputGradSize) {
    if (inputGradSize)
    {
        *inputGradSize = num_elements * sizeof(float);
    }
    return d_input_grad;
}

