#include "softmaxLayer.cuh"
#include <cuda_runtime.h>
#include <cmath>

// Kernel for forward pass
__global__ void softmaxForwardKernel(float* d_input, float* d_output, int batch, int num_classes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch) return;

    float* input = d_input + idx * num_classes;
    float* output = d_output + idx * num_classes;

    float max_val = input[0];
    for (int i = 1; i < num_classes; i++) {
        if (input[i] > max_val) max_val = input[i];
    }

    float sum_exp = 0.0f;
    for (int i = 0; i < num_classes; i++) {
        output[i] = expf(input[i] - max_val);
        sum_exp += output[i];
    }

    for (int i = 0; i < num_classes; i++) {
        output[i] /= sum_exp;
    }
}

// Kernel for backward pass
__global__ void softmaxBackwardKernel(float* d_output_grad, float* d_softmax, float* d_input_grad, int batch, int num_classes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch) return;

    float* grad_out = d_output_grad + idx * num_classes;
    float* softmax = d_softmax + idx * num_classes;
    float* grad_in = d_input_grad + idx * num_classes;

    for (int i = 0; i < num_classes; i++) {
        float grad = 0.0f;
        for (int j = 0; j < num_classes; j++) {
            float delta = (i == j) ? 1.0f : 0.0f;
            grad += grad_out[j] * softmax[i] * (delta - softmax[j]);
        }
        grad_in[i] = grad;
    }
}

// Constructor
SoftmaxLayer::SoftmaxLayer(int batch, int channels, int height, int width) {
    this->batch = batch;
    this->channels = channels;
    this->height = height;
    this->width = width;
    this->num_classes = channels * height * width;
    this->num_elements = batch * num_classes;

    cudaMalloc(&d_output, num_elements * sizeof(float));
    cudaMalloc(&d_input_grad, num_elements * sizeof(float));
}

// Forward pass
void SoftmaxLayer::forward(float* d_input) {
    int threads = 256;
    int blocks = (batch + threads - 1) / threads;
    softmaxForwardKernel << <blocks, threads >> > (d_input, d_output, batch, num_classes);
    cudaDeviceSynchronize();
}

// Backward pass
void SoftmaxLayer::backward(float* d_input, float* d_output_grad, float lr) {
    int threads = 256;
    int blocks = (batch + threads - 1) / threads;
    softmaxBackwardKernel << <blocks, threads >> > (d_output_grad, d_output, d_input_grad, batch, num_classes);
    cudaDeviceSynchronize();
}

// Get output
float* SoftmaxLayer::getOutput(int* outputSize) {
    if (outputSize) *outputSize = num_elements*sizeof(float);
    return d_output;
}

// Get input gradient
float* SoftmaxLayer::getInputGrad(int* inputGradSize) {
    if (inputGradSize) *inputGradSize = num_elements*sizeof(float);
    return d_input_grad;
}

// Destructor
SoftmaxLayer::~SoftmaxLayer() {
    cudaFree(d_output);
    cudaFree(d_input_grad);
}

float* SoftmaxLayer::getAllWeights(int* outputSize) {
    *outputSize = 0;
    return nullptr;
}
