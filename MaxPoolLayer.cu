#include "MaxPoolLayer.cuh"
#include <iostream>

MaxPoolLayer::MaxPoolLayer(int batch, int channels, int height, int width, int pool_size)
    : batch(batch), channels(channels), height(height), width(width), pool_size(pool_size) {

    pooled_height = height / pool_size;
    pooled_width = width / pool_size;

    cudaMalloc(&d_output, batch * channels * pooled_height * pooled_width * sizeof(float));
    cudaMalloc(&d_input_grad, batch * channels * height * width * sizeof(float));

    cudnnCreate(&cudnn);

    // Define input and output tensor descriptors
    cudnnCreateTensorDescriptor(&inputDesc);
    cudnnCreateTensorDescriptor(&outputDesc);
    cudnnCreatePoolingDescriptor(&poolingDesc);

    cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, channels, height, width);
    cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, channels, pooled_height, pooled_width);

    cudnnSetPooling2dDescriptor(poolingDesc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN, pool_size, pool_size, 0, 0, pool_size, pool_size);
}

// Destructor
MaxPoolLayer::~MaxPoolLayer() {
    cudaFree(d_output);
    cudaFree(d_input_grad);

    cudnnDestroyPoolingDescriptor(poolingDesc);
    cudnnDestroyTensorDescriptor(inputDesc);
    cudnnDestroyTensorDescriptor(outputDesc);
    cudnnDestroy(cudnn);
}

// Forward Pass
void MaxPoolLayer::forward(float* d_input) {
    float alpha = 1.0f, beta = 0.0f;
    cudnnPoolingForward(cudnn, poolingDesc, &alpha, inputDesc, d_input, &beta, outputDesc, d_output);
}

// Backward Pass
void MaxPoolLayer::backward(float* d_input, float* d_output_grad, float lr) {
    float alpha = 1.0f, beta = 0.0f;
    cudnnPoolingBackward(cudnn, poolingDesc, &alpha, outputDesc, d_output, outputDesc, d_output_grad,
        inputDesc, d_input, &beta, inputDesc, d_input_grad);
}

// Get Output
float* MaxPoolLayer::getOutput(int* outputSize) {
    if (outputSize) *outputSize = batch * channels * pooled_height * pooled_width;
    return d_output;
}

// Get Input Gradient
float* MaxPoolLayer::getInputGrad(int* inputGradSize) {
    if (inputGradSize) *inputGradSize = batch * channels * height * width;
    return d_input_grad;
}
