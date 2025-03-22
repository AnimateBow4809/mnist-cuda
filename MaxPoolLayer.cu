#include "MaxPoolLayer.cuh"
#include <iostream>

#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#define CUDNN_CHECK(call) \
do { \
    cudnnStatus_t status = call; \
    if (status != CUDNN_STATUS_SUCCESS) { \
        std::cerr << "cuDNN error at " << __FILE__ << ":" << __LINE__ << " - " << cudnnGetErrorString(status) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while (0)

MaxPoolLayer::MaxPoolLayer(int batch, int channels, int height, int width, int pool_size)
    : batch(batch), channels(channels), height(height), width(width), pool_size(pool_size) {

    pooled_height = (height - pool_size) / pool_size + 1;
    pooled_width = (width - pool_size) / pool_size + 1;

    CUDA_CHECK(cudaMalloc(&d_output, batch * channels * pooled_height * pooled_width * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_input_grad, batch * channels * height * width * sizeof(float)));

    CUDNN_CHECK(cudnnCreate(&cudnn));

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&inputDesc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&outputDesc));
    CUDNN_CHECK(cudnnCreatePoolingDescriptor(&poolingDesc));

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, channels, height, width));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, channels, pooled_height, pooled_width));

    int padding = 0;
    int stride = pool_size;
    CUDNN_CHECK(cudnnSetPooling2dDescriptor(poolingDesc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN, pool_size, pool_size, padding, padding, stride, stride));
}

// Destructor
MaxPoolLayer::~MaxPoolLayer() {
    if (d_output) cudaFree(d_output);
    if (d_input_grad) cudaFree(d_input_grad);
    if (poolingDesc) cudnnDestroyPoolingDescriptor(poolingDesc);
    if (inputDesc) cudnnDestroyTensorDescriptor(inputDesc);
    if (outputDesc) cudnnDestroyTensorDescriptor(outputDesc);
    if (cudnn) cudnnDestroy(cudnn);
}

// Forward Pass
void MaxPoolLayer::forward(float* d_input) {
    float alpha = 1.0f, beta = 0.0f;
    CUDNN_CHECK(cudnnPoolingForward(cudnn, poolingDesc, &alpha, inputDesc, d_input, &beta, outputDesc, d_output));
    cudaDeviceSynchronize();
}

// Backward Pass
void MaxPoolLayer::backward(float* d_input, float* d_output_grad, float lr) {
    float alpha = 1.0f, beta = 0.0f;
    CUDA_CHECK(cudaMemset(d_input_grad, 0, batch * channels * height * width * sizeof(float)));
    CUDNN_CHECK(cudnnPoolingBackward(cudnn, poolingDesc, &alpha, outputDesc, d_output, outputDesc, d_output_grad,
        inputDesc, d_input, &beta, inputDesc, d_input_grad));
    cudaDeviceSynchronize();

}

// Get Output
float* MaxPoolLayer::getOutput(int* outputSize) {
    if (outputSize) *outputSize = batch * channels * pooled_height * pooled_width * sizeof(float);
    return d_output;
}

// Get Input Gradient
float* MaxPoolLayer::getInputGrad(int* inputGradSize) {
    if (inputGradSize) *inputGradSize = batch * channels * height * width * sizeof(float);
    return d_input_grad;
}
