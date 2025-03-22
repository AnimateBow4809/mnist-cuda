#include "ReluLayer.cuh"

#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#define CUDNN_CHECK(call) \
do { \
    cudnnStatus_t err = call; \
    if (err != CUDNN_STATUS_SUCCESS) { \
        std::cerr << "cuDNN Error: " << cudnnGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#define CUBLAS_CHECK(call) \
do { \
    cublasStatus_t err = call; \
    if (err != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS Error: " << err << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while (0)


// CUDA Kernel: Forward Pass (ReLU)
__global__ void reluForwardKernel(float* input, float* output, int num_elements, float leak) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        output[idx] = (input[idx] > 0) ? input[idx] : leak * input[idx];  // Leaky ReLU: x if x > 0, else leak * x
    }
}


// CUDA Kernel: Backward Pass (ReLU Derivative)
__global__ void reluBackwardKernel(float* input, float* grad_output, float* grad_input, int num_elements, float leak) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        grad_input[idx] = (input[idx] > 0) ? grad_output[idx] : leak * grad_output[idx];  // dLReLU/dx
    }
}

// Constructor: Allocate memory
ReLULayer::ReLULayer(int batch, int channels, int height, int width,float leak)
    : batch(batch), channels(channels), height(height), width(width) {

    num_elements = batch * channels * height * width;
    this->leak = leak;

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

    reluForwardKernel << <blocks, threads >> > (d_input, d_output, num_elements,leak);
    CUDA_CHECK(cudaGetLastError());  // Check launch errors
    CUDA_CHECK(cudaDeviceSynchronize());  // Ensure execution completes

}

// Backward Pass
void ReLULayer::backward(float* d_input, float* d_output_grad,float lr) {
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;

    reluBackwardKernel << <blocks, threads >> > (d_input, d_output_grad, d_input_grad, num_elements,leak);
    CUDA_CHECK(cudaGetLastError());  // Check launch errors
    CUDA_CHECK(cudaDeviceSynchronize());  // Ensure execution completes

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

