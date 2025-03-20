#include "LinearLayer.cuh"
#include <cuda_runtime.h>
#include <cudnn.h>
#include <iostream>
#include <stdexcept>
#include <cublas_v2.h>
#include <cuda_runtime.h>  // Core CUDA runtime API
#include <device_launch_parameters.h>  // Required for kernel launch parameters
#include <curand_kernel.h>
#include "Utils.cuh"

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

float* printGpuArray1(float* d_in, int size, int newLine) {
    float* h_temp = (float*)malloc(size * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_temp, d_in, size * sizeof(float), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < size; i++) {
        printf("%f ", h_temp[i]);
        if ((i + 1) % newLine == 0) {
            printf("\n");
        }
    }
    return h_temp;
}


// Constructor
LinearLayer::LinearLayer(int batch_size, int in_features, int out_features)
    : batch_size(batch_size), in_features(in_features), out_features(out_features) {

    CUDNN_CHECK(cudnnCreate(&handle));

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, in_features, 1, 1));

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, out_features, 1, 1));

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&bias_desc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, out_features, 1, 1));

    CUDA_CHECK(cudaMalloc(&d_weight, out_features * in_features * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bias, out_features * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, batch_size * out_features * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_input_grad, batch_size * in_features * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weight_grad, out_features * in_features * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bias_grad, out_features * sizeof(float)));

    initWeights(d_weight, in_features, out_features);
    initWeights(d_bias, 1, out_features);
}

__global__ void initSingleWeight(float* d_weight, int num_elements, float std_dev) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        curandState local_state;
        curand_init(1234, idx, 0, &local_state); // Seed the random number generator

        // Generate random float from normal distribution
        float rand_value = curand_normal(&local_state);

        // Scale the random number by the standard deviation
        d_weight[idx] = std_dev * rand_value;
    }
}


void LinearLayer::initWeights(float* d_weight, int input_feat, int output_feat) {
    int totalThreadsNeeded = input_feat * output_feat;

    int threadPerBlock = 256;
    int numberOfBlocks = (totalThreadsNeeded + threadPerBlock - 1) / threadPerBlock;
    float std_dev = sqrt(2.0f / (input_feat + output_feat));

    // Launch kernel
    initSingleWeight << <numberOfBlocks, threadPerBlock >> > (d_weight, totalThreadsNeeded, std_dev);
    CUDA_CHECK(cudaGetLastError());  // Check launch errors
    CUDA_CHECK(cudaDeviceSynchronize());  // Ensure execution completes

}


// Destructor
LinearLayer::~LinearLayer() {
    CUDNN_CHECK(cudnnDestroy(handle));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(output_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(bias_desc));

    CUDA_CHECK(cudaFree(d_weight));
    CUDA_CHECK(cudaFree(d_bias));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_input_grad));
    CUDA_CHECK(cudaFree(d_weight_grad));
    CUDA_CHECK(cudaFree(d_bias_grad));
}

__global__ void matmul_kernel(float* A, float* B, float* C, int A_rows, int A_cols, int B_cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < A_rows && col < B_cols) {
        float value = 0;
        for (int k = 0; k < A_cols; ++k) {
            value += A[row * A_cols + k] * B[k * B_cols + col];  // Row-major access
        }
        C[row * B_cols + col] = value;
    }
}

__global__ void matmul_1xN_MxN_transposed(float* A, float* B, float* C, int N, int M) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < M) {
        float value = 0;
        // B is now stored as transposed, so accessing B[k * M + col] is efficient
        for (int k = 0; k < N; ++k) {
            value += A[k] * B[col * N + k];  // Transposed access of B
        }
        C[col] = value;
    }
}




// Forward pass: y = xW^T + b
void LinearLayer::forward(float* d_input) {
    const float alpha = 1.0f;
    //const float beta = 0.0f;
    // GEMM-like operation: output = input * W^T   //weight outxn   input 1xn
    dim3 block(256, 1, 1); // 256 threads per block
    dim3 grid((out_features + block.x - 1) / block.x, 1, 1); // Number of blocks needed to cover all M columns

    for (size_t i = 0; i < batch_size; i++) {
        // Adjust linear indexing based on the memory layout of d_input and d_output
        matmul_1xN_MxN_transposed << <grid, block >> > (
            &d_input[i * in_features],  // Access the i-th input in the batch
            d_weight,                    // Weight matrix (transposed)
            &d_output[i * out_features],// Access the i-th output in the batch
            in_features, out_features
            );
        CUDA_CHECK(cudaGetLastError());  // Check launch errors
        CUDA_CHECK(cudaDeviceSynchronize());  // Ensure execution completes
    }

    cudaDeviceSynchronize();

    //printf("\nresult before bias addition:\n");
    //printGpuArray1(d_output, out_features, out_features);
    // Add bias using cuDNN (broadcasted add)
    (cudnnAddTensor(handle,
        &alpha, bias_desc, d_bias,
        &alpha, output_desc, d_output));
    cudaDeviceSynchronize();


}



__global__ void matmul_BxN_NxM(float* A, float* B, float* C, int N, int M) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int bIdx = blockIdx.y;  // Each batch gets a separate row of C
    if (col < M) {
        float value = 0;
        for (int k = 0; k < N; k++) {
            value += A[bIdx * N + k] * B[col + k * M];  // Transposed access of B
        }
        C[bIdx * M + col] = value;  // Write to correct batch row in C
    }
}


// Backward pass: grad_input = grad_output=(Bxout) * W(outxin)    
void LinearLayer::backwardData(float* d_input, float* d_output_grad) {
    //const float alpha = 1.0f;
    //const float beta = 0.0f;

    dim3 block(256, 1, 1);
    dim3 grid((out_features + block.x - 1) / block.x, batch_size); // Use batch_size in the y-dimension

    matmul_BxN_NxM << <grid, block >> > (
        d_output_grad,
        d_weight,
        d_input_grad,
        out_features, in_features
        );

    CUDA_CHECK(cudaGetLastError());  // Check launch errors
    CUDA_CHECK(cudaDeviceSynchronize());  // Ensure execution completes


    //printf("\nDATA_GRAD:\n");
    //printGpuArray1(d_input_grad, batch_size * in_features, in_features);
}


__global__ void matmul_BxO_BxM(float* A, float* B, float* C, int Batch, int M, int O) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // Column index of output (I)
    int row = blockIdx.y;  // Row index of output (O)

    if (col < M) {
        float value = 0;
        for (int k = 0; k < Batch; k++) {
            value += A[k * O + row] * B[k * M + col];  // Properly transpose A on-the-fly
        }
        C[row * M + col] = value;
    }
}

void LinearLayer::backwardWeights(float* d_input, float* d_output_grad) {
    dim3 block(256, 1, 1);
    dim3 grid((in_features + block.x - 1) / block.x, out_features);

    matmul_BxO_BxM << <grid, block >> > (
        d_output_grad,  // (B × O), will be transposed inside kernel
        d_input,        // (B × I)
        d_weight_grad,  // (O × I)
        batch_size,
        in_features,
        out_features
        );

    CUDA_CHECK(cudaGetLastError());  // Check launch errors
    CUDA_CHECK(cudaDeviceSynchronize());  // Ensure execution completes
    /*
    printf("Weight Grad:\n");
    printGpuArray1(d_weight_grad, out_features * in_features, in_features);*/
}

// Compute bias gradients: sum over batch
void LinearLayer::backwardBias(float* d_output_grad) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    (cudnnConvolutionBackwardBias(handle,
        &alpha,
        output_desc, d_output_grad,
        &beta,
        bias_desc, d_bias_grad));
}

// Update weights and biases using SGD
void LinearLayer::updateWeights(float learning_rate) {
    const float alpha = -learning_rate;

    int wgrad_size = out_features*in_features; // Number of gradients
    float clip_threshold = 5.0f; // Adjust as needed
    clipGradients << <(wgrad_size + 255) / 256, 256 >> > (d_weight_grad, wgrad_size, clip_threshold);

    int bgrad_size = out_features;
    clipGradients << <(bgrad_size + 255) / 256, 256 >> > (d_bias_grad, bgrad_size, clip_threshold);

    cudaDeviceSynchronize();



    // Weight update: W -= lr * grad_W
    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);
    cublasSaxpy(cublasHandle,
        out_features * in_features,
        &alpha,
        d_weight_grad, 1,
        d_weight, 1);

    // Bias update: b -= lr * grad_b
    cublasSaxpy(cublasHandle,
        out_features,
        &alpha,
        d_bias_grad, 1,
        d_bias, 1);

    cublasDestroy(cublasHandle);
}

void LinearLayer::backward(float* d_input, float* d_output_grad, float lr) {
    backwardData(d_input, d_output_grad);
    CUDA_CHECK(cudaDeviceSynchronize());
    backwardWeights(d_input, d_output_grad);
    backwardBias(d_output_grad);
    updateWeights(lr);
}


float* LinearLayer::getOutput(int* outputSize) {
    if (outputSize)
    {
        *outputSize = batch_size * out_features * sizeof(float);
    }
    return d_output;
}

float* LinearLayer::getInputGrad(int* inputGradSize) {
    if (inputGradSize)
    {
        *inputGradSize = batch_size * in_features * sizeof(float);
    }
    return d_input_grad;
}