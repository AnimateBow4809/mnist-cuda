#include "LinearLayerQuantised.cuh"
#include <cuda_runtime.h>
#include <cudnn.h>
#include <iostream>
#include <stdexcept>
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



// Constructor
LinearLayerQuantised::LinearLayerQuantised(int batch_size, int in_features, int out_features)
    : batch_size(batch_size), in_features(in_features), out_features(out_features) {

    CUDA_CHECK(cudaMalloc(&d_weight, in_features * out_features * sizeof(Float10)));
    CUDA_CHECK(cudaMalloc(&d_bias, out_features * sizeof(Float10)));
    CUDA_CHECK(cudaMalloc(&d_output, batch_size * out_features * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_input_grad, batch_size * in_features * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weight_grad, out_features * in_features * sizeof(Float10)));
    CUDA_CHECK(cudaMalloc(&d_bias_grad, out_features * sizeof(Float10)));
    cublasCreate(&cublasHandle);

    initWeights(d_weight, in_features, out_features);
    //initWeights(d_bias, 1, out_features);
    CUDA_CHECK(cudaMemset(d_bias, 0, out_features * sizeof(Float10)));

}

__global__ void initSingleWeightf10(Float10* d_weight, int num_elements, float std_dev) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        curandState local_state;
        curand_init(clock64(), idx, 0, &local_state);

        // Generate random float from normal distribution
        float rand_value = curand_normal(&local_state);

        // Scale the random number by the standard deviation
        d_weight[idx] = std_dev * rand_value;
    }
}


void LinearLayerQuantised::initWeights(Float10* d_weight, int input_feat, int output_feat) {
    int totalThreadsNeeded = input_feat * output_feat;

    int threadPerBlock = 256;
    int numberOfBlocks = (totalThreadsNeeded + threadPerBlock - 1) / threadPerBlock;
    float std_dev = sqrt(2.0f / (input_feat + output_feat));

    // Launch kernel
    initSingleWeightf10 << <numberOfBlocks, threadPerBlock >> > (d_weight, totalThreadsNeeded, std_dev);
    CUDA_CHECK(cudaGetLastError());  // Check launch errors
    CUDA_CHECK(cudaDeviceSynchronize());  // Ensure execution completes

}


// Destructor
LinearLayerQuantised::~LinearLayerQuantised() {
    cublasDestroy(cublasHandle);

    CUDA_CHECK(cudaFree(d_weight));
    CUDA_CHECK(cudaFree(d_bias));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_input_grad));
    CUDA_CHECK(cudaFree(d_weight_grad));
    CUDA_CHECK(cudaFree(d_bias_grad));
}


__global__ void linearKernelf10(float* d_A, Float10* d_B, Float10* d_bias, float* d_Y, int B, int in, int out) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < B && col < out) {
        float sum = 0.0f;

        // Matrix multiplication A (B x in) * B (in x out)
        for (int k = 0; k < in; k++) {
            sum += d_A[row * in + k] * d_B[k * out + col];
        }

        // Add bias (1 x out) to each row of the result
        sum += d_bias[col];

        d_Y[row * out + col] = sum;
    }
}

void LinearLayerQuantised::forward(float* d_input) {
    dim3 threadsPerBlock(32, 32);  // Example: 16x16 threads per block
    dim3 numBlocks((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (out_features + threadsPerBlock.y - 1) / threadsPerBlock.y);

    linearKernelf10 << <numBlocks, threadsPerBlock >> > (d_input, d_weight, d_bias, d_output, batch_size, in_features, out_features);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}




__global__ void linearBackwardInputKernelf10(float* d_output_grad, Float10* d_weights, float* d_input_grad, int B, int in, int out) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= B || col >= in) return;

    if (row < B && col < in) {
        float grad = 0.0f;

        // Multiply d_output_grad (B x out) with transpose of weights (out x in)
        for (int k = 0; k < out; k++) {
            grad += d_output_grad[row * out + k] * d_weights[col * out + k];
        }

        d_input_grad[row * in + col] = grad;
    }
}

// grad= output_grad x weights^T ==== (bxout) (inxout)^T
void LinearLayerQuantised::backwardData(float* d_input, float* d_output_grad) {
    dim3 threadsPerBlock(32, 32);  // Example: 32x32 threads per block
    dim3 numBlocks((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (in_features + threadsPerBlock.y - 1) / threadsPerBlock.y);
    linearBackwardInputKernelf10 << <numBlocks, threadsPerBlock >> >
        (d_output_grad, d_weight, d_input_grad, batch_size, in_features, out_features);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    //printf("\nDATA_GRAD:\n");
    //printGpuArray1(d_input_grad, batch_size * in_features, in_features);
}


__global__ void linearBackwardWeightKernelf10(float* d_A, float* d_output_grad, Float10 *d_weight_grad, int B, int in, int out) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= in || col >= out) return;


    if (row < in && col < out) {
        float grad = 0.0f;

        // Compute the gradient dB[row, col] = sum over batch (A^T * dY)
        for (int k = 0; k < B; k++) {
            grad += d_A[k * in + row] * d_output_grad[k * out + col];  // A^T (in x B) * dY (B x out)
        }

        d_weight_grad[row * out + col] = grad/B;
    }
}


void LinearLayerQuantised::backwardWeights(float* d_input, float* d_output_grad) {
    dim3 threadsPerBlock(32, 32);  // Example: 16x16 threads per block
    dim3 numBlocks((in_features + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (out_features + threadsPerBlock.y - 1) / threadsPerBlock.y);

    linearBackwardWeightKernelf10 << <numBlocks, threadsPerBlock >> >
        (d_input, d_output_grad, d_weight_grad, batch_size, in_features, out_features);
    cudaDeviceSynchronize();

    CUDA_CHECK(cudaGetLastError());  // Check launch errors
    CUDA_CHECK(cudaDeviceSynchronize());  // Ensure execution completes
    /*
    printf("Weight Grad:\n");
    printGpuArray1(d_weight_grad, out_features * in_features, in_features);*/
}

__global__ void computeBiasGradientsf10(float* d_output_grad, Float10* d_bias_grad, int batch_size, int output_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= output_features) return;

    float grad = 0.0f;
    // Sum the gradients over the batch for this output feature
    for (int i = 0; i < batch_size; i++) {
        int index = i * output_features + idx;
        grad += d_output_grad[index];  // Gradient of output at (i, idx)
    }

    d_bias_grad[idx] = grad/ batch_size;  // The summed gradient for this bias
}


// Compute bias gradients: sum over batch
void LinearLayerQuantised::backwardBias(float* d_output_grad) {
    // Loop over each output feature (bias term corresponds to each output feature)
    int threads = 256;
    int blocks = (out_features + threads - 1) / threads;

    // Kernel to compute the gradient w.r.t. bias
    computeBiasGradientsf10 << <blocks, threads >> > (d_output_grad, d_bias_grad, batch_size, out_features);
    CUDA_CHECK(cudaDeviceSynchronize());
}



__global__ void changeFormat(Float10* src, float* dst, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size)return;
    dst[idx] = src[idx];

}

__global__ void changeFormat(float* src,Float10* dst , int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size)return;
    dst[idx] = src[idx];
}

__global__ void updateWeightKernelf10(Float10* d_A, Float10* d_B, float coeficient, int numberOfElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numberOfElements)return;
    d_A[idx] = d_A[idx] + d_B[idx] * coeficient;
}


// Update weights and biases using SGD
void LinearLayerQuantised::updateWeights(float learning_rate) {
    float alpha = -learning_rate;

    int wgrad_size = out_features * in_features;
    int bgrad_size = out_features;

    int threadsPerBlock = 256;
    int numBlocksForWeights = (wgrad_size + threadsPerBlock - 1) / threadsPerBlock;
    int numBlocksForBias = (bgrad_size + threadsPerBlock - 1) / threadsPerBlock;

    updateWeightKernelf10 << <numBlocksForWeights, threadsPerBlock >> > (d_weight, d_weight_grad, alpha, wgrad_size);
    updateWeightKernelf10 << <numBlocksForBias, threadsPerBlock >> > (d_bias, d_bias_grad, alpha, bgrad_size);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void LinearLayerQuantised::backward(float* d_input, float* d_output_grad, float lr) {
    backwardData(d_input, d_output_grad);
    CUDA_CHECK(cudaDeviceSynchronize());
    backwardWeights(d_input, d_output_grad);
    backwardBias(d_output_grad);
    updateWeights(lr);
}


float* LinearLayerQuantised::getOutput(int* outputSize) {
    if (outputSize)
    {
        *outputSize = batch_size * out_features * sizeof(float);
    }
    return d_output;
}

float* LinearLayerQuantised::getInputGrad(int* inputGradSize) {
    if (inputGradSize)
    {
        *inputGradSize = batch_size * in_features * sizeof(float);
    }
    return d_input_grad;
}

float* LinearLayerQuantised::getAllWeights(int* outputSize) {
    *outputSize = (in_features * out_features + out_features);
    float* h_temp = (float*)malloc((in_features * out_features + out_features) * sizeof(float));

    Float10* hf_temp = (Float10*)malloc((in_features * out_features + out_features) * sizeof(Float10));
    CUDA_CHECK(cudaMemcpy(hf_temp, d_weight, in_features * out_features * sizeof(Float10), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&hf_temp[in_features * out_features], d_bias, out_features * sizeof(Float10), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < (in_features * out_features + out_features); i++)
    {
        h_temp[i] = hf_temp[i];
    }
    free(hf_temp);
    return h_temp;
}