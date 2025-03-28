#include "LinearLayer.cuh"
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

    CUDA_CHECK(cudaMalloc(&d_weight, in_features* out_features * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bias, out_features * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, batch_size * out_features * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_input_grad, batch_size * in_features * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weight_grad, out_features * in_features * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bias_grad, out_features * sizeof(float)));
    cublasCreate(&cublasHandle);

    initWeights(d_weight, in_features, out_features);
    //initWeights(d_bias, 1, out_features);
    CUDA_CHECK(cudaMemset(d_bias, 0, out_features * sizeof(float)));

}

__global__ void initSingleWeight(float* d_weight, int num_elements, float std_dev) {
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


void LinearLayer::initWeights(float* d_weight, int input_feat, int output_feat) {
    int totalThreadsNeeded = input_feat * output_feat;

    int threadPerBlock = 256;
    int numberOfBlocks = (totalThreadsNeeded + threadPerBlock - 1) / threadPerBlock;
    float std_dev = sqrt(2.0f / (input_feat+output_feat));

    // Launch kernel
    initSingleWeight << <numberOfBlocks, threadPerBlock >> > (d_weight, totalThreadsNeeded, std_dev);
    CUDA_CHECK(cudaGetLastError());  // Check launch errors
    CUDA_CHECK(cudaDeviceSynchronize());  // Ensure execution completes

}


// Destructor
LinearLayer::~LinearLayer() {
    cublasDestroy(cublasHandle);

    CUDA_CHECK(cudaFree(d_weight));
    CUDA_CHECK(cudaFree(d_bias));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_input_grad));
    CUDA_CHECK(cudaFree(d_weight_grad));
    CUDA_CHECK(cudaFree(d_bias_grad));
}


__global__ void linearKernel(float* d_A, float* d_B, float* d_bias, float* d_Y, int B, int in, int out) {
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

void LinearLayer::forward(float* d_input) {
    dim3 threadsPerBlock(32, 32);  // Example: 16x16 threads per block
    dim3 numBlocks((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (out_features + threadsPerBlock.y - 1) / threadsPerBlock.y);

    linearKernel << <numBlocks, threadsPerBlock >> > (d_input, d_weight, d_bias, d_output, batch_size, in_features, out_features);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}




__global__ void linearBackwardInputKernel(float* d_output_grad, float* d_weights, float* d_input_grad, int B, int in, int out) {
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
void LinearLayer::backwardData(float* d_input, float* d_output_grad) {
    dim3 threadsPerBlock(32, 32);  // Example: 32x32 threads per block
    dim3 numBlocks((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (in_features + threadsPerBlock.y - 1) / threadsPerBlock.y);
    linearBackwardInputKernel << <numBlocks, threadsPerBlock >> > 
        (d_output_grad, d_weight, d_input_grad, batch_size, in_features, out_features);

    CUDA_CHECK(cudaGetLastError()); 
    CUDA_CHECK(cudaDeviceSynchronize());  

    //printf("\nDATA_GRAD:\n");
    //printGpuArray1(d_input_grad, batch_size * in_features, in_features);
}


__global__ void linearBackwardWeightKernel(float* d_A, float* d_output_grad, float* d_weight_grad, int B, int in, int out) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= in || col >= out) return;


    if (row < in && col < out) {
        float grad = 0.0f;

        // Compute the gradient dB[row, col] = sum over batch (A^T * dY)
        for (int k = 0; k < B; k++) {
            grad += d_A[k * in + row] * d_output_grad[k * out + col];  // A^T (in x B) * dY (B x out)
        }

        d_weight_grad[row * out + col] = grad;
    }
}


void LinearLayer::backwardWeights(float* d_input, float* d_output_grad) {
    dim3 threadsPerBlock(32, 32);  // Example: 16x16 threads per block
    dim3 numBlocks((in_features + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (out_features + threadsPerBlock.y - 1) / threadsPerBlock.y);

    linearBackwardWeightKernel << <numBlocks, threadsPerBlock >> > 
        (d_input, d_output_grad, d_weight_grad, batch_size, in_features, out_features);
    cudaDeviceSynchronize();

    CUDA_CHECK(cudaGetLastError());  // Check launch errors
    CUDA_CHECK(cudaDeviceSynchronize());  // Ensure execution completes
    /*
    printf("Weight Grad:\n");
    printGpuArray1(d_weight_grad, out_features * in_features, in_features);*/
}

__global__ void computeBiasGradients(float* d_output_grad, float* d_bias_grad, int batch_size, int output_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= output_features) return;

    float grad = 0.0f;
    // Sum the gradients over the batch for this output feature
    for (int i = 0; i < batch_size; i++) {
        int index = i * output_features + idx;
        grad += d_output_grad[index];  // Gradient of output at (i, idx)
    }

    d_bias_grad[idx] = grad;  // The summed gradient for this bias
}


// Compute bias gradients: sum over batch
void LinearLayer::backwardBias(float* d_output_grad) {
    // Loop over each output feature (bias term corresponds to each output feature)
    int threads = 256;
    int blocks = (out_features + threads - 1) / threads;

    // Kernel to compute the gradient w.r.t. bias
    computeBiasGradients << <blocks, threads >> > (d_output_grad, d_bias_grad, batch_size, out_features);
    CUDA_CHECK(cudaDeviceSynchronize());
}

__global__ void updateWeightKernel(float* d_A,float *d_B,float coeficient,int numberOfElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numberOfElements)return;
    d_A[idx] = d_A[idx] - d_B[idx] * coeficient;
}



// Update weights and biases using SGD
void LinearLayer::updateWeights(float learning_rate) {


    float alpha = -learning_rate;

    int wgrad_size = out_features * in_features;
    int bgrad_size = out_features;

    int threadsPerBlock = 256;

    // Normalize gradients by batch size
    float scale = 1.0f / batch_size;
    cublasSscal(cublasHandle, wgrad_size, &scale, d_weight_grad, 1);
    cublasSscal(cublasHandle, bgrad_size, &scale, d_bias_grad, 1);
    CUDA_CHECK(cudaGetLastError());  // Check launch errors
    CUDA_CHECK(cudaDeviceSynchronize());  // Ensure execution completes

    // Clip gradients (optional)
    //float clip_threshold = 5.0f;
    //clipGradients << <(wgrad_size + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> > (d_weight_grad, wgrad_size, clip_threshold);
    //CUDA_CHECK(cudaGetLastError());  // Check launch errors
    //CUDA_CHECK(cudaDeviceSynchronize());  // Ensure execution completes

    //clipGradients << <(bgrad_size + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> > (d_bias_grad, bgrad_size, clip_threshold);
    //CUDA_CHECK(cudaGetLastError());  // Check launch errors
    //CUDA_CHECK(cudaDeviceSynchronize());  // Ensure execution completes

    // Weight update: W -= lr * grad_W
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
    //printf("\n\n\n\n");
    //printGpuArray1(d_weight_grad, out_features * in_features, 10);

  
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

float* LinearLayer::getAllWeights(int* outputSize) {
    *outputSize = (in_features * out_features + out_features);
    float* h_temp = (float*)malloc((in_features * out_features + out_features) * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_temp,d_weight,in_features * out_features * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_temp[in_features * out_features], d_bias, out_features * sizeof(float), cudaMemcpyDeviceToHost));
    return h_temp;
}