#include "ConvLayer2D.cuh"
#include <cublas_v2.h>
#include <random>
#include "Utils.cuh"

#define CUDA_CHECK(call) \
do { \
cudaError_t err = call; \
if (err != cudaSuccess) { \
    printf("CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    exit(EXIT_FAILURE); \
} \
} while (0)

#define CUDNN_CHECK(call) \
do { \
cudnnStatus_t status = call; \
if (status != CUDNN_STATUS_SUCCESS) { \
    printf("cuDNN error in %s at line %d: %s\n", __FILE__, __LINE__, cudnnGetErrorString(status)); \
    exit(EXIT_FAILURE); \
} \
} while (0)


ConvLayer2D::ConvLayer2D(int batch, int in_channels, int in_height, int in_width,
    int out_channels, int kernel_size, int stride, int padding)
    : batch(batch), in_channels(in_channels), in_height(in_height), in_width(in_width),
    out_channels(out_channels), kernel_size(kernel_size), stride(stride), padding(padding) {

    CUDNN_CHECK(cudnnCreate(&handle));

    // Input Descriptor
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        batch, in_channels, in_height, in_width));

    // Filter Descriptor
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
        out_channels, in_channels, kernel_size, kernel_size));

    // Convolution Descriptor
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc, padding, padding, stride, stride, 1, 1,
        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    int batch_t = batch;
    int out_channels_t = out_channels;
    // Output Shape Calculation
    CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(conv_desc, input_desc, filter_desc,
        &batch_t, &out_channels_t, &out_height, &out_width));

    // Output Descriptor
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        batch, out_channels, out_height, out_width));

    // Bias Descriptor
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&bias_desc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, out_channels, 1, 1));

    // Allocate Memory for filter, output, and bias
    CUDA_CHECK(cudaMalloc(&d_filter, out_channels * in_channels * kernel_size * kernel_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bias, out_channels * sizeof(float)));  // New: Bias
    CUDA_CHECK(cudaMalloc(&d_output, batch * out_channels * out_height * out_width * sizeof(float)));

    // Initialize weights and bias (random init)
    std::vector<float> host_filter(out_channels * in_channels * kernel_size * kernel_size + out_channels);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 0.1f);

    for (auto& x : host_filter) x = dist(gen);

    CUDA_CHECK(cudaMemcpy(d_filter, host_filter.data(), (host_filter.size() - out_channels) * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias, &host_filter.data()[host_filter.size() - out_channels],
        out_channels * sizeof(float), cudaMemcpyHostToDevice));

    // Gradients
    CUDA_CHECK(cudaMalloc(&d_input_grad, batch * in_channels * in_height * in_width * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_filter_grad, out_channels * in_channels * kernel_size * kernel_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bias_grad, out_channels * sizeof(float)));
}


void ConvLayer2D::forward(float* d_input) {
    float alpha = 1.0f, beta = 0.0f;
    size_t workspace_size = 0;
    cudnnConvolutionFwdAlgo_t algo;

    int returnedAlgoCount;
    cudnnConvolutionFwdAlgoPerf_t perfResults;
    CUDNN_CHECK(cudnnFindConvolutionForwardAlgorithm(handle, input_desc, filter_desc, conv_desc,
        output_desc, 1, &returnedAlgoCount, &perfResults));
    algo = perfResults.algo;

    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(handle, input_desc, filter_desc, conv_desc,
        output_desc, algo, &workspace_size));
    void* d_workspace;
    CUDA_CHECK(cudaMalloc(&d_workspace, workspace_size));

    CUDNN_CHECK(cudnnConvolutionForward(handle, &alpha, input_desc, d_input, filter_desc, d_filter,
        conv_desc, algo, d_workspace, workspace_size, &beta, output_desc, d_output));

    CUDNN_CHECK(cudnnAddTensor(handle, &alpha, bias_desc, d_bias, &alpha, output_desc, d_output));

    CUDA_CHECK(cudaFree(d_workspace));
}

void ConvLayer2D::backwardData(float* d_input, float* d_output_grad) {

    CUDA_CHECK(cudaMemset(d_input_grad, 0, batch * in_channels * in_height * in_width * sizeof(float)));

    float alpha = 1.0f, beta = 0.0f;
    cudnnConvolutionBwdDataAlgo_t algo;
    size_t workspace_size = 0;

    int returnedAlgoCount;
    cudnnConvolutionBwdDataAlgoPerf_t perfResults;

    // Use cudnnGetConvolutionBackwardDataAlgorithm_v7 instead if using newer cuDNN versions
    CUDNN_CHECK(cudnnFindConvolutionBackwardDataAlgorithm(handle, filter_desc, output_desc, conv_desc,
        input_desc, 1, &returnedAlgoCount, &perfResults));

    algo = perfResults.algo;

    CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(handle, filter_desc, output_desc, conv_desc,
        input_desc, algo, &workspace_size));

    void* d_workspace = nullptr;
    if (workspace_size > 0) { // Allocate workspace only if needed
        CUDA_CHECK(cudaMalloc(&d_workspace, workspace_size));
    }

    CUDNN_CHECK(cudnnConvolutionBackwardData(handle, &alpha, filter_desc, d_filter, output_desc,
        d_output_grad, conv_desc, algo, d_workspace, workspace_size,
        &beta, input_desc, d_input_grad));

    CUDA_CHECK(cudaGetLastError());  // Check launch errors
    CUDA_CHECK(cudaDeviceSynchronize());  // Ensure execution completes

    //float clip_threshold = 5.0f;
    //int size = batch * in_channels * in_height * in_width;
    //int threadsPerBlock = 256;
    //clipGradients << <(size + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> > (d_input_grad, size, clip_threshold);
    //CUDA_CHECK(cudaGetLastError());  // Check launch errors
    //CUDA_CHECK(cudaDeviceSynchronize());  // Ensure execution completes


    if (d_workspace) {
        CUDA_CHECK(cudaFree(d_workspace));
    }
}


void ConvLayer2D::backwardFilter(float* d_input, float* d_output_grad) {

    int filter_size = out_channels * in_channels * kernel_size * kernel_size;
    CUDA_CHECK(cudaMemset(d_filter_grad, 0, filter_size * sizeof(float)));

    float alpha = 1.0f, beta = 0.0f;
    cudnnConvolutionBwdFilterAlgo_t algo;
    size_t workspace_size = 0;

    int returnedAlgoCount;
    cudnnConvolutionBwdFilterAlgoPerf_t perfResults;

    // Use cudnnGetConvolutionBackwardFilterAlgorithm_v7 if using cuDNN 8+
    CUDNN_CHECK(cudnnFindConvolutionBackwardFilterAlgorithm(handle, input_desc, output_desc, conv_desc,
        filter_desc, 1, &returnedAlgoCount, &perfResults));

    algo = perfResults.algo;

    CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(handle, input_desc, output_desc, conv_desc,
        filter_desc, algo, &workspace_size));

    void* d_workspace = nullptr;
    if (workspace_size > 0) { // Allocate workspace only if needed
        CUDA_CHECK(cudaMalloc(&d_workspace, workspace_size));
    }

    CUDNN_CHECK(cudnnConvolutionBackwardFilter(handle, &alpha, input_desc, d_input, output_desc,
        d_output_grad, conv_desc, algo, d_workspace, workspace_size,
        &beta, filter_desc, d_filter_grad));

    if (d_workspace) { // Free workspace only if allocated
        CUDA_CHECK(cudaFree(d_workspace));
    }
}


void ConvLayer2D::backwardBias(float* d_output_grad) {

    float alpha = 1.0f, beta = 0.0f;

    CUDNN_CHECK(cudnnConvolutionBackwardBias(handle, &alpha, output_desc, d_output_grad,
        &beta, bias_desc, d_bias_grad));
}


float* printGpuArray2(float* d_in, int size, int newLine) {
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

// New: Apply gradients (SGD update for filter & bias)
void ConvLayer2D::updateWeights(float learning_rate) {
    float alpha = -learning_rate;

    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);


    int filter_size = out_channels * in_channels * kernel_size * kernel_size;
    int bgrad_size = out_channels;

    int threadsPerBlock = 256;

    // Normalize gradients by batch size

    float scale = 1.0f / batch;
    cublasSscal(cublasHandle, filter_size, &scale, d_filter_grad, 1);
    cublasSscal(cublasHandle, bgrad_size, &scale, d_bias_grad, 1);
    CUDA_CHECK(cudaGetLastError());  // Check launch errors
    CUDA_CHECK(cudaDeviceSynchronize());  // Ensure execution completes
    //printGpuArray2(d_filter_grad, filter_size, 10);

    
    // Clip gradients (optional)
    float clip_threshold = 1.0f;
    clipGradients << <(filter_size + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> > (d_filter_grad, filter_size, clip_threshold);
    CUDA_CHECK(cudaGetLastError());  // Check launch errors
    CUDA_CHECK(cudaDeviceSynchronize());  // Ensure execution completes

    clipGradients << <(bgrad_size + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> > (d_bias_grad, bgrad_size, clip_threshold);
    CUDA_CHECK(cudaGetLastError());  // Check launch errors
    CUDA_CHECK(cudaDeviceSynchronize());  // Ensure execution completes


    cublasSaxpy(cublasHandle, filter_size, &alpha, d_filter_grad, 1, d_filter, 1);

    cublasSaxpy(cublasHandle, out_channels, &alpha, d_bias_grad, 1, d_bias, 1);
    //printGpuArray2(d_filter, filter_size, 10);


    cublasDestroy(cublasHandle);
}

void ConvLayer2D::backward(float* d_input, float* d_output_grad, float lr) {
    backwardData(d_input, d_output_grad);
    cudaDeviceSynchronize();
    backwardFilter(d_input, d_output_grad);
    backwardBias(d_output_grad);
    updateWeights(lr);
}

float* ConvLayer2D::getOutput(int* outputSize) {
    if (outputSize)
    {
        *outputSize = batch * out_channels * out_height * out_width * sizeof(float);
    }
    return d_output;
}

float* ConvLayer2D::getInputGrad(int* inputGradSize) {
    if (inputGradSize)
    {
        *inputGradSize = batch * in_channels * in_height * in_width * sizeof(float);
    }
    return d_input_grad;
}


ConvLayer2D::~ConvLayer2D() {
    CUDA_CHECK(cudaFree(d_filter));
    CUDA_CHECK(cudaFree(d_bias));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_input_grad));
    CUDA_CHECK(cudaFree(d_filter_grad));
    CUDA_CHECK(cudaFree(d_bias_grad));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(bias_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(output_desc));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(filter_desc));
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc));
    CUDNN_CHECK(cudnnDestroy(handle));
}
