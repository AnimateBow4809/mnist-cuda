#include "ConvLayer2D.cuh"
#include <cublas_v2.h>
#include <random>

ConvLayer2D::ConvLayer2D(int batch, int in_channels, int in_height, int in_width,
    int out_channels, int kernel_size, int stride, int padding)
    : batch(batch), in_channels(in_channels), in_height(in_height), in_width(in_width),
    out_channels(out_channels), kernel_size(kernel_size), stride(stride), padding(padding) {

    cudnnCreate(&handle);

    // Input Descriptor
    cudnnCreateTensorDescriptor(&input_desc);
    cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        batch, in_channels, in_height, in_width);

    // Filter Descriptor
    cudnnCreateFilterDescriptor(&filter_desc);
    cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
        out_channels, in_channels, kernel_size, kernel_size);

    // Convolution Descriptor
    cudnnCreateConvolutionDescriptor(&conv_desc);
    cudnnSetConvolution2dDescriptor(conv_desc, padding, padding, stride, stride, 1, 1,
        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

    // Output Shape Calculation
    cudnnGetConvolution2dForwardOutputDim(conv_desc, input_desc, filter_desc,
        &batch, &out_channels, &out_height, &out_width);

    // Output Descriptor
    cudnnCreateTensorDescriptor(&output_desc);
    cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        batch, out_channels, out_height, out_width);

    // Bias Descriptor
    cudnnCreateTensorDescriptor(&bias_desc);
    cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, out_channels, 1, 1);


    // Allocate Memory for filter, output, and bias
    cudaMalloc(&d_filter, out_channels * in_channels * kernel_size * kernel_size * sizeof(float));
    cudaMalloc(&d_bias, out_channels * sizeof(float));  // New: Bias
    cudaMalloc(&d_output, batch * out_channels * out_height * out_width * sizeof(float));

    // Initialize weights and bias (optional: random init or zero)
    std::vector<float> host_filter(out_channels * in_channels * kernel_size * kernel_size + out_channels);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 0.1f);

    for (auto& x : host_filter) x = dist(gen);  // random init

    cudaMemcpy(d_filter, host_filter.data(), (host_filter.size() - out_channels) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, &host_filter.data()[host_filter.size() - out_channels]
        , out_channels * sizeof(float), cudaMemcpyHostToDevice);

    // Gradients
    cudaMalloc(&d_input_grad, batch * in_channels * in_height * in_width * sizeof(float));
    cudaMalloc(&d_filter_grad, out_channels * in_channels * kernel_size * kernel_size * sizeof(float));
    cudaMalloc(&d_bias_grad, out_channels * sizeof(float));
}

void ConvLayer2D::forward(float* d_input) {
    float alpha = 1.0f, beta = 0.0f;
    size_t workspace_size = 0;
    cudnnConvolutionFwdAlgo_t algo;

    // Select Fastest Algorithm
    int returnedAlgoCount;
    cudnnConvolutionFwdAlgoPerf_t perfResults;
    cudnnFindConvolutionForwardAlgorithm(handle, input_desc, filter_desc, conv_desc,
        output_desc, 1, &returnedAlgoCount, &perfResults);
    algo = perfResults.algo;

    cudnnGetConvolutionForwardWorkspaceSize(handle, input_desc, filter_desc, conv_desc,
        output_desc, algo, &workspace_size);
    void* d_workspace;
    cudaMalloc(&d_workspace, workspace_size);

    // Perform Convolution
    cudnnConvolutionForward(handle, &alpha, input_desc, d_input, filter_desc, d_filter,
        conv_desc, algo, d_workspace, workspace_size, &beta,
        output_desc, d_output);

    cudnnAddTensor(handle, &alpha, bias_desc, d_bias, &alpha, output_desc, d_output);

    cudaFree(d_workspace);
}

void ConvLayer2D::backwardData(float* d_input, float* d_output_grad) {
    float alpha = 1.0f, beta = 0.0f;
    cudnnConvolutionBwdDataAlgo_t algo;
    size_t workspace_size = 0;

    int returnedAlgoCount;
    cudnnConvolutionBwdDataAlgoPerf_t perfResults;
    cudnnFindConvolutionBackwardDataAlgorithm(handle, filter_desc, output_desc, conv_desc,
        input_desc, 1, &returnedAlgoCount, &perfResults);
    algo = perfResults.algo;

    cudnnGetConvolutionBackwardDataWorkspaceSize(handle, filter_desc, output_desc, conv_desc,
        input_desc, algo, &workspace_size);

    void* d_workspace;
    cudaMalloc(&d_workspace, workspace_size);

    cudnnConvolutionBackwardData(handle, &alpha, filter_desc, d_filter, output_desc,
        d_output_grad, conv_desc, algo, d_workspace, workspace_size,
        &beta, input_desc, d_input_grad);

    cudaMemcpy(d_input, d_input_grad, batch * in_channels * in_height * in_width * sizeof(float), cudaMemcpyDeviceToDevice);

    cudaFree(d_workspace);
}

void ConvLayer2D::backwardFilter(float* d_input, float* d_output_grad) {
    float alpha = 1.0f, beta = 0.0f;
    cudnnConvolutionBwdFilterAlgo_t algo;
    size_t workspace_size = 0;

    int returnedAlgoCount;
    cudnnConvolutionBwdFilterAlgoPerf_t perfResults;
    cudnnFindConvolutionBackwardFilterAlgorithm(handle, input_desc, output_desc, conv_desc,
        filter_desc, 1, &returnedAlgoCount, &perfResults);
    algo = perfResults.algo;

    cudnnGetConvolutionBackwardFilterWorkspaceSize(handle, input_desc, output_desc, conv_desc,
        filter_desc, algo, &workspace_size);

    void* d_workspace;
    cudaMalloc(&d_workspace, workspace_size);

    cudnnConvolutionBackwardFilter(handle, &alpha, input_desc, d_input, output_desc,
        d_output_grad, conv_desc, algo, d_workspace, workspace_size,
        &beta, filter_desc, d_filter_grad);

    cudaFree(d_workspace);
}

void ConvLayer2D::backwardBias(float* d_output_grad) {
    float alpha = 1.0f, beta = 0.0f;

    cudnnTensorDescriptor_t bias_desc;
    cudnnCreateTensorDescriptor(&bias_desc);
    cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, out_channels, 1, 1);

    cudnnConvolutionBackwardBias(handle, &alpha, output_desc, d_output_grad,
        &beta, bias_desc, d_bias_grad);

    cudnnDestroyTensorDescriptor(bias_desc);
}

// New: Apply gradients (SGD update for filter & bias)
void ConvLayer2D::updateWeights(float learning_rate) {
    float alpha = -learning_rate;

    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);

    int filter_size = out_channels * in_channels * kernel_size * kernel_size;
    cublasSaxpy(cublas_handle, filter_size, &alpha, d_filter_grad, 1, d_filter, 1);

    cublasSaxpy(cublas_handle, out_channels, &alpha, d_bias_grad, 1, d_bias, 1);

    cublasDestroy(cublas_handle);
}

ConvLayer2D::~ConvLayer2D() {
    cudaFree(d_filter);
    cudaFree(d_bias);  // Free bias memory
    cudaFree(d_output);
    cudaFree(d_input_grad);
    cudaFree(d_filter_grad);
    cudaFree(d_bias_grad);
    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(bias_desc);
    cudnnDestroyTensorDescriptor(output_desc);
    cudnnDestroyFilterDescriptor(filter_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroy(handle);
}
