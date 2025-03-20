#ifndef CONV_LAYER_2D_CUH
#define CONV_LAYER_2D_CUH

#include <cudnn.h>
#include <cuda_runtime.h>
#include <iostream>

class ConvLayer2D {
public:
    ConvLayer2D(int batch, int in_channels, int in_height, int in_width,
        int out_channels, int kernel_size, int stride, int padding);
    ~ConvLayer2D();

    void forward(float* d_input);
    void backward(float* d_input, float* d_output_grad,float lr);

    void backwardData(float* d_input, float* d_output_grad);
    void backwardFilter(float* d_input, float* d_output_grad);
    void backwardBias(float* d_output_grad);
    void updateWeights(float learning_rate);

public:
    int batch, in_channels, in_height, in_width;
    int out_channels, out_height, out_width;
    int kernel_size, stride, padding;

    cudnnHandle_t handle;
    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnTensorDescriptor_t bias_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    float* d_filter;       // Convolution filters (weights)
    float* d_bias;         // <-- Bias for each output channel
    float* d_output;       // Forward pass output

    float* d_input_grad;   // Gradient w.r.t. input
    float* d_filter_grad;  // Gradient w.r.t. filter
    float* d_bias_grad;    // <-- Gradient w.r.t. bias (added for bias handling)

};


#endif // CONV_LAYER_2D_CUH
