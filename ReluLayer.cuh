#ifndef RELU_LAYER_CUH
#define RELU_LAYER_CUH

#include <device_launch_parameters.h>  // Required for kernel launch parameters

#include <cuda_runtime.h>
#include <iostream>

class ReLULayer {
public:
    ReLULayer(int batch, int channels, int height, int width);
    ~ReLULayer();

    void forward(float* d_input);
    void backward(float* d_input, float* d_output_grad);

public:
    int batch, channels, height, width;
    int num_elements;

    float* d_output;      // Forward output
    float* d_input_grad;  // Backward gradient w.r.t. input
};

#endif // RELU_LAYER_H
