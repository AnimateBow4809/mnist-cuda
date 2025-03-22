#ifndef RELU_LAYER_CUH
#define RELU_LAYER_CUH

#include <device_launch_parameters.h>  // Required for kernel launch parameters

#include <cuda_runtime.h>
#include <iostream>
#include "NNLayer.cuh"

class ReLULayer:public NNLayer {
public:
    ReLULayer(int batch, int channels, int height, int width,float leak=0.0f);
    ~ReLULayer();

    void forward(float* d_input)override;
    void backward(float* d_input, float* d_output_grad,float lr)override;
    float* getOutput(int* outputSize=nullptr)override;
    float* getInputGrad(int* inputGradSize=nullptr)override;

public:
    int batch, channels, height, width;
    int num_elements;
    float leak;
    float* d_output;      // Forward output
    float* d_input_grad;  // Backward gradient w.r.t. input
};

#endif // RELU_LAYER_H
