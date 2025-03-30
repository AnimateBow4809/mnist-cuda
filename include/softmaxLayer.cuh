#ifndef SOFTMAX_LAYER_CUH
#define SOFTMAX_LAYER_CUH

#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <iostream>
#include "NNLayer.cuh"

class SoftmaxLayer : public NNLayer {
public:
    SoftmaxLayer(int batch, int channels, int height, int width);
    ~SoftmaxLayer();

    void forward(float* d_input) override;
    void backward(float* d_input, float* d_output_grad, float lr) override;
    float* getOutput(int* outputSize = nullptr) override;
    float* getInputGrad(int* inputGradSize = nullptr) override;
    float* getAllWeights(int* outputSize)override;

public:
    int batch, channels, height, width;
    int num_classes;
    int num_elements;

    float* d_output;      // Forward output
    float* d_input_grad;  // Backward gradient w.r.t. input
};

#endif // SOFTMAX_LAYER_CUH
