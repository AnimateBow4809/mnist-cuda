#ifndef MAXPOOL_LAYER_CUH
#define MAXPOOL_LAYER_CUH

#include <cudnn.h>
#include <cuda_runtime.h>  // Core CUDA runtime API
#include <device_launch_parameters.h>  // Required for kernel launch parameters

#include "NNLayer.cuh"

class MaxPoolLayer : public NNLayer {
public:
    MaxPoolLayer(int batch, int channels, int height, int width, int pool_size);
    ~MaxPoolLayer();

    void forward(float* d_input) override;
    void backward(float* d_input, float* d_output_grad, float lr) override;
    float* getOutput(int* outputSize = nullptr) override;
    float* getInputGrad(int* inputGradSize = nullptr) override;

private:
    int batch, channels, height, width;
    int pooled_height, pooled_width;
    int pool_size;

    float* d_output;
    float* d_input_grad;

    // cuDNN Handles
    cudnnHandle_t cudnn;
    cudnnPoolingDescriptor_t poolingDesc;
    cudnnTensorDescriptor_t inputDesc, outputDesc;
};

#endif // MAXPOOL_LAYER_CUH
