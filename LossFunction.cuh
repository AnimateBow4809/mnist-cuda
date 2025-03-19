#pragma once

#include <cuda_runtime.h>
#include <cudnn.h>
#include <device_launch_parameters.h>  // Required for kernel launch parameters

class LossFunction {
protected:
    cudnnHandle_t cudnn; // cuDNN handle for GPU operations

public:
    LossFunction() { cudnnCreate(&cudnn); }
    virtual ~LossFunction() { cudnnDestroy(cudnn); }

    // Pure virtual functions
    virtual float* forward(const float* predictions, const float* targets, int size,  int batch) = 0;
    virtual void backward(const float* predictions, const float* targets, float* grad, int size, int batch) = 0;
};

class MSELoss : public LossFunction {
public:
    float* forward(const float* predictions, const float* targets, int size, int batch)override;
    void backward(const float* predictions, const float* targets, float* grad, int size, int batch)override;
};

