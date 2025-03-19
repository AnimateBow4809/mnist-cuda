#ifndef UTILS_CUH
#define UTILS_CUH

#include <cuda_runtime.h>
#include <cudnn.h>
#include <iostream>
#include <stdexcept>
#include <cublas_v2.h>
#include <cuda_runtime.h>  // Core CUDA runtime API
#include <device_launch_parameters.h>  // Required for kernel launch parameters
#include <curand_kernel.h>


__global__ void clipGradients(float* gradients, int size, float clip_value);
__global__ void clipGradientsL2(float* grads, int size, float clip_threshold);

#endif