#ifndef DATASET_LOADER_CUH
#define DATASET_LOADER_CUH

#include <cuda_runtime.h>
#include <cudnn.h>
#include <iostream>
#include <stdexcept>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include "Utils.cuh"

class DatasetLoader {
public:
    DatasetLoader(int totalSize, int batchSize, int height, int width, float* data, bool copyData = false);
    ~DatasetLoader();

    void Next(float** out);

public:
    int width;
    int height;
    int batchSize;
    int totalBatches;
    int currentBatchCount;
    float* data;
    float* currentBatchDevice;
    bool ownsData;
};

#endif // DATASET_LOADER_CUH
