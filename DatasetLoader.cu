#include "DatasetLoader.cuh"

DatasetLoader::DatasetLoader(int totalSize, int batchSize, int height, int width, float* data, bool copyData) {
    this->batchSize = batchSize;
    this->totalBatches = totalSize / batchSize;
    this->width = width;
    this->height = height;
    this->currentBatchCount = 0;
    this->ownsData = copyData;

    if (copyData) {
        this->data = (float*)malloc(totalSize * sizeof(float));
        if (!this->data) {
            throw std::runtime_error("Failed to allocate memory for dataset");
        }
        memcpy(this->data, data, totalSize * sizeof(float));
    }
    else {
        this->data = data;
    }
    this->currentBatchDevice = nullptr;
}

DatasetLoader::~DatasetLoader() {
    cudaFree(currentBatchDevice);
    if (ownsData && data) {
        free(data);
        data = nullptr;
    }
}

float* MoveArrayIntoGpu(float* arr_h, dim3 dimensions) {
    float* arr_d;
    size_t size = dimensions.x * dimensions.y * dimensions.z * sizeof(float);
    cudaMalloc((void**)&arr_d, size);
    cudaMemcpy(arr_d, arr_h, size, cudaMemcpyHostToDevice);
    return arr_d;
}

void DatasetLoader::Next(float** out) {
    if (currentBatchCount >= totalBatches) {
        std::cerr << "Warning: All batches processed. Resetting count.\n";
        currentBatchCount = 0;
    }

    dim3 dimensions(batchSize, width, height);

    if (*out != nullptr) {
        //cudaFree(*out);
    }

    *out = MoveArrayIntoGpu(&data[width * height * currentBatchCount * batchSize], dimensions);
    currentBatchDevice = *out;
    currentBatchCount++;
}
