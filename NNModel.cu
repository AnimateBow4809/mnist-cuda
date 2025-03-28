#include "NNModel.cuh"

NNModel::NNModel(std::vector<NNLayer*> layers) {
    this->layers = layers;
    this->d_input_grad = layers.at(0)->getInputGrad();
    this->d_output = layers.at(layers.size() - 1)->getOutput();
}

NNModel::~NNModel() {
    // Delete all layers if NNModel owns them
    for (auto layer : layers) {
        delete layer;
    }
}

void NNModel::forward(float* d_input) {
    layers.at(0)->forward(d_input);
    cudaDeviceSynchronize();

    for (size_t i = 1; i < layers.size(); i++) {
        layers.at(i)->forward(layers.at(i - 1)->getOutput());
        cudaDeviceSynchronize();
    }
    this->d_output = layers.at(layers.size() - 1)->getOutput();
    cudaDeviceSynchronize();
}

void NNModel::backward(float* d_input, float* d_output_grad, float lr) {
    // Revised backward pass: propagate gradients in reverse order.
    float* grad = d_output_grad;
    for (int i = layers.size() - 1; i >= 0; i--) {
        float* layer_input = (i == 0) ? d_input : layers.at(i - 1)->getOutput();
        layers.at(i)->backward(layer_input, grad, lr);
        grad = layers.at(i)->getInputGrad();
        cudaDeviceSynchronize();
    }
    this->d_input_grad = grad;
}

float* NNModel::getOutput(int* outputSize) {
    if (outputSize) {
        int temp;
        layers.at(layers.size() - 1)->getOutput(&temp);
        *outputSize = temp;
    }
    return d_output;
}

float* NNModel::getInputGrad(int* inputGradSize) {
    if (inputGradSize) {
        int temp;
        layers.at(0)->getInputGrad(&temp);
        *inputGradSize = temp;
    }
    return d_input_grad;
}


float* NNModel::getAllWeights(int* outputSize) {
    int size = 0;

    // First, compute the total size
    for (size_t i = 0; i < layers.size(); i++) {
        int temp = 0;
        float* ptr = layers.at(i)->getAllWeights(&temp);
        if (ptr) {  // Check for nullptr
            size += temp;
            free(ptr); // Prevent memory leak
        }
    }

    // Allocate only if size > 0
    float* h_temp = nullptr;
    if (size > 0) {
        h_temp = (float*)malloc(size * sizeof(float));
        if (!h_temp) {
            std::cerr << "Memory allocation failed!" << std::endl;
            return nullptr;
        }

        int offset = 0;
        for (size_t i = 0; i < layers.size(); i++) {
            int temp = 0;
            float* ptr = layers.at(i)->getAllWeights(&temp);
            if (ptr) {
                memcpy(h_temp + offset, ptr, temp * sizeof(float));  // Efficient copy
                offset += temp;
                free(ptr);  // Free layer's allocated memory
            }
        }
    }

    *outputSize = size;
    return h_temp;
}

