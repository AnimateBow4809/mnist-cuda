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
