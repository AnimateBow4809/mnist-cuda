#ifndef LINEAR_LAYER_H
#define LINEAR_LAYER_H

#include <cudnn.h>


class LinearLayer {
public:
    LinearLayer(int batch_size, int in_features, int out_features);
    ~LinearLayer();

    void forward(float* d_input);                   // Forward pass
    void backwardData(float* d_input, float* d_output_grad);   // Grad w.r.t. input
    void backwardWeights(float* d_input, float* d_output_grad); // Grad w.r.t. weights
    void backwardBias(float* d_output_grad);        // Grad w.r.t. bias
    void updateWeights(float learning_rate);        // Update weights and bias
    void initWeights(float* d_weight, int input_feat, int output_feat);

public:
    int batch_size;
    int in_features;
    int out_features;

    cudnnHandle_t handle;

    cudnnTensorDescriptor_t input_desc;
    cudnnTensorDescriptor_t output_desc;
    cudnnTensorDescriptor_t bias_desc;

    float* d_weight;        // [out_features, in_features]
    float* d_bias;          // [out_features]
    float* d_output;        // Forward output

    float* d_input_grad;    // Grad w.r.t. input
    float* d_weight_grad;   // Grad w.r.t. weights
    float* d_bias_grad;     // Grad w.r.t. bias
};

#endif // LINEAR_LAYER_H
