#ifndef LINEAR_LAYER_QUANTISED_CUH
#define LINEAR_LAYER_QUANTISED_CUH

#include "NNLayer.cuh"
#include <cublas_v2.h>
#include "Float10.cu"
class LinearLayerQuantised : public NNLayer {
public:
    LinearLayerQuantised(int batch_size, int in_features, int out_features);
    ~LinearLayerQuantised();

    void forward(float* d_input)override;                   // Forward pass
    void backward(float* d_input, float* d_output_grad, float lr)override;
    float* getOutput(int* outputSize = nullptr)override;
    float* getInputGrad(int* inputGradSize = nullptr)override;
    float* getAllWeights(int* outputSize)override;


    void backwardData(float* d_input, float* d_output_grad);   // Grad w.r.t. input
    void backwardWeights(float* d_input, float* d_output_grad); // Grad w.r.t. weights
    void backwardBias(float* d_output_grad);        // Grad w.r.t. bias
    void updateWeights(float learning_rate);        // Update weights and bias
    void initWeights(Float10* d_weight, int input_feat, int output_feat);

public:
    int batch_size;
    int in_features;
    int out_features;
    cublasHandle_t cublasHandle;

    Float10* d_weight;        // [out_features, in_features]
    Float10* d_bias;          // [out_features]
    float* d_output;        // Forward output

    float* d_input_grad;    // Grad w.r.t. input
    Float10* d_weight_grad;   // Grad w.r.t. weights
    Float10* d_bias_grad;     // Grad w.r.t. bias
};

#endif // LINEAR_LAYER_QUANTISED_CUH
