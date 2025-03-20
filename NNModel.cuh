#ifndef NN_MODEL_CUH
#define NN_MODEL_CUH

#include "NNLayer.cuh"

#include <device_launch_parameters.h>  // Required for kernel launch parameters

#include <cuda_runtime.h>
#include <iostream>
#include <vector>


class NNModel:public NNLayer
{
public:
	NNModel(std::vector<NNLayer*> layers);
	~NNModel();

	void forward(float* d_input)override;
	void backward(float* d_input, float* d_output_grad, float lr)override;
	float* getOutput(int* outputSize=nullptr)override;
	float* getInputGrad(int* inputGradSize=nullptr)override;

public:
	float* d_output;       // Forward pass output
	std::vector<NNLayer*> layers;
	float* d_input_grad;   // Gradient w.r.t. input

};


#endif // NN_MODEL_CUH
