#ifndef LINEAR_LAYER_H
#define LINEAR_LAYER_H




class NNLayer
{
public:
	NNLayer() {};
	virtual ~NNLayer() {};

	virtual void forward(float* d_input) = 0;
	virtual void backward(float* d_input, float* d_output_grad, float lr) = 0;
	virtual float* getOutput(int* outputSize=nullptr) = 0;
	virtual float* getInputGrad(int* inputGradSize=nullptr) = 0;
};

#endif