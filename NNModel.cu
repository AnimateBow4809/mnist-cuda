#include "NNModel.cuh"


NNModel::NNModel(std::vector<NNLayer*> layers) {

	this->layers = layers;
	this->d_input_grad =layers.at(0)->getInputGrad();
	this->d_output = layers.at(layers.size() - 1)->getOutput();

}


NNModel::~NNModel() {

}

void NNModel::forward(float* d_input){
	layers.at(0)->forward(d_input);
	for (size_t i = 1; i < layers.size(); i++)
	{
		layers.at(i)->forward(layers.at(i - 1)->getOutput());
	}
	this->d_output = layers.at(layers.size() - 1)->getOutput();


}


void NNModel::backward(float* d_input, float* d_output_grad, float lr){
	layers.at(layers.size() - 1)->backward(layers.at(layers.size() - 2)->getOutput(), d_output_grad, lr);
	for (size_t i = layers.size() - 1; i > 0; i--)
	{
		layers.at(i)->backward(layers.at(i - 1)->getOutput(), layers.at(i - 1)->getInputGrad(), lr);
	}
	layers.at(0)->backward(d_input, layers.at(1)->getInputGrad(), lr);
	this->d_input_grad = layers.at(0)->getInputGrad();
}


float* NNModel::getOutput(int* outputSize){
	if (outputSize)
	{
		int temp;
		layers.at(layers.size() - 1)->getOutput(&temp);
		*outputSize = temp;
	}
	return d_output;
}
float* NNModel::getInputGrad(int* inputGradSize){
	if (inputGradSize)
	{
		int temp;
		layers.at(0)->getInputGrad(&temp);
		*inputGradSize = temp;
	}
	return d_input_grad;
}