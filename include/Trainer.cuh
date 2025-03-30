#ifndef TRAINER_CUH
#define TRAINER_CUH
#include"DatasetLoader.cuh"
#include "NNModel.cuh"
#include "LossFunction.cuh"

class Trainer
{
public:
	Trainer(NNModel& model, DatasetLoader& trainData, DatasetLoader& trainLabels
		, DatasetLoader& testData, DatasetLoader& testLabels,LossFunction* loss, float lr = 0.01);
	~Trainer();

	void Train(int epochs);
	void Test();
	void SaveWeightsToFile();
	void ShowWeights();
	void ShowMinumumWeight();
	void ShowMaximumWeight();
	float lr;


private:
	NNModel model;
	DatasetLoader trainData;
	DatasetLoader trainLabels;
	int numberForOneEpochTrain;
	int numberForOneEpochTest;
	int outputFeature;
	LossFunction* lossFunc;
	DatasetLoader testData;
	DatasetLoader testLabels;
};


#endif