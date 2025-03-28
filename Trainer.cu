#include "Trainer.cuh"
#include <fstream>

Trainer::Trainer(NNModel& model, DatasetLoader& trainData, DatasetLoader& trainLabels,
    DatasetLoader& testData, DatasetLoader& testLabels,
    LossFunction* loss, float lr)
    : model(model), trainData(trainData), trainLabels(trainLabels),
    testData(testData), testLabels(testLabels), lossFunc(loss), lr(lr),
    numberForOneEpochTrain(trainData.totalBatches),
    numberForOneEpochTest(testData.totalBatches)
{
    model.getOutput(&this->outputFeature);
    this->outputFeature = this->outputFeature / (trainData.batchSize*sizeof(float));
}

Trainer::~Trainer(){}

__global__ void sumKernel(float* d_array, float* d_partialSums, int size) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory (or 0 if out of bounds)
    sdata[tid] = (i < size) ? d_array[i] : 0.0f;
    __syncthreads();

    // Perform parallel reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write block sum to global memory
    if (tid == 0) {
        d_partialSums[blockIdx.x] = sdata[0];
    }
}


float computeAverage(float* d_array, int M, int N) {
    int size = M * N;
    int threadsPerBlock = 256;
    int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;

    float* d_partialSums;
    cudaMalloc(&d_partialSums, blocks * sizeof(float));

    // Launch kernel with shared memory allocation
    sumKernel << <blocks, threadsPerBlock, threadsPerBlock * sizeof(float) >> > (d_array, d_partialSums, size);
    cudaDeviceSynchronize();

    // Copy partial sums to host
    float* h_partialSums = (float*)malloc(blocks * sizeof(float));
    cudaMemcpy(h_partialSums, d_partialSums, blocks * sizeof(float), cudaMemcpyDeviceToHost);

    // Final sum reduction on CPU
    float totalSum = 0.0f;
    for (int i = 0; i < blocks; i++) {
        totalSum += h_partialSums[i];
    }

    // Cleanup
    cudaFree(d_partialSums);
    free(h_partialSums); // <-- Free before returning

    return totalSum / size;
}

void Trainer::Train(int epochs) {
    float loss = 0.0;
    int batch = trainData.batchSize;
    float* d_grad;
    cudaMalloc((void**)&d_grad, outputFeature * batch * sizeof(float));
    for (int i = 0; i < epochs * numberForOneEpochTrain; i++)
    {
        float* target, * d_input;
        trainData.Next(&d_input);
        trainLabels.Next(&target);

        model.forward(d_input);

        float* d_loss = lossFunc->forward(model.getOutput(), target, outputFeature, batch);
        float tLoss = computeAverage(d_loss, batch, 1);
        loss += tLoss;

        if (i % (numberForOneEpochTrain-1) == 0 && i != 0)
        {
            printf("%d Epoch loss:%f\n", i / (numberForOneEpochTrain), loss / (numberForOneEpochTrain));
            loss = 0.0f;
        }
        cudaFree(d_loss);
        lossFunc->backward(model.getOutput(), target, d_grad, outputFeature, batch);
        model.backward(d_input, d_grad, lr);
        cudaDeviceSynchronize();
    }
}


float* GpuArrayToCpu(float* d_in, int size) {
    float* h_temp = (float*)malloc(size * sizeof(float));
    cudaMemcpy(h_temp, d_in, size * sizeof(float), cudaMemcpyDeviceToHost);
    return h_temp;
}

int argmax(float* input, int size) {
    int index = 0;
    for (size_t i = 0; i < size; i++)
    {
        if (input[i] > input[index]) {
            index = i;
        }
    }
    return index;
}

void Trainer::Test() {
    float loss = 0.0f;
    float accuracy = 0.0f;
    int batch = testData.batchSize;

    for (int i = 0; i < numberForOneEpochTest; i++)
    {
        float* target, * d_input;
        testData.Next(&d_input);
        testLabels.Next(&target);
        model.forward(d_input);

        float* d_loss = lossFunc->forward(model.getOutput(), target, outputFeature, batch);
        float tLoss = computeAverage(d_loss, batch, 1);
        loss += tLoss;;

        float* h_target = GpuArrayToCpu(target, outputFeature * batch);
        float* h_output = GpuArrayToCpu(model.getOutput(), batch * outputFeature);

        int correct = 0;
        for (int j = 0; j < batch; j++) {
            int pred_class = argmax(&h_output[j * outputFeature], outputFeature);  // Get index of max prob
            int true_class = argmax(&h_target[j * outputFeature], outputFeature);
            if (pred_class == true_class) correct++;
        }
        accuracy += (float)correct / batch;
        if (i % (numberForOneEpochTest -1) == 0 && i != 0)
        {
            printf("%d batch loss:%f\n", i / (numberForOneEpochTest), loss / (numberForOneEpochTest));
            printf("batch Accuracy: %.2f%%\n", (accuracy * 100.0f) / (numberForOneEpochTest));
            loss = 0.0f;
            accuracy = 0.0f;
        }
        cudaFree(d_loss);
    }
}


void Trainer::SaveWeightsToFile() {
    const std::string& filename = "weights.csv";
    int size = 0;
    float* weights = model.getAllWeights(&size);

    if (!weights || size == 0) {
        printf("No weights to save.\n");
        return;
    }

    std::ofstream outFile(filename);
    if (!outFile) {
        printf("Failed to open file for writing: %s\n", filename.c_str());
        return;
    }

    int columns = 32;  // Number of columns per row
    for (int i = 0; i < size; i++) {
        outFile << weights[i];

        // Check if it's the end of a row (32 columns per row)
        if ((i + 1) % columns == 0) {
            outFile << "\n";  // New line after 32 columns
        }
        else {
            outFile << ", ";  // Add comma between values
        }
    }

    // If the total number of weights isn't a multiple of 32, add a newline at the end
    if (size % columns != 0) {
        outFile << "\n";
    }

    outFile.close();
    printf("Weights saved to %s (%d values)\n", filename.c_str(), size);

    free(weights);  // Free allocated memory
}

void Trainer::ShowMinumumWeight() {
    int size = 0;
    float* weights = model.getAllWeights(&size);
    float min = 1000000000;
    for (size_t i = 0; i < size; i++)
    {
        if (weights[i] < min) {
            min = weights[i];
        }
    }
    printf("min weight: %f\n", min);
}

void Trainer::ShowMaximumWeight() {
    int size = 0;
    float* weights = model.getAllWeights(&size);
    float max = -1000000000;
    for (size_t i = 0; i < size; i++)
    {
        if (weights[i] > max) {
            max = weights[i];
        }
    }
    printf("max weight: %f\n", max);
}

void Trainer::ShowWeights() {

}