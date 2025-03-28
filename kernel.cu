#include "ConvLayer2D.cuh"
#include <cublas_v2.h>
#include "LinearLayer.cuh"
#include <cuda_runtime.h>  // Core CUDA runtime API
#include <device_launch_parameters.h>  // Required for kernel launch parameters
#include <curand_kernel.h>
#include "LossFunction.cuh"
#include "MaxPoolLayer.cuh"
#include "ReluLayer.cuh"
#include "MNISTTest.h"
#include "softmaxLayer.cuh"
#include"NNModel.cuh"
#include "DatasetLoader.cuh"
#include "Trainer.cuh"

float* printGpuArray(float* d_in, int size, int newLine,bool print=true) {
    float* h_temp = (float*)malloc(size * sizeof(float));
    cudaMemcpy(h_temp, d_in, size * sizeof(float), cudaMemcpyDeviceToHost);
    if (!print)
    {
        return h_temp;
    }
    for (size_t i = 0; i < size; i++)
    {
        printf("%f ", h_temp[i]);
        if ((i + 1) % newLine == 0)
        {
            printf("\n");
        }
    }
    return h_temp;
}

float* initialiseGpuArrayRandom(float* d_in, int size,int newLine) {
    float* h_input = (float*)malloc(size* sizeof(float));
    srand(time(0));
    for (size_t i = 0; i < size; i++)
    {
        h_input[i] = (rand() / (float)RAND_MAX);
        if (newLine!=-1)
        {
            printf("%f ", h_input[i]);
            if ((i + 1) % newLine == 0)
            {
                printf("\n");
            }
        }
    }
    cudaMemcpy(d_in, h_input, size* sizeof(float), cudaMemcpyHostToDevice);
    return h_input;
}

void multiplyMatrix(float* d_matrix, int rows, int cols,float alpha, cublasHandle_t handle) {
    int size = rows * cols;
    cublasSscal(handle, size, &alpha, d_matrix, 1);
}






__global__ void matMulKernelRowMajor(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

float* PushArrayIntoGpu(float* arr_h,dim3 dimentions) {
    float* arr_d;
    cudaMalloc((void**)&arr_d, dimentions.x * dimentions.y * dimentions.z * sizeof(float));
    cudaMemcpy(arr_d, arr_h, dimentions.x * dimentions.y * dimentions.z * sizeof(float), cudaMemcpyHostToDevice);
    return arr_d;
}

float* createMatrix(int n, int m) {
    float* matrix = new float[n * m]; // Allocate memory for a 1D array
    for (int i = 0; i < n * m; ++i) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX; // Random value between 0 and 1
    }
    return matrix;
}

float* multMatrix(float* in,int row,int col, float alpha) {

    float* ans =(float*) malloc(sizeof(float) * row * col);
    for (size_t i = 0; i < row*col; i++)
    {
        ans[i] = alpha * in[i];
    }
    return ans;
}






int main() {
    float* train_images;
    float* train_labels;
    float* test_images, *test_labels;

    int num_train, img_size,num_test,test_img_size;
    read_mnist_images("train-images.idx3-ubyte", train_images, num_train, img_size);
    read_mnist_labels("train-labels.idx1-ubyte", train_labels, num_train);
    read_mnist_images("t10k-images.idx3-ubyte", test_images, num_test, test_img_size);
    read_mnist_labels("t10k-labels.idx1-ubyte", test_labels, num_test);

    
    srand(static_cast<unsigned>(time(0))); // Seed for randomness

    int batch = 200;
    int input_feat = 28*28;
    int output_feat = 10;
    
    DatasetLoader train_image_loader(num_train, batch, 28, 28, train_images);
    DatasetLoader train_label_loader(num_train, batch,1, output_feat, train_labels);

    DatasetLoader test_image_loader(num_test, batch, 28, 28, test_images);
    DatasetLoader test_label_loader(num_test, batch, 1, output_feat, test_labels);


    std::vector<NNLayer*> layers;
    layers.push_back(new ConvLayer2D(batch, 1, 28, 28, 32, 3, 2, 1));
    layers.push_back(new ReLULayer(batch, 32, 14, 14));
    layers.push_back(new ConvLayer2D(batch, 32, 14, 14, 64, 3, 2, 1));
    layers.push_back(new ReLULayer(batch, 64, 7, 7));
    layers.push_back(new ConvLayer2D(batch, 64, 7, 7, 128, 3, 2, 1));
    layers.push_back(new ReLULayer(batch, 128, 4, 4));
    layers.push_back(new ConvLayer2D(batch, 128, 4, 4, 32, 3, 2, 1));
    layers.push_back(new ReLULayer(batch, 32, 2, 2));
    //layers.push_back(new ConvLayer2D(batch, 32, 2, 2,10, 3,1 , 1));
    layers.push_back(new LinearLayer(batch, 128, 64));
    layers.push_back(new ReLULayer(batch, 1, 1, 64));
    layers.push_back(new LinearLayer(batch, 64, 32));
    layers.push_back(new ReLULayer(batch, 1, 1, 32));
    layers.push_back(new LinearLayer(batch, 32, 10));
    layers.push_back(new SoftmaxLayer(batch, 1, 1, 10));

    NNModel model(layers);
    LossFunction* l1 = new CrossEntropyLoss();

    Trainer trainer(model, train_image_loader, train_label_loader, test_image_loader, test_label_loader, l1, 0.01);

    int choice = 0;
    while (true) {
        printf("Enter your choice: ");

        // Check if scanf successfully reads an integer
        if (scanf("%d", &choice) != 1) {
            printf("Invalid input! Please enter a number.\n");
            while (getchar() != '\n'); // Clear the input buffer
            continue;
        }

        if (choice == 1) {
            trainer.Train(10);
        }
        else if (choice == 2) {
            trainer.Test();
        }
        else {
            printf("Invalid choice! Enter 1 or 2.\n");
        }
    }

    
    return 0;

}
