#include "ConvLayer2D.cuh"
#include <cublas_v2.h>
#include "LinearLayer.cuh"
#include <cuda_runtime.h>  // Core CUDA runtime API
#include <device_launch_parameters.h>  // Required for kernel launch parameters
#include <curand_kernel.h>
#include "LossFunction.cuh"
#include "ReluLayer.cuh"
#include "MNISTTest.h"
#include"NNModel.cuh"
#include "DatasetLoader.cuh"
float* printGpuArray(float* d_in, int size, int newLine) {
    float* h_temp = (float*)malloc(size * sizeof(float));
    cudaMemcpy(h_temp, d_in, size * sizeof(float), cudaMemcpyDeviceToHost);

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
    int num_train, img_size;



    read_mnist_images("t10k-images.idx3-ubyte", train_images, num_train, img_size);
    read_mnist_labels("train-labels.idx1-ubyte", train_labels, num_train);
    
    srand(static_cast<unsigned>(time(0))); // Seed for randomness
    cublasHandle_t cchandle;
    cublasCreate(&cchandle);
    //float arr_h[] = { 1,2,3,4,5,6,
    //            /*    7,8,9,10,11,12,
    //                13,14,15,16,17,18,
    //                19,20,21,22,23,24 
    //};
    int batch = 200;
    int input_feat = 28*28;
    int output_feat = 1;
    //int hidden = 10;
    dim3 dimensions_in(1, batch, input_feat);  // 1x4x6 tensor
    dim3 dimensions_out(1, batch, output_feat);  // 1x4x6 tensor
    
   // float* d_input = PushArrayIntoGpu(arr_h, dimensions_in);
    DatasetLoader image_loader(num_train/6, batch, 28, 28, train_images);
    DatasetLoader label_loader(num_train/6, batch, 1, 1, train_labels);

    std::vector<NNLayer*> layers;
    layers.push_back(new ConvLayer2D(batch, 1, 28, 28, 32, 3, 1, 1));
    layers.push_back(new ReLULayer(batch, 32, 28, 28));

    layers.push_back(new ConvLayer2D(batch, 32, 28, 28, 64, 3, 1, 1));
    layers.push_back(new ReLULayer(batch, 64, 28, 28));

    layers.push_back(new ConvLayer2D(batch, 64, 28, 28, 128, 3, 1, 1));
    layers.push_back(new ReLULayer(batch, 128, 28, 28));

    layers.push_back(new ConvLayer2D(batch, 128, 28, 28, 256, 3, 1, 1));
    layers.push_back(new ReLULayer(batch, 256, 28, 28));


    layers.push_back(new LinearLayer(batch, 256*28*28, 512));
    layers.push_back(new ReLULayer(batch, 1, 1, 512));

    layers.push_back(new LinearLayer(batch, 512, 1));

    NNModel model(layers);
    LossFunction* l1 = new MSELoss();

    cudaDeviceSynchronize();
    float* d_grad;
    cudaMalloc((void**) & d_grad, output_feat *batch*sizeof(float));

    for (int i = 0; i < 51; i++)
    {
        printf("\n%d iter:\n", i);
        float* target, * d_input;
        image_loader.Next(&d_input);
        label_loader.Next(&target);
        printf("Target:\n");
        printGpuArray(target, output_feat * batch, 10);
        model.forward(d_input);
        printf("\nResults:\n");
        float* h_output = printGpuArray(model.getOutput(), batch * output_feat, 10);

        float* d_loss = l1->forward(model.getOutput(), target, output_feat, batch);
        cudaFree(d_loss);
        l1->backward(model.getOutput(), target, d_grad, output_feat, batch);
        float lr = 0.01;
        model.backward(d_input,d_grad,lr);
        cudaDeviceSynchronize();
    }

    return 0;

}



//conv.forward(d_input);
//
//printf("bias:\n");
//float* h_bias = printGpuArray(conv.d_bias, 1, 1);
//
//printf("Filter:\n");
//float* h_filter = printGpuArray(conv.d_filter, 3 * 3, 3);
//
//printf("Results:\n");
//float* h_output = printGpuArray(conv.d_output, 3 * 3, 3);
//
//multiplyMatrix(conv.d_output, 3, 3, 2, handle);
////cudaMemset(&conv.d_output[4], 0, 1 * sizeof(float));
//
//
//conv.backwardFilter(d_input, conv.d_output);
//conv.backwardBias(conv.d_output);
//
//conv.updateWeights(0.05);
//




////linear test
//// Define Conv Layer: (Batch=1, InChannels=1, Height=5, Width=5, OutChannels=1, Kernel=3, Stride=1, Padding=0)
    //ConvLayer conv(1, 1, 5, 5, 1, 3, 1, 0);
    //cublasHandle_t handle;
    //cublasCreate(&handle);
    //int input_feat = 3;
    //int output_feat = 10;

    //LinearLayer lin(1, input_feat, output_feat);

    //cudaDeviceSynchronize();
    //int size = input_feat;

    //float *d_input,*d1_input;
    //cudaMalloc((void**)&d_input, size*sizeof(float));
    //cudaMalloc((void**)&d1_input, size*sizeof(float));

    //float* test = (float*)malloc(2 * sizeof(float));
    //test[0] = 0;
    //printf("input:\n");
    //float* h_input=initialiseGpuArrayRandom(d_input, input_feat,input_feat);
    //
    //for (size_t i = 0; i < 1; i++)
    //{

    //    lin.forward(d_input);
    //    printf("\nbias:\n");
    //    float* h_linbias = printGpuArray(lin.d_bias, output_feat, output_feat);
    //    printf("\nWeights:\n");
    //    float* h_linweights = printGpuArray(lin.d_weight, input_feat*output_feat, input_feat);
    //    printf("\nResults:\n");
    //    float* h_output = printGpuArray(lin.d_output, output_feat, output_feat);

    //    cudaMemcpy(&test[1], &lin.d_output[2], 1 * sizeof(float), cudaMemcpyDeviceToHost);
    //    test[0] = test[1] - 3.14; //output - target
    //    cudaMemcpy(&lin.d_output[2], &test[0], 1 * sizeof(float), cudaMemcpyHostToDevice);

    //    multiplyMatrix(lin.d_output, 1, output_feat, 2, handle);
    //    printf("\GRAD::\n");
    //    printGpuArray(lin.d_output, output_feat, output_feat);
    //    lin.backwardData(d_input, lin.d_output);
    //    lin.backwardWeights(d_input, lin.d_output);
    //    lin.backwardBias(lin.d_output);
    //    lin.updateWeights(0.05);

    //}


    //
    //return 0;



///// old loop

//for (size_t i = 0; i < 51; i++)
//{
//    printf("\n%d iter:\n", i);
//
//    //   float* h_input = createMatrix(batch,input_feat);
//    //   float* d_input = PushArrayIntoGpu(h_input, dimensions_in);
//      // float* h_target = multMatrix(h_input,batch,output_feat, 1000);
//      // float* target = PushArrayIntoGpu(h_target, dimensions_out);
//    float* target, * d_input;
//    image_loader.Next(&d_input);
//    label_loader.Next(&target);
//
//    // printf("Input:\n");
//     //printGpuArray(d_input, input_feat * batch, input_feat);
//
//    printf("Target:\n");
//    printGpuArray(target, output_feat * batch, 10);
//    model.forward(d_input);
//    //printf("\nbias:\n");
//    //float* h_linbias = printGpuArray(lin.d_bias, output_feat, output_feat);
//    //printf("\nWeights:\n");
//    //float* h_linweights = printGpuArray(lin.d_weight, input_feat*output_feat, input_feat);
//    printf("\nResults:\n");
//    float* h_output = printGpuArray(model.getOutput(), batch * output_feat, 10);
//    //float arr_h1[] = { 1,2,3,20,5,6,7 };
//    //            /* 1,2,3,4,5,6,7,
//    //             1,2,3,4,5,6,7,
//    //             1,2,3,4,5,6,7 };
//
//    //float* target = PushArrayIntoGpu(arr_h1, dimensions_out);
//
//    //printGpuArray(target, 7 * batch, 7);
//    float* d_loss = l1->forward(layers.at(layers.size() - 1)->getOutput(), target, output_feat, batch);
//    //printf("Loss:\n");
//    //printGpuArray(d_loss, batch, 1);
//    cudaFree(d_loss);
//    l1->backward(layers.at(layers.size() - 1)->getOutput(), target, d_grad, output_feat, batch);
//    //printf("OUT_PUT_GRAD:\n");
//    //printGpuArray(d_grad, output_feat *batch, output_feat);
//    float lr = 0.01;
//    layers.at(layers.size() - 1)->backward(layers.at(layers.size() - 2)->getOutput(), d_grad, lr);
//    for (size_t i = layers.size() - 1; i > 0; i--)
//    {
//        layers.at(i)->backward(layers.at(i - 1)->getOutput(), layers.at(i - 1)->getInputGrad(), lr);
//    }
//    layers.at(0)->backward(d_input, layers.at(1)->getInputGrad(), lr);
//    cudaDeviceSynchronize();
//}
//
//return 0;
