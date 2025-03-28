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


// Function to compute the average
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


int argmax(float* input,int size) {
    int index = 0;
    for (size_t i = 0; i < size; i++)
    {
        if (input[i]>input[index]) {
            index = i;
        }
    }
    return index;
}

int main() {
    float* train_images;
    float* train_labels;
    int num_train, img_size;



    read_mnist_images("t10k-images.idx3-ubyte", train_images, num_train, img_size);
    read_mnist_labels("t10k-labels.idx1-ubyte", train_labels, num_train);
    
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
    int output_feat = 10;
    //int hidden = 10;
    dim3 dimensions_in(1, batch, input_feat);  // 1x4x6 tensor
    dim3 dimensions_out(1, batch, output_feat);  // 1x4x6 tensor
    
   // float* d_input = PushArrayIntoGpu(arr_h, dimensions_in);
    DatasetLoader image_loader(num_train, batch, 28, 28, train_images);
    DatasetLoader label_loader(num_train, batch,1, output_feat, train_labels);

    std::vector<NNLayer*> layers;
    layers.push_back(new ConvLayer2D(batch, 1, 28, 28, 32, 3, 2, 1));
    layers.push_back(new ReLULayer(batch, 32, 14, 14));
    layers.push_back(new ConvLayer2D(batch, 32, 14, 14, 64, 3, 2, 1));
    layers.push_back(new ReLULayer(batch, 64, 7, 7));
    layers.push_back(new ConvLayer2D(batch, 64, 7, 7, 32, 3, 2, 1));
    layers.push_back(new ReLULayer(batch, 32, 4, 4));
    //layers.push_back(new ConvLayer2D(batch, 128, 4, 4, 32, 3, 2, 1));
    //layers.push_back(new ReLULayer(batch, 32, 2, 2));

    layers.push_back(new ConvLayer2D(batch, 32, 4, 4,64, 3,2 , 1));
    layers.push_back(new ReLULayer(batch, 64, 2, 2));

    //layers.push_back(new LinearLayer(batch, 128, 64));
    //layers.push_back(new ReLULayer(batch, 1, 1, 64));

    layers.push_back(new LinearLayer(batch, 64*4, 32));
    layers.push_back(new ReLULayer(batch, 1, 1, 32));

    layers.push_back(new LinearLayer(batch, 32, 10));
    layers.push_back(new ReLULayer(batch, 1, 1, 10));


    layers.push_back(new SoftmaxLayer(batch, 1, 1, 10));




    NNModel model(layers);
    LossFunction* l1 = new CrossEntropyLoss();

    cudaDeviceSynchronize();
    float* d_grad;
    cudaMalloc((void**) & d_grad, output_feat *batch*sizeof(float));

    
    float loss=0.0f;
    float accuracy = 0.0f;

    for (int i = 0; i < 10000; i++)
    {
        float* target, * d_input;
        image_loader.Next(&d_input);
        label_loader.Next(&target);

       /* float* h_input = createMatrix(batch, 10);
        float* d_input = PushArrayIntoGpu(h_input, dimensions_in);
        float* h_target = multMatrix(h_input,batch,output_feat, 10);
        float* target = PushArrayIntoGpu(h_target, dimensions_out);
        */

        //printf("\n%d iter:\n", i);
        
        model.forward(d_input);
        //cudaMemcpy(a, model.getOutput(), sizeof(float), cudaMemcpyDeviceToHost);


        float* d_loss = l1->forward(model.getOutput(), target, output_feat, batch);
        float tLoss = computeAverage(d_loss, batch, 1);
        loss += tLoss;;
        //printf("%dth Loss:%f\n",i, tLoss);printf("Target:\n");
        ///////////////////////////////////////////////////////////////////
        float* h_target = printGpuArray(target, output_feat * batch, 10, false);
        //printf("\nResults:\n");
        float* h_output = printGpuArray(model.getOutput(), batch * output_feat, 10, false);

        int correct = 0;
        for (int j = 0; j < batch; j++) {
            int pred_class = argmax(&h_output[j * output_feat], output_feat);  // Get index of max prob
            int true_class = argmax(&h_target[j * output_feat], output_feat);
            if (pred_class == true_class) correct++;
        }
        accuracy += (float)correct / batch ;
        //printf("%d\n", correct);
        //printf("Batch Accuracy: %.2f%%\n", ((float)correct*100.0f) / batch);


        if (i%(num_train/batch)==0 && i!=0)
        {
            printf("%d Epoch loss:%f\n", i / (num_train / batch), loss/ (num_train / batch));
            printf("Epoch Accuracy: %.2f%%\n", (accuracy*100.0f)/ (num_train / batch));

            loss = 0.0f;
            accuracy = 0.0f;
        }
        cudaFree(d_loss);
        float lr = 0.01;

        l1->backward(model.getOutput(), target, d_grad, output_feat, batch);
        model.backward(d_input,d_grad,lr);
        cudaDeviceSynchronize();
    }

    return 0;

}
