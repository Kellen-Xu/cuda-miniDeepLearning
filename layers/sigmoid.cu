#include "sigmoid.h"

Sigmoid::Sigmoid(nn::shape_3d input_shape)
{
    this->type = SIGMOID_LAYER_TYPE;
    this->input_shape = input_shape;
    this->output_shape = input_shape;
    
    this->output_fdim = output_shape.width*output_shape.height*output_shape.channel;

    printf("+ Sigmoid layer,  output_shape=(%d,%d,%d)\n",
            output_shape.width, output_shape.height, output_shape.channel);
    cudaMalloc(&output, sizeof(float) * output_fdim);
    cudaMalloc(&d_output, sizeof(float) * output_fdim);
}

Sigmoid::~Sigmoid()
{
    cudaFree(output);
    cudaFree(d_output); 
}

void Sigmoid::forward()
{
    sigmoid::forward<<<64,64>>>(output, prev_output, output_fdim);
}

__global__ void sigmoid::forward(float *output, float *prev_output, int output_fdim)
{
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int size = blockDim.x * gridDim.x;
    const int N = output_fdim;

    for (int i = N*pos/size; i<N*(pos+1)/size; ++i)
    {
        output[i] = 1/(1+exp(-prev_output[i]));
    }
}

void Sigmoid::backward()
{
    sigmoid::backward<<<64,64>>>(prev_d_output, d_output, output, output_fdim);
}

__global__ void sigmoid::backward(float *prev_d_output, float *d_output, float *output, int output_fdim)
{
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int size = blockDim.x * gridDim.x;
    const int N = output_fdim;

    for (int i = N*pos/size; i<N*(pos+1)/size; ++i)
    {
        prev_d_output[i] = d_output[i]*(output[i]*(1-output[i])+EPSILON);
    } 
}

void Sigmoid::clear()
{
    cudaMemset(output, 0x00, sizeof(float) * output_fdim);
    cudaMemset(d_output, 0x00, sizeof(float) * output_fdim);
}