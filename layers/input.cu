#include "input.h"

Input::Input(nn::shape_3d input_shape)
{
    this->type = INPUT_LAYER_TYPE;
    this->input_shape = input_shape;
    this->output_shape = input_shape;

    this->output_fdim = output_shape.width*output_shape.height*output_shape.channel;

    printf("+ Input layer,  output_shape=(%d,%d,%d)\n",
            output_shape.width, output_shape.height, output_shape.channel);
    cudaMalloc(&output, sizeof(float) * output_fdim);
    cudaMalloc(&d_output, sizeof(float) * output_fdim);
}

Input::~Input()
{
    cudaFree(output);
    cudaFree(d_output);
}

void Input::feed(float* sample)
{
    cudaError_t cudaStatus; 
    cudaStatus = cudaMemcpy(output, sample, sizeof(float) * output_fdim, cudaMemcpyHostToDevice);
    if (cudaSuccess != cudaStatus)
    {
        throw "Failed: cannot copy memory in Inputlayer feed\n";
    }
}

void Input::forward(){};
void Input::backward(){};
void Input::clear()
{
    cudaMemset(output, 0x00, sizeof(float) * output_fdim);
    cudaMemset(d_output, 0x00, sizeof(float) * output_fdim);
}






