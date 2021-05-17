#include "softmax.h"

Softmax::Softmax(nn::shape_3d input_shape)
{
    this->type = SOFTMAX_LAYER_TYPE;
    this->input_shape = input_shape;
    this->output_shape = input_shape;

    this->output_fdim = output_shape.width * output_shape.height * output_shape.channel;

    printf("+ Softmax layer,  output_shape=(%d,%d,%d)\n",
            output_shape.width, output_shape.height, output_shape.channel);
    cudaMalloc(&expSum, sizeof(float));
    cudaMalloc(&loss, sizeof(float));
    cudaMalloc(&output, sizeof(float) * output_fdim);
}

Softmax::~Softmax()
{
    cudaFree(expSum);
    cudaFree(loss);
    cudaFree(output);
};

void Softmax::feed(int label)
{
    this->label = label;
}

float Softmax::computeLoss()
{
    // fetch probability
    float prob[output_fdim];
    cudaMemcpy(prob, output, sizeof(float) * output_fdim, cudaMemcpyDeviceToHost);
    return -log(prob[label]);
}

/*      Forward      */
void Softmax::forward()
{
    softmax::forwardSum<<<1,output_fdim>>>(expSum, prev_output, output_fdim);
    softmax::forwardNorm<<<1,output_fdim>>>(output, prev_output, expSum, output_fdim);
}

__global__ void softmax::forwardSum(float *expSum, float *prev_output, int output_fdim)
{
    int tid = threadIdx.x;
    if(tid < output_fdim)
    {
        float expLogit = exp(prev_output[tid]);
        atomicAdd(expSum, expLogit);
    }
}

__global__ void softmax::forwardNorm(float *output, float *prev_output, float *expSum, int output_fdim)
{
    int tid = threadIdx.x;
    if(tid < output_fdim)
    {
        output[tid] = exp(prev_output[tid])/(*expSum+EPSILON);
    }
}


/*      Backward      */
void Softmax::backward()
{
    softmax::backward<<<1,output_fdim>>>(prev_d_output, output, label, output_fdim);
}

__global__ void softmax::backward(float *prev_d_output, float *output, int label, int output_fdim)
{
    int tid = threadIdx.x;
    if(tid < output_fdim) 
    {
        prev_d_output[tid] = ((label==tid ? -1.0f : 0.0f) + output[tid]);
    }        
}

void Softmax::clear()
{
    cudaMemset(loss, 0x00, sizeof(float));
    cudaMemset(expSum, 0x00, sizeof(float));
    cudaMemset(output, 0x00, sizeof(float)*output_fdim);
};