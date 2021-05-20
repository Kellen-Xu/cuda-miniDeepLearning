#include "pooling.h"

Pooling::Pooling(nn::shape_3d input_shape, int pl_win_size, int win_num, int stride, bool verbose)
{
    this->type = POOLING_LAYER_TYPE;

    //-------------------------------------------------------------------
    // Initialize member variables
    //-------------------------------------------------------------------
    // input shape
    this->input_shape = input_shape;

    // output shape
    const int output_width   = input_shape.width / pl_win_size;
    const int output_height  = input_shape.height / pl_win_size;
    const int output_channel = win_num;
    this->output_shape = {output_width, output_height, output_channel};

    // others
    this->win_size = pl_win_size;
    this->stride   = stride;
    this->output_fdim   = output_width * output_height * output_channel;
    this->d_output_fdim = input_shape.width * input_shape.height * input_shape.channel;
    this->maxID_fdim    = output_width * output_height * output_channel; 

    if (verbose)
        printf("+ Pooling Layer,  input_shape=(%d,%d,%d), output_shape=(%d,%d,%d)\n",
               input_shape.width, input_shape.height, input_shape.channel,
               output_shape.width, output_shape.height, output_shape.channel);

    //-------------------------------------------------------------------
    // Initialize CUDA device vectors
    //-------------------------------------------------------------------
    cudaError_t cudaStatus; 
    // allocate cuda memory
    cudaMalloc(&output,   sizeof(float) * output_fdim);
    cudaMalloc(&maxID,    sizeof(float) * maxID_fdim);
    cudaMalloc(&d_output, sizeof(float) * d_output_fdim);
}

Pooling::~Pooling()
{
    cudaFree(ouput);
    cudaFree(maxID);
    cudaFree(d_output);
}

void Pooling::clear()
{
    // clear forward part
    cudaMemset(output, 0x00, sizeof(float) * output_fdim);

    // clear backward part
    cudaMemset(maxID,    0x00, sizeof(float) * maxID_fdim);
    cudaMemset(d_output, 0x00, sizeof(float) * d_output_fdim);
}