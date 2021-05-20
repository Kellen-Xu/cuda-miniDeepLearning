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
    //cudaError_t cudaStatus; 
    // allocate cuda memory
    cudaMalloc(&output,   sizeof(float) * output_fdim);
    cudaMalloc(&maxID,    sizeof(float) * maxID_fdim);
    cudaMalloc(&d_output, sizeof(float) * d_output_fdim);
}

Pooling::~Pooling()
{
    cudaFree(output);
    cudaFree(maxID);
    cudaFree(d_output);
}

void Pooling::forward()
{
    pl::forwardPooling<<<64,64>>>(prev_output, output, param);
}

__global__ void pl::forwardPooling(float *prev_output, float *output, pl::param param)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    const int win_sz = param.win_size;
    const int in_wd    = param.input.width;
    const int ot_wd    = param.output.width;

    float tmp  = 0.0;
    for(int i = 0; i < win_sz; i++) {
        for(int k = 0; k < win_sz; k++) {
            float val = prev_output[i*in_wd+pos*2+(pos/ot_wd)*in_wd+k];
            tmp  = (tmp > val) ? tmp : val;
        }
    }
    output[pos] = tmp; 
    __syncthreads();

}

__global__ void pl::forwardPoolingStoreID(float *prev_output, float *maxID, pl::param param)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    const int win_sz = param.win_size;
    const int in_wd    = param.input.width;
    const int ot_wd    = param.output.width;

    float tmp = 0.0;
    int tmp_id  = 0;
    for(int i = 0; i < win_sz; i++) {
        for(int k = 0; k < win_sz; k++) {
            float val = prev_output[i*in_wd+pos*2+(pos/ot_wd)*in_wd+k];
            tmp_id = (tmp > val) ? tmp_id : (i*in_wd+pos*2+(pos/ot_wd)*in_wd+k);
            tmp    = (tmp > val) ? tmp : val;
        }
    }
    maxID[pos]  = tmp_id;
    __syncthreads();
}

void Pooling::backward()
{
    pl::backwardPooling<<<64,64>>>(prev_d_output, d_output, maxID, param);
}

__global__ void pl::backwardPooling(float *prev_d_output, float *d_output, int *maxid, pl::param param)
{
    int pos  = blockIdx.x * blockDim.x + threadIdx.x;

    int id = maxid[pos];

    float val = prev_d_output[pos];

    d_output[id] = val;
}

void Pooling::clear()
{
    // clear forward part
    cudaMemset(output, 0x00, sizeof(float) * output_fdim);

    // clear backward part
    cudaMemset(maxID,    0x00, sizeof(float) * maxID_fdim);
    cudaMemset(d_output, 0x00, sizeof(float) * d_output_fdim);
}