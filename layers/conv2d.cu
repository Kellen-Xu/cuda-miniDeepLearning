#include "conv2d.h"

Conv2d::Conv2d(nn::shape_3d input_shape, int kernel_size, int kernel_num, int stride, bool verbose)
{
    this->type = CONV_LAYER_TYPE;
    //-------------------------------------------------------------------
    // Initialize member variables
    //-------------------------------------------------------------------
    // input shape
    this->input_shape = input_shape;
    // output shape
    const int output_width = (input_shape.width - kernel_size) / stride + 1;
    const int output_height = (input_shape.height - kernel_size) / stride + 1;
    const int output_channel = kernel_num;
    this->output_shape = {output_width, output_height, output_channel};
    // others ..
    this->kernel_size = kernel_size;
    this->stride = stride;
    this->weight_fdim = kernel_size * kernel_size * kernel_num;
    this->bias_fdim = kernel_num;
    this->output_fdim = output_width * output_height * output_channel;
    this->param = {input_shape, output_shape, kernel_size, stride};
    // print shape info to the console
    if (verbose)
        printf("+ Conv Layer,   input_shape=(%d,%d,%d), output_shape=(%d,%d,%d)\n",
               input_shape.width, input_shape.height, input_shape.channel,
               output_shape.width, output_shape.height, output_shape.channel);

    //-------------------------------------------------------------------
    // Initialize CUDA device vectors
    //-------------------------------------------------------------------
    cudaError_t cudaStatus; 
    // allocate cuda memory
    cudaMalloc(&weight, sizeof(float) * weight_fdim);
    cudaMalloc(&d_weight, sizeof(float) * weight_fdim);
    cudaMalloc(&bias, sizeof(float) * bias_fdim);
    cudaMalloc(&d_bias, sizeof(float) * bias_fdim);

    cudaMalloc(&output, sizeof(float) * output_fdim);
    cudaMalloc(&d_output, sizeof(float) * output_fdim);

    // initialize weight and bias
    float h_weight[weight_fdim];
    float h_bias[bias_fdim];

    for (int i = 0; i < weight_fdim; i++)
    {
        h_weight[i] = nn::generateUniformRandom(0.5f);
        // h_weight[i] = 1;
    }
    cudaStatus = cudaMemcpy(weight, h_weight, sizeof(float) * weight_fdim, cudaMemcpyHostToDevice);
    if (cudaSuccess != cudaStatus)
        throw "Failed: cannot copy memory for Convlayer weight\n";
    
    for (int i = 0; i < bias_fdim; i++)
    {
        h_bias[i] = nn::generateUniformRandom(0.5f);
        // h_bias[i] = 0;
    }
    cudaStatus = cudaMemcpy(bias, h_bias, sizeof(float) * bias_fdim, cudaMemcpyHostToDevice);
    if (cudaSuccess != cudaStatus)
        throw "Failed: cannot copy memory for Convlayer bias\n";
}

Conv2d::~Conv2d()
{
    cudaFree(weight);
    cudaFree(d_weight);
    cudaFree(bias);
    cudaFree(d_bias);

    cudaFree(output);
    cudaFree(d_output);
}
/*  -- Begin of Forward --  */
void Conv2d::forward()
{
    // Convolution
    conv::forwardConv<<<64,64>>>(prev_output, output, weight, param);
    // Add Bias
    conv::forwardAddBias<<<64,64>>>(output, bias, param);
}

__global__ void conv::forwardConv(float *prev_output, float *output, float *weight, conv::param param)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    int size = blockDim.x * gridDim.x;

    const int in_wd = param.input.width;
    const int in_ht = param.input.height;
    const int in_cn = param.input.channel;
    const int ot_wd = param.output.width;
    const int ot_ht = param.output.height;
    const int ot_cn = param.output.channel;
    const int k_sz = param.kernel_size;
    const int sd = param.stride;

    const int N = k_sz * k_sz * in_cn * ot_cn * ot_ht * ot_wd;

    for (int n = N * pos / size; n < N * (pos + 1) / size; ++n)
    {
        int idx = n;
        const int i1 = ((idx /= 1) % k_sz);
        const int i2 = ((idx /= k_sz) % k_sz);
        const int i3 = ((idx /= k_sz) % in_cn);
        const int i4 = ((idx /= in_cn) % ot_cn);
        const int i5 = ((idx /= ot_cn) % ot_ht);
        const int i6 = ((idx /= ot_ht) % ot_wd);

        float* output_elem = &output[IDX3(i4,i6,i5, ot_cn,ot_wd,ot_ht)];
        float weight_elem = weight[IDX3(i4,i2,i1, ot_cn,k_sz,k_sz)];
        float input_elem = prev_output[IDX3(i3,(i6*sd+i2),(i5*sd+i1), in_cn,in_wd,in_ht)];
        atomicAdd(output_elem, weight_elem * input_elem);
    }
}

__global__ void conv::forwardAddBias(float* output, float* bias, conv::param param)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    int size = blockDim.x * gridDim.x;

    const int ot_wd = param.output.width;
    const int ot_ht = param.output.height;
    const int ot_cn = param.output.channel;
    
    const int N = ot_cn * ot_ht * ot_wd;

    for (int n = N * pos / size; n < N * (pos + 1) / size; ++n)
    {
        int idx = n;
        const int i1 = ((idx /= 1) % ot_ht);
        const int i2 = ((idx /= ot_ht) % ot_wd);
        const int i3 = ((idx /= ot_wd) % ot_cn);

        float* output_elem = &output[IDX3(i3,i2,i1, ot_cn,ot_wd,ot_ht)];
        float bias_elem = bias[i3];
        
        *output_elem += bias_elem;
    }
}
/*   -- End of Forward --   */
void Conv2d::backward()
{
    // Gradient to weight
    conv::backwardGradientWeight<<<64,64>>>(d_weight, d_output, prev_output, param);
    // Gradient to bias
    conv::backwardGradientBias<<<64,64>>>(d_bias, d_output, param);
    // Gradient to input
    conv::backwardGradientPrev<<<64,64>>>(prev_d_output, weight, d_output, param);
}

__global__ void conv::backwardGradientWeight(float *d_weight, float *d_output, float *prev_output, conv::param param)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    int size = blockDim.x * gridDim.x;

    const int in_wd = param.input.width;
    const int in_ht = param.input.height;
    const int in_cn = param.input.channel;
    const int ot_wd = param.output.width;
    const int ot_ht = param.output.height;
    const int ot_cn = param.output.channel;
    const int k_sz = param.kernel_size;
    const int sd = param.stride;

    const int N = k_sz * k_sz * in_cn * ot_cn * ot_ht * ot_wd;

    for (int n = N * pos / size; n < N * (pos + 1) / size; ++n)
    {
        int idx = n;
        const int i1 = ((idx /= 1) % k_sz);
        const int i2 = ((idx /= k_sz) % k_sz);
        const int i3 = ((idx /= k_sz) % in_cn);
        const int i4 = ((idx /= in_cn) % ot_cn);
        const int i5 = ((idx /= ot_cn) % ot_ht);
        const int i6 = ((idx /= ot_ht) % ot_wd);

        float* d_weight_elem = &d_weight[IDX3(i4,i2,i1, ot_cn,k_sz,k_sz)];
        float d_output_elem = d_output[IDX3(i4,i6,i5, ot_cn,ot_wd,ot_ht)];
        float input_elem = prev_output[IDX3(i3,(i6*sd+i2),(i5*sd+i1), in_cn,in_wd,in_ht)];

        atomicAdd(d_weight_elem, d_output_elem * input_elem);
    } 
}

__global__ void conv::backwardGradientBias(float *d_bias, float *d_output, conv::param param)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    int size = blockDim.x * gridDim.x;

    const int ot_wd = param.output.width;
    const int ot_ht = param.output.height;
    const int ot_cn = param.output.channel;
    
    const int N = ot_cn * ot_ht * ot_wd;

    for (int n = N * pos / size; n < N * (pos + 1) / size; ++n)
    {
        int idx = n;
        const int i1 = ((idx /= 1) % ot_ht);
        const int i2 = ((idx /= ot_ht) % ot_wd);
        const int i3 = ((idx /= ot_wd) % ot_cn);

        float* d_bias_elem = &d_bias[i3];
        float d_output_elem = d_output[IDX3(i3,i2,i1, ot_cn,ot_wd,ot_ht)];
        
        atomicAdd(d_bias_elem, d_output_elem);
    } 
}

__global__ void conv::backwardGradientPrev(float *prev_d_output, float *weight, float *d_output, conv::param param)
{
   int pos = blockIdx.x * blockDim.x + threadIdx.x;
    int size = blockDim.x * gridDim.x;

    const int in_wd = param.input.width;
    const int in_ht = param.input.height;
    const int in_cn = param.input.channel;
    const int ot_wd = param.output.width;
    const int ot_ht = param.output.height;
    const int ot_cn = param.output.channel;
    const int k_sz = param.kernel_size;
    const int sd = param.stride;

    const int N = k_sz * k_sz * in_cn * ot_cn * ot_ht * ot_wd;

    for (int n = N * pos / size; n < N * (pos + 1) / size; ++n)
    {
        int idx = n;
        const int i1 = ((idx /= 1) % k_sz);
        const int i2 = ((idx /= k_sz) % k_sz);
        const int i3 = ((idx /= k_sz) % in_cn);
        const int i4 = ((idx /= in_cn) % ot_cn);
        const int i5 = ((idx /= ot_cn) % ot_ht);
        const int i6 = ((idx /= ot_ht) % ot_wd);

        float* prev_d_ouput_elem = &prev_d_output[IDX3(i3,(i6*sd+i2),(i5*sd+i1), in_cn,in_wd,in_ht)];
        float d_output_elem = d_output[IDX3(i4,i6,i5, ot_cn,ot_wd,ot_ht)];
        float weight_elem = weight[IDX3(i4,i2,i1, ot_cn,k_sz,k_sz)];

        atomicAdd(prev_d_ouput_elem, d_output_elem * weight_elem);
    } 
}

void Conv2d::clear()
{   
    // forward part
    cudaMemset(output, 0x00, sizeof(float) * output_fdim);
    // backward part
    cudaMemset(d_output, 0x00, sizeof(float) * output_fdim);
    cudaMemset(d_weight, 0x00, sizeof(float) * weight_fdim);
    cudaMemset(d_bias, 0x00, sizeof(float) * bias_fdim);
}