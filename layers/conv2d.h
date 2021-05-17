/*****************************************************************************
 *  Convolutional Layer
 * 
 *  Copyright (C) 2021 Kellen.Xu  zx272@cornell.edu. 
 *
 *  @file   conv2d.h 
 *  @brief  This file defines convolutional layer class type.        
 *  @author kellen xu 
 *****************************************************************************/
#ifndef __CONV2D_H__
#define __CONV2D_H__

#include "layer.h"

namespace conv
{
    struct param
    {
        nn::shape_3d input;
        nn::shape_3d output;
        int kernel_size;
        int stride;
    };

    // forward
    __global__ void forwardConv(float *prev_output, float *output, float *weight, conv::param param);
    __global__ void forwardAddBias(float *output, float *bias, conv::param param);
    // backward
    __global__ void backwardGradientWeight(float *d_weight, float *d_output, float *prev_output, conv::param param);
    __global__ void backwardGradientBias(float *d_bias, float *d_output, conv::param param);
    __global__ void backwardGradientPrev(float *prev_d_output, float *weight, float *d_output, conv::param param);
}

/**
 * @brief Convolutional Layer 
 * 
 */
class Conv2d : public Layer
{
private:
    int kernel_size; // kernel size, the kernel should be (kernel_size, kernel_size) matrix
    int stride;      // stride, the step length that kernel moves in both dimension
    conv::param param;

public:
    float *weight;
    float *bias;

    float *d_weight; // gradient to the weight
    float *d_bias;   // gradient to the bias

    int weight_fdim; // flattened dimension of weight/kernel
    int bias_fdim;   // flattened dimension of bias
    int output_fdim; // flattened dimension of output

    /** 
     * @brief Convolutional Layer 
     * @param 
     *      - nn::shape_3d  input_shape
     *      - int           kernel_size
     *      - int           kernel_num
     *      - int           stride 
     */
    Conv2d(nn::shape_3d input_shape, int kernel_size, int kernel_num, int stride, bool verbose = true);
    ~Conv2d();
    virtual void forward();
    virtual void backward();
    virtual void clear();
};

#endif