/*****************************************************************************
 *  Softmax Loss Layer
 * 
 *  Copyright (C) 2021 Kellen.Xu  zx272@cornell.edu. 
 *
 *  @file   softmax.h 
 *  @brief  This file defines softmax layer for multi classification loss
 *  @author kellen xu 
 *****************************************************************************/
#ifndef __SOFTMAX_H__
#define __SOFTMAX_H__

#include "layer.h"

namespace softmax{
    __global__ void forwardSum(float *expSum, float *prev_output, int output_fdim);
    __global__ void forwardNorm(float *output, float *prev_output, float *expSum, int output_fdim);
    __global__ void backward(float *prev_d_output, float *output, int label, int output_fdim);
}

class Softmax : public Layer
{
public:
    float *expSum;
    float *loss;
    int label;
    int output_fdim;

    Softmax(nn::shape_3d input_shape);
    ~Softmax();

    /**
     *@brief feed label of a sample
     */
    void feed(int label);

    /**
     *@brief compute cross-entropy loss
     */
    float computeLoss();

    virtual void forward();
    virtual void backward();
    virtual void clear();
};

#endif