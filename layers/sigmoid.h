/*****************************************************************************
 *  Sigmoid Activation Layer
 * 
 *  Copyright (C) 2021 Kellen.Xu  zx272@cornell.edu. 
 *
 *  @file   sigmoid.h 
 *  @brief  This file defines sigmoid activation function layer class type.        
 *  @author kellen xu 
 *****************************************************************************/
#ifndef __SIGMOID_H__
#define __SIGMOID_H__

#include "layer.h"

namespace sigmoid
{
__global__ void forward(float *output, float *prev_output, int output_fdim);
__global__ void backward(float *prev_d_output, float *d_output, float *output, int output_fdim);
}

class Sigmoid : public Layer
{
public:
    int output_fdim;

    Sigmoid(nn::shape_3d input_shape);
    ~Sigmoid();

    virtual void forward();
    virtual void backward();
    virtual void clear();
};

#endif