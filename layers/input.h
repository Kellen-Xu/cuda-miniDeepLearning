/*****************************************************************************
 *  Input Layer
 * 
 *  Copyright (C) 2021 Kellen.Xu  zx272@cornell.edu. 
 *
 *  @file   input.h 
 *  @brief  This file defines the input layer.        
 *  @author kellen xu 
 *****************************************************************************/
#ifndef __INPUT_H__
#define __INPUT_H__

#include "layer.h"

class Input : public Layer
{
public:
    int output_fdim;

    Input(nn::shape_3d input_shape);
    ~Input();
    void feed(float* sample);
    virtual void forward();
    virtual void backward();
    virtual void clear();
};

#endif