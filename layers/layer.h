/*****************************************************************************
 *  Layer Abstruction
 * 
 *  Copyright (C) 2021 Kellen.Xu  zx272@cornell.edu. 
 *
 *  @file   layer.h 
 *  @brief  This file defines abstraction of layer class type, can be 
 *          derivited by concrete layer class.        
 *  @author kellen xu 
 *****************************************************************************/

#ifndef __LAYER_H__
#define __LAYER_H__

#include "common.h"

class Layer
{
public:
    std::string type = "none";
    float* output;              // forward output of current layer
    float* d_output;            // gradient to the output of current layer           
    
    bool isPrevInput;
    float* prev_output;         // forward output of last layer
    float* prev_d_output;       // gradient to the output of last layer

    nn::shape_3d input_shape;   // input shape, 3-D 
    nn::shape_3d output_shape;  // output shape, 3-D

    Layer(){};
    virtual void forward() = 0;     // forward computation
    virtual void backward() = 0;    // backward computation
    virtual void clear() = 0;       // clear memory    
    void setInputLayer(Layer *prev_layer); // set the input layer
};

#endif