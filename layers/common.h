/*****************************************************************************
 *  Common Utilities
 * 
 *  Copyright (C) 2021 Kellen.Xu  zx272@cornell.edu. 
 *
 *  @file   common.h 
 *  @brief  This file describe some common utilities.        
 *  @author kellen xu 
 *****************************************************************************/
#ifndef __COMMON_H__
#define __COMMON_H__

#include <string>
#include <cstdio>
#include <cuda.h>
#include <cublas.h>
#include"cublas_v2.h"


#define IDX3(a,b,c,A,B,C) (a)*(B)*(C)+(b)*(C)+c
#define INPUT_LAYER_TYPE "input"
#define CONV_LAYER_TYPE "conv"
#define POOLING_LAYER_TYPE "max_pooling"
#define SIGMOID_LAYER_TYPE "sigmoid"
#define SOFTMAX_LAYER_TYPE "softmax"

#define EPSILON 1e-6

namespace nn{
    /** 
     * @brief shape struct, 3 Dimension 
     */
    struct shape_3d
    {
        int width; 
        int height; 
        int channel;
    };

    /** 
     * @brief shape struct, 2 Dimension 
     */
    struct shape_2d
    {
        int width; 
        int height; 
    };


    /** 
     * @brief sample from uniform distribution of given range
     * @param 
     *      - float range: U[-range, range]
     * 
     * @return float type
     */ 
    float generateUniformRandom(float range);

   /** 
     * @brief apply gradient descent to learnable parameters
     * @param 
     *      - float* param: parameter vector to update
     *      - float* grad: gradient vector 
     *      - float  lr: learning rate
     */  
    void apply_grad(float* param, float* grad, float lr, int param_fdim);
    __global__ void cuda_apply_grad(float* param, float* grad, float lr, int N);
}

#endif