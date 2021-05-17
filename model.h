/*****************************************************************************
 *  Model Definition 
 * 
 *  Copyright (C) 2021 Kellen.Xu  zx272@cornell.edu. 
 *
 *  @file   model.h 
 *  @brief  This file describe model structure and its API.        
 *  @author kellen xu 
 *****************************************************************************/
#ifndef __MODEL_H__
#define __MODEL_H__

#include "layers/common.h"
#include "layers/conv2d.h"
#include "layers/input.h"
#include "layers/sigmoid.h"
#include "layers/softmax.h"

class Model
{
    public:

    cublasHandle_t handle;

    /* Model Structure */
    nn::shape_3d image_shape = {28,28,1};
    Input input_layer = Input(image_shape);
    Conv2d conv_1 = Conv2d(input_layer.output_shape, 5,6,1);
    Sigmoid sigm_1 = Sigmoid(conv_1.output_shape);
    Conv2d conv_2 = Conv2d(sigm_1.output_shape, 4,6,4);
    Sigmoid sigm_2 = Sigmoid(conv_2.output_shape);
    Conv2d conv_3 = Conv2d(sigm_2.output_shape, 6,10,1);
    Sigmoid sigm_3 = Sigmoid(conv_3.output_shape);
    Softmax softmax_layer = Softmax(sigm_3.output_shape);

    float lr;

    Model(float lr)
    {
        this->lr = lr;
        cublasCreate(&handle);
    }

    ~Model(){};    

    /** 
     * @brief build model topology
     */
    void build()
    {
        conv_1.setInputLayer(&input_layer);
        sigm_1.setInputLayer(&conv_1);
        conv_2.setInputLayer(&sigm_1);
        sigm_2.setInputLayer(&conv_2);
        conv_3.setInputLayer(&sigm_2);
        sigm_3.setInputLayer(&conv_3);
        softmax_layer.setInputLayer(&sigm_3);
    }

    /** 
     * @brief clear intermediate results
     */
    void clear()
    {
        input_layer.clear();
        conv_1.clear();
        sigm_1.clear();
        conv_2.clear();
        sigm_2.clear();
        conv_3.clear();
        sigm_3.clear();
        softmax_layer.clear();
    }

    /** 
     * @brief feed sample
     */
    void feed(float* image, int label)
    {
        input_layer.feed(image);
        softmax_layer.feed(label);
    }

    /** 
     * @brief forward computation
     */
    void forward()
    {
        conv_1.forward();
        sigm_1.forward();
        conv_2.forward();
        sigm_2.forward();
        conv_3.forward();
        sigm_3.forward();
        softmax_layer.forward();
    }

    /** 
     * @brief back propagation and update parameter
     */
    void backward()
    {
        softmax_layer.backward();
        sigm_3.backward();
        conv_3.backward();
        sigm_2.backward();
        conv_2.backward();
        sigm_1.backward();
        conv_1.backward();

        // update parameter
        nn::apply_grad(conv_1.weight, conv_1.d_weight, lr*10, conv_1.weight_fdim);
        nn::apply_grad(conv_1.bias, conv_1.d_bias, lr, conv_1.bias_fdim);
        
        nn::apply_grad(conv_2.weight, conv_2.d_weight, lr*5, conv_2.weight_fdim);
        nn::apply_grad(conv_2.bias, conv_2.d_bias, lr, conv_2.bias_fdim);

        nn::apply_grad(conv_3.weight, conv_3.d_weight, lr*2, conv_3.weight_fdim);
        nn::apply_grad(conv_3.bias, conv_3.d_bias, lr, conv_3.bias_fdim);
    }

    /** 
     * @brief get model predict loss
     */
    float getLoss()
    {
        return softmax_layer.computeLoss();
    }

    /** 
     * @brief predict class of given sample
     */
    int predict(float *image)
    {
        clear();
        feed(image, NULL);
        forward();

        int predLabel = 0;
        float outputVec[10];

        cudaMemcpy(outputVec, softmax_layer.output, sizeof(float) * 10, cudaMemcpyDeviceToHost);

        for (int i = 1; i < 10; ++i) {
            if (outputVec[predLabel] < outputVec[i]) {
                predLabel = i;
            }
        }

        return predLabel;
    }
};

#endif