#ifndef __POOLING_H__
#define __POOLING_H__

#include "layer.h"

namespace pl
{
    struct param
    {
        nn::shape_3d input;
        nn::shape_3d output;
        int win_size;
        int stride;
    };

    // forward
    //__global__ void forwardPooling(float *pre_ouput, float *output, pl::param param);

    // backward
};

/**
 * @brief Pooling Layer
 * 
 */

class Pooling : public Layer
{
private:
    int win_size;  // Pooling Window Size; Usually it's 2x2
    int stride;    // stride, the step length that kernel moves in both dimension
                   // Usually equal to the win_size

    pl::param param;

public:

    int *maxID;
    int output_fdim;
    int d_output_fdim;
    int maxID_fdim;
    

    Pooling(nn::shape_3d input_shape, int pl_win_size, int win_num, int stride, bool verbose = true);
    ~Pooling();

    virtual void forward();
    virtual void backward();
    virtual void clear();


};

#endif