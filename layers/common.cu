#include "common.h"

/** 
 * @brief sample from a uniform distribution
 */
float nn::generateUniformRandom(float range)
{
    return range - float(rand()) / float(RAND_MAX);
}

// void nn::apply_grad(float* param, float* grad, float lr, int param_fdim, cublasHandle_t handle)
// {
//     float nlr = -lr;

//     cublasStatus_t stat = cublasSaxpy(handle, param_fdim, &nlr, grad, 1, param, 1);
//     if(stat != CUBLAS_STATUS_SUCCESS)
//     {
//         printf("cublasSaxpy failed\n");
//     }
// }

void nn::apply_grad(float* param, float* grad, float lr, int param_fdim)
{

    cuda_apply_grad<<<64,64>>>(param, grad, lr, param_fdim);
}

__global__ void nn::cuda_apply_grad(float* param, float* grad, float lr, int N)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
	int size = blockDim.x * gridDim.x;

	for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
		param[idx] -=  lr * grad[idx];
	}
}
