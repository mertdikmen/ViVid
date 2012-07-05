#include "../src/DeviceMatrix.hpp"
//#include "../src/PairwiseDistanceLocal.hpp"


//static const unsigned int BLOCK_SIZE = 16;

//DeviceMatrixCL a, DeviceMatrixCL b,DeviceMatrixCL out, const int type
											  
/**
 * @note This kernel is based on the blocked matrix multiply.  We
 * expect to be caleld with blockDim(BLOCK_SIZE, BLOCK_SIZE) and a
 * sufficiently large grid to cover all othe output values.
 */
__kernel void pairwiseDistanceKernelGeneric(__global DeviceMatrixCL a, __global DeviceMatrixCL b,__global DeviceMatrixCL out, const int type)
{
    /*const int out_ry = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    const int out_cx = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    // Index used for reading b.  We use the fact that our thread
    // blocks are square here.
    const int b_ry = blockIdx.x * BLOCK_SIZE + threadIdx.y;

    __shared__ float a_cache[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float b_cache[BLOCK_SIZE][BLOCK_SIZE];

    float dst = 0;
    for (unsigned int i=0; i < a.width; i+=BLOCK_SIZE) {
        int read_cx = i + threadIdx.x;
        if (read_cx < a.width) {
            if (out_ry < a.height) {
                a_cache[threadIdx.y][threadIdx.x] =
                       *getPtr(a, out_ry, read_cx);
            }
            if (b_ry < b.height) {
                b_cache[threadIdx.y][threadIdx.x] =
                       *getPtr(b, b_ry, read_cx);
            }
        }
        __syncthreads();
        int end = min(BLOCK_SIZE, (unsigned int)(a.width - i));
        for (int k=0; k < end; k++) {
            if (type == EUCLIDEAN){
                float diff = a_cache[threadIdx.y][k] - b_cache[threadIdx.x][k];
                dst += diff * diff;
            }
            else if (type == DOTPRODUCT){
                dst += a_cache[threadIdx.y][k] * b_cache[threadIdx.x][k];
            }
            else if (type == ABSDOTPRODUCT){
                dst += a_cache[threadIdx.y][k] * b_cache[threadIdx.x][k];
            }
            else if (type == CHISQUARED){
                float diff, sum;
                diff = a_cache[threadIdx.y][k] - b_cache[threadIdx.x][k];
                sum  = a_cache[threadIdx.y][k] + b_cache[threadIdx.x][k];
                dst += diff * diff / sum;
            }
            else if (type == CITYBLOCK){
                dst += fabs(a_cache[threadIdx.y][k] - b_cache[threadIdx.x][k]);
            }
        }
        __syncthreads();
    }

    if ((out_cx < out.width) & (out_ry < out.height)) {
        if (type == ABSDOTPRODUCT){
            *getPtr(out, out_ry, out_cx) = abs(dst);
        }
        else {
            *getPtr(out, out_ry, out_cx) = dst;
        }
    }
	*/
}