#include "PairwiseDistance.hpp"
#include "PairwiseDistanceLocal.hpp"

static __device__ float* getPtr(const DeviceMatrix mat, unsigned int r,
                         unsigned int c)
{
    return mat.data + r*mat.pitch + c;
}

/**
 * @note This version does not coalesing memory reads *at all*, but
 * should be a simple first pass.
 */
/* Unused - leaving here for reference
static __global__ void pairwiseDistanceKernel(DeviceMatrix a, DeviceMatrix b,
                                       DeviceMatrix out)
{
    int ry = blockIdx.y * blockDim.y + threadIdx.y;
    int cx = blockIdx.x * blockDim.x + threadIdx.x;

    if ((cx < out.width) & (ry < out.height)) {
        float dst = 0;
        for (int i=0; i < a.width; i++) {
            float diff = *getPtr(a, ry, i) - *getPtr(b, cx, i);
            dst += diff * diff;
        }
        *getPtr(out, ry, cx) = dst;
    }
}
*/

static const unsigned int BLOCK_SIZE = 16;

/**
 * @note This kernel is based on the blocked matrix multiply.  We
 * expect to be caleld with blockDim(BLOCK_SIZE, BLOCK_SIZE) and a
 * sufficiently large grid to cover all othe output values.
 */
__global__ void pairwiseDistanceKernelGeneric(DeviceMatrix a, DeviceMatrix b,
                                              DeviceMatrix out, const int type)
{
    const int out_ry = blockIdx.y * BLOCK_SIZE + threadIdx.y;
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
//            if (type == EUCLIDEAN){
//                float diff = a_cache[threadIdx.y][k] - b_cache[threadIdx.x][k];
//                dst += diff * diff;
//            }
//            else if (type == DOTPRODUCT){
                dst += a_cache[threadIdx.y][0] * b_cache[threadIdx.x][0];
                dst += a_cache[threadIdx.y][1] * b_cache[threadIdx.x][1];
                dst += a_cache[threadIdx.y][2] * b_cache[threadIdx.x][2];
                dst += a_cache[threadIdx.y][3] * b_cache[threadIdx.x][3];
                dst += a_cache[threadIdx.y][4] * b_cache[threadIdx.x][4];
                dst += a_cache[threadIdx.y][5] * b_cache[threadIdx.x][5];
                dst += a_cache[threadIdx.y][6] * b_cache[threadIdx.x][6];
                dst += a_cache[threadIdx.y][7] * b_cache[threadIdx.x][7];
                dst += a_cache[threadIdx.y][8] * b_cache[threadIdx.x][8];
//            }
//            else if (type == ABSDOTPRODUCT){
//                dst += a_cache[threadIdx.y][k] * b_cache[threadIdx.x][k];
//            }
//            else if (type == CHISQUARED){
//                float diff, sum;
//                diff = a_cache[threadIdx.y][k] - b_cache[threadIdx.x][k];
//                sum  = a_cache[threadIdx.y][k] + b_cache[threadIdx.x][k];
//                dst += diff * diff / sum;
//            }
//            else if (type == CITYBLOCK){
//                dst += fabs(a_cache[threadIdx.y][k] - b_cache[threadIdx.x][k]);
//            }
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
}


void pwdist_generic( const DeviceMatrix* features_train,
                     const DeviceMatrix* features_test,
                     DeviceMatrix* output,
                     int type) {

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    int grid_ry = (features_train->height-1) / dimBlock.y + 1;
    int grid_cx = (features_test->height-1) / dimBlock.x + 1;
    dim3 dimGrid(grid_cx, grid_ry);

    pairwiseDistanceKernelGeneric<<<dimGrid, dimBlock>>>(*features_train,
                                                         *features_test,
                                                         *output,
                                                         type);
    cudaThreadSynchronize();
}



/**
 * @note This is a simple first pass that does nothing to manage
 * memory access
 */
__global__ void argminKernel(DeviceMatrix matrix, DeviceMatrix output)
{
    int ry = blockIdx.y * blockDim.y + threadIdx.y;

    if (ry < matrix.height) {
        int argmin = 0;
    	float minval = *getPtr(matrix, ry, 0);

        for (int cx=1; cx < matrix.width; cx++) {
            float val = *getPtr(matrix, ry, cx);
            if (val < minval) {
                argmin = cx;
                minval = val;
            }
        }

        *getPtr(output, ry, 0) = argmin;
    }
}

void argmin_cuda_local(const DeviceMatrix* matrix, DeviceMatrix* output)
{
    dim3 dimBlock(1,256);
    int grid_ry = (matrix->height-1) / dimBlock.y + 1;
    int grid_cx = 1;
    dim3 dimGrid(grid_cx, grid_ry);

    argminKernel<<<dimGrid, dimBlock>>>(*matrix, *output);
}

__global__ void argmaxKernel(DeviceMatrix matrix, DeviceMatrix output)
{
    int ry = blockIdx.y * blockDim.y + threadIdx.y;

    if (ry < matrix.height) {
        int argmax = 0;
    	float maxval = *getPtr(matrix, ry, 0);

        for (int cx=1; cx < matrix.width; cx++) {
            float val = *getPtr(matrix, ry, cx);
            if (val > maxval) {
                argmax = cx;
                maxval = val;
            }
        }

        *getPtr(output, ry, 0) = argmax;
    }
}

void argmax_cuda_local(const DeviceMatrix* matrix, DeviceMatrix* output)
{
    dim3 dimBlock(1,256);
    int grid_ry = (matrix->height-1) / dimBlock.y + 1;
    int grid_cx = 1;
    dim3 dimGrid(grid_cx, grid_ry);

    argmaxKernel<<<dimGrid, dimBlock>>>(*matrix, *output);
}


__global__ void minKernel(DeviceMatrix matrix, DeviceMatrix output)
{
    int ry = blockIdx.y * blockDim.y + threadIdx.y;

    if (ry < matrix.height) {
    	float minval = *getPtr(matrix, ry, 0);

        for (int cx=1; cx < matrix.width; cx++) {
            float val = *getPtr(matrix, ry, cx);
            if (val < minval) {
                minval = val;
            }
        }

        *getPtr(output, ry, 0) = minval;
    }
}

void min_cuda_local(const DeviceMatrix* matrix, DeviceMatrix* output)
{
    dim3 dimBlock(1,256);
    int grid_ry = (matrix->height-1) / dimBlock.y + 1;
    int grid_cx = 1;
    dim3 dimGrid(grid_cx, grid_ry);

    minKernel<<<dimGrid, dimBlock>>>(*matrix, *output);
}

__global__ void maxKernel(DeviceMatrix matrix, DeviceMatrix output)
{
    int ry = blockIdx.y * blockDim.y + threadIdx.y;

    if (ry < matrix.height) {
    	float maxval = *getPtr(matrix, ry, 0);

        for (int cx=1; cx < matrix.width; cx++) {
            float val = *getPtr(matrix, ry, cx);
            if (val > maxval) {
                maxval = val;
            }
        }

        *getPtr(output, ry, 0) = maxval;
    }
}

void max_cuda_local(const DeviceMatrix* matrix, DeviceMatrix* output)
{
    dim3 dimBlock(1,256);
    int grid_ry = (matrix->height-1) / dimBlock.y + 1;
    int grid_cx = 1;
    dim3 dimGrid(grid_cx, grid_ry);

    maxKernel<<<dimGrid, dimBlock>>>(*matrix, *output);
}

