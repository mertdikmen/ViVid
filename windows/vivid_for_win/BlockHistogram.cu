#include "BlockHistogramLocal.hpp"

#define MAX_HISTOGRAM_SIZE 500

static __device__ float* getPtr(const DeviceMatrix mat, unsigned int r,
                         unsigned int c)
{
    return mat.data + r*mat.pitch + c;
}

static __device__ float* getPtr(const DeviceMatrix3D mat, unsigned int x0,
    unsigned int x1, unsigned int x2)
{
    return mat.data + x0*mat.pitch_t + x1 * mat.pitch_y + x2;
}

#ifdef METHOD_1

__global__ void cellHistogramKernel(DeviceMatrix3D histogram,
    DeviceMatrix assignments, DeviceMatrix weights,
    const int start_y, const int start_x,
    const int max_bin)
{
    const float aval = *getPtr(
        assignments,
        start_y + blockDim.y * blockIdx.y + threadIdx.y,
        start_x + blockDim.x * blockIdx.x + threadIdx.x);

    const float wval = *getPtr(
        weights,
        start_y + blockDim.y * blockIdx.y + threadIdx.y,
        start_x + blockDim.x * blockIdx.x + threadIdx.x);

    __shared__ float block_hist_cache[MAX_HISTOGRAM_SIZE];
    
    int bin_count = 0;
    int thread_bin = threadIdx.x * blockDim.x + threadIdx.y;

    __syncthreads();

    while (bin_count < max_bin){
        if (thread_bin < max_bin){     
            block_hist_cache[thread_bin] = 0;
        }
        thread_bin += (blockDim.x * blockDim.y);
        bin_count += (blockDim.x * blockDim.y);
    }

    __syncthreads();

    if (wval > 0.01f){
        atomicAdd(block_hist_cache + int(aval), wval);
    }

    bin_count = 0;
    thread_bin = threadIdx.x * blockDim.x + threadIdx.y;

    __syncthreads();

    while (bin_count < max_bin){
        if (thread_bin < max_bin){     
            *getPtr(
                histogram, 
                blockIdx.y,
                blockIdx.x,
                thread_bin) = block_hist_cache[thread_bin];
        }

        thread_bin += (blockDim.x * blockDim.y);
        bin_count += (blockDim.x * blockDim.y);
           
    }

}


void cell_histogram_dense_device(DeviceMatrix3D* histogram,
                                 const DeviceMatrix* assignments,
                                 const DeviceMatrix* weights,
                                 const int max_bin,
                                 const int cell_size,
                                 const int start_y,
                                 const int start_x)
{
    histogram->zero();

    dim3 dimBlock(cell_size, cell_size);
    
    int grid_ry = histogram->dim_t;
    int grid_cx = histogram->dim_y;
    
    dim3 dimGrid(grid_cx, grid_ry);

    assert(histogram->dim_x == max_bin);

    cellHistogramKernel<<<dimGrid, dimBlock>>>(
        *histogram, *assignments, *weights,
        start_y, start_x,
        max_bin);
}

#endif

#ifdef METHOD_2

#define BLOCK_SIZE 16

__global__ void cellHistogramKernel(
    DeviceMatrix3D histogram,
    DeviceMatrix assignments,
    DeviceMatrix weights,
    const int max_bin,
    const int cell_size,
    const int start_y,
    const int start_x)
{
    const int cb_ind_y = threadIdx.y / 8;
    const int cb_ind_x = threadIdx.x / 8;

    const int tc_ind_y = threadIdx.y % 8;
    const int tc_ind_x = threadIdx.x % 8;

    const int target_y = blockIdx.y * 2 + cb_ind_y;
    const int target_x = blockIdx.x * 2 + cb_ind_x;

    const int source_y = start_y + BLOCK_SIZE * blockIdx.y + threadIdx.y;
    const int source_x = start_x + BLOCK_SIZE * blockIdx.x + threadIdx.x;

    const float aval = *getPtr(assignments, source_y, source_x);
    const float wval = *getPtr(weights, source_y, source_x);

    const int cells_per_block_dim = 2;
    
    const int histogram_cache_sz = 4 * MAX_HISTOGRAM_SIZE;

    __shared__ float histogram_cache[histogram_cache_sz];

    const int cache_offset = MAX_HISTOGRAM_SIZE * 
        (cb_ind_y * cells_per_block_dim + cb_ind_x);

    //initialize the histogram
    int thread_bin_offset = tc_ind_y * cell_size + tc_ind_x;
    while (thread_bin_offset < max_bin)
    {
        const int cache_addr = cache_offset + thread_bin_offset;
        histogram_cache[cache_addr] = 0;
        thread_bin_offset += (cell_size * cell_size);
    }

    __syncthreads();

    //if (wval > 0.01f){
        atomicAdd(histogram_cache + cache_offset + (int) aval, wval);
    //}

    __syncthreads();

    //if ((target_y < histogram.dim_t) && (target_x < histogram.dim_x))
    {
        thread_bin_offset = tc_ind_y * cell_size + tc_ind_x;
        while (thread_bin_offset < max_bin)
        {
            const int cache_addr = cache_offset + thread_bin_offset;
            *getPtr(
                histogram, 
                target_y,
                target_x,
                thread_bin_offset) = 
            histogram_cache[cache_addr];
            thread_bin_offset += (cell_size * cell_size);
        }
    }
    __syncthreads();
}


void cell_histogram_dense_device(DeviceMatrix3D* histogram,
                                 const DeviceMatrix* assignments,
                                 const DeviceMatrix* weights,
                                 const int max_bin,
                                 const int cell_size,
                                 const int start_y,
                                 const int start_x)
{
    histogram->zero();

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    
    int grid_ry = (histogram->dim_t + 1) / 2;
    int grid_cx = (histogram->dim_y + 1) / 2;
    
    dim3 dimGrid(grid_cx, grid_ry);

    assert(histogram->dim_x == max_bin);

    cellHistogramKernel<<<dimGrid, dimBlock>>>(
        *histogram, *assignments, *weights,
        max_bin, cell_size,
        start_y, start_x);
}


#endif
