// -*- mode: c++; c-basic-offset: 4 -*-

#include "Convolution.hpp"
#include "exceptions.hpp"

DeviceMatrix3D::Ptr convolve3d(const DeviceMatrix3D::Ptr& video,
                               const DeviceMatrix3D::Ptr& kernel)
{
    DeviceMatrix3D::Ptr retval
        = makeDeviceMatrix3D(video->dim_t - kernel->dim_t + 1,
                             video->dim_y - kernel->dim_y + 1,
                             video->dim_x - kernel->dim_x + 1);

    convolve3d(video, kernel, retval);
    return retval;

}

DeviceMatrix3D::Ptr convolve3d_specific(const DeviceMatrix3D::Ptr& video,
                                        const DeviceMatrix3D::Ptr& kernel,
                                        int algorithm)
{
    DeviceMatrix3D::Ptr retval
        = makeDeviceMatrix3D(video->dim_t - kernel->dim_t + 1,
                             video->dim_y - kernel->dim_y + 1,
                             video->dim_x - kernel->dim_x + 1);

    convolve3d(video, kernel, retval, algorithm);
    return retval;
}


////////////////////////////////////////////////////////////
// Constant memory convolution kernel
////////////////////////////////////////////////////////////

static const int CONSTANT_KERNEL_SIZE = 4096;
__device__ __constant__  float constant_kernel[CONSTANT_KERNEL_SIZE];

// We use a #define here because we need to access the implicit KERN_SIZE
#define convolution_kernel_get(t,y,x) \
    constant_kernel[((((t) * KERN_SIZE) + (y)) * KERN_SIZE) + (x)]

/**
 * This function sets the constant convolution kernel if possible.  It
 * returns true if successful.
 */
inline bool convolution_kernel_set(const DeviceMatrix3D::Ptr kernel)
{
    if ((kernel->pitch_t != kernel->dim_y * kernel->pitch_y)
        || (kernel->pitch_y != kernel->dim_x)) {
        // We have to have to have a packed kernel matrix here because
        // we can't use cudaMemcpy2D to copy things into constant
        // memory -- cudaGetSymbolAddress can't take the address of
        // constant memory.
        return false;
    }

    if (kernel->dim_x * kernel->dim_y * kernel->dim_t
        > CONSTANT_KERNEL_SIZE) {
        // Kernel is too big to fit in our statically allocated array
        return false;
    }

    // Copy the kernel
    CUDA_CALL(cudaMemcpyToSymbol
              (constant_kernel, kernel->data,
               kernel->dim_t * kernel->dim_y
               * kernel->dim_x * sizeof(float),
               0, cudaMemcpyDeviceToDevice));

    return true;
}


////////////////////////////////////////////////////////////
// Helper Functions
////////////////////////////////////////////////////////////

__device__ float* get(const DeviceMatrix3D& mat,
                      unsigned int t,
                      unsigned int y,
                      unsigned int x)
{
    return &mat.data[t * mat.pitch_t + y * mat.pitch_y + x];
}


////////////////////////////////////////////////////////////
// convolution0
////////////////////////////////////////////////////////////

/**
 * This is the ultimate failsafe fallback.  It should *run* under any
 * circumstances, but there will be issues with performance.
 *
 * We expect to be called with dimBlock(16,16) and a sufficiently
 * large grid to have one thread per output value.
 */
__global__ void do_convolution0(DeviceMatrix3D video,
                                DeviceMatrix3D kernel,
                                DeviceMatrix3D output)
{
    const unsigned int out_x = 16 * blockIdx.x + threadIdx.x;
    const unsigned int yt = 16 * blockIdx.y + threadIdx.y;
    const unsigned int out_y = yt % output.dim_y;
    const unsigned int out_t = yt / output.dim_y;

    if (!((out_x < output.dim_x) &&
          (out_y < output.dim_y) &&
          (out_t < output.dim_t))) {
        // Out of bounds -- bail
        return;
    }

    float sum = 0;
    for (unsigned int x=0; x < kernel.dim_x; x++) {
        for (unsigned int y=0; y < kernel.dim_y; y++) {
            for (unsigned int t=0; t < kernel.dim_t; t++) {
                sum += *get(video, out_t + t, out_y + y, out_x + x)
                    * constant_kernel
                    [((((kernel.dim_t - t - 1) * kernel.dim_t)
                       + (kernel.dim_y - y - 1)) * kernel.dim_y)
                     + (kernel.dim_x - x - 1)];
            }
        }
    }
    *get(output, out_t, out_y, out_x) = sum;
}

bool try_convolution0(const DeviceMatrix3D::Ptr& video,
                      const DeviceMatrix3D::Ptr& kernel,
                      DeviceMatrix3D::Ptr& output)
{
    unsigned int yt = output->dim_y * output->dim_t;

    if (!convolution_kernel_set(kernel)) {
        return false;
    }

    dim3 dimBlock(16, 16);
    dim3 dimGrid((output->dim_x-1) / 16 + 1,
                 (yt-1) / 16 + 1);
    do_convolution0<<<dimGrid, dimBlock>>>(*video, *kernel, *output);
    return true;
}


////////////////////////////////////////////////////////////
// convolution1
////////////////////////////////////////////////////////////

/**
 *  The most insanely dumb algorithm possible.  We read in a block and
 *  use shared memory only to compute the sum.  We expect to be called
 *  with blockSize = (kernel.dim_x, kernel.dim_y) and gridSize =
 *  (output.dim_x, output.dim_y * output.dim_t).  (Stupid lack of 3D
 *  grids...)
 */
__global__ void do_convolution1(DeviceMatrix3D video,
                                DeviceMatrix3D kernel,
                                DeviceMatrix3D output)
{
    float sum = 0;
    const unsigned int block_y = blockIdx.y % output.dim_y;
    const unsigned int block_t = blockIdx.y / output.dim_y;

    for (unsigned int t=0; t < kernel.dim_t; t++) {
        sum += *get(video, block_t + t,
                    block_y + threadIdx.y,
                    blockIdx.x + threadIdx.x)
            * constant_kernel
            [((((kernel.dim_t - t - 1) * kernel.dim_t)
               + (kernel.dim_y - threadIdx.y - 1)) * kernel.dim_y)
             + (kernel.dim_x - threadIdx.x - 1)];
    }

  const int linear_idx = threadIdx.y * blockDim.x + threadIdx.x;
  // HACK: We can't have more than 512 threads...
  __shared__ float buffer[512];
  buffer[linear_idx] = sum;

  __syncthreads();

  unsigned int len = blockDim.x * blockDim.y;
  while (len > 1) {
      unsigned int stride =  (len+1)/ 2;
      if (linear_idx + stride < len) {
          buffer[linear_idx] += buffer[linear_idx + stride];
      }
      __syncthreads();
      len = stride;
  }

  if (linear_idx == 0) {
      *get(output, block_t, block_y, blockIdx.x) = buffer[0];
      //*get(output, block_t, block_y, blockIdx.x) = blockIdx.x;

  }
}

/**
 * This version should succeed most of the time.  The only problem
 * might be that the kernel is too large.
 */
bool try_convolution1(const DeviceMatrix3D::Ptr& video,
                      const DeviceMatrix3D::Ptr& kernel,
                      DeviceMatrix3D::Ptr& output)
{
    if (kernel->dim_x * kernel->dim_y > 512) {
        // We can't launch that many threads in a block.  (We may want
        // to rethink why we want a kernel that big...)
        return false;
    }

    if (!convolution_kernel_set(kernel)) {
        return false;
    }

    dim3 dimBlock(kernel->dim_x, kernel->dim_y);
    dim3 dimGrid(output->dim_x, output->dim_y * output->dim_t);
    do_convolution1<<<dimGrid, dimBlock>>>(*video, *kernel, *output);
    return true;
}


////////////////////////////////////////////////////////////
// convolution2
////////////////////////////////////////////////////////////


//! Helper
__device__ void sum_buffer(float buffer[], size_t len, unsigned int linear_idx)
{
  while (len > 1) {
      unsigned int stride =  (len+1)/ 2;
      if (linear_idx + stride < len) {
          buffer[linear_idx] += buffer[linear_idx + stride];
      }
      __syncthreads();
      len = stride;
  }
}

/**
 * We expected to be called with dimBlock(16, KERN_SIZE).  We also
 * need KERN_SIZE == kernel->dim_x == kernel->dim_y == kernel->dim_t,
 * and for the dimensions of output to be a multiple of SWATH_SIZE.
 *
 * @warning We need SWATH_SIZE to be 4.  See note below
 */
template< int KERN_SIZE, int SWATH_SIZE >
__global__ void do_convolution2(DeviceMatrix3D video,
                                DeviceMatrix3D kernel,
                                DeviceMatrix3D output)
{
    const int DATA_SIZE = SWATH_SIZE + KERN_SIZE - 1;
    __shared__ float data[DATA_SIZE][DATA_SIZE][DATA_SIZE];

    const unsigned int block_y = blockIdx.y % (output.dim_y / SWATH_SIZE);
    const unsigned int block_t = blockIdx.y / (output.dim_y / SWATH_SIZE);
    const unsigned int offset_x = blockIdx.x * SWATH_SIZE;
    const unsigned int offset_y = block_y * SWATH_SIZE;
    const unsigned int offset_t = block_t * SWATH_SIZE;

    for (unsigned int t=0; t < DATA_SIZE; t++)
    {
        for (unsigned int y=0; y < DATA_SIZE; y += KERN_SIZE)
        {
            const unsigned int my_y = y + threadIdx.y;
            if (my_y < DATA_SIZE)
            {
                for (unsigned int x=0; x < DATA_SIZE; x += 16)
                {
                    const unsigned int my_x = x + threadIdx.x;
                    if (my_x < DATA_SIZE) {
                        data[t][my_y][my_x] =
                            *get(video, offset_t + t,
                                        offset_y + my_y,
                                        offset_x + my_x);
                    }
                }
            }
        }
    }

    __syncthreads();

    /**
     * @todo put this whole thing in a loop in case SWATH_SIZE ever
     * gets bigger than 4.  Actually, we count on the fact hat
     * SWATH_SIZE * SWATH_SIZE = 16.
     */
    const int base_x = threadIdx.x % 4;
    const int base_t = threadIdx.x / 4;
    const int base_y = threadIdx.y;

    /**
     * @warning We depend on blockDim.y == KERN_SIZE >= SWATH_SIZE
     * @todo Remove dependency between KERN_SIZE and SWATH_SIZE
     */
    if (base_y >= SWATH_SIZE) {
        return;
    }

    float sum = 0;
    for (unsigned int t=0; t < KERN_SIZE; t++) {
        for (unsigned int y=0; y < KERN_SIZE; y++) {
            for (unsigned int x=0; x < KERN_SIZE; x++) {
                sum += data[base_t + t][base_y + y][base_x + x] *
                    convolution_kernel_get(KERN_SIZE-t-1, KERN_SIZE-y-1,
                                           KERN_SIZE-x-1);
            }
        }
    }
    // Write out the solution
    *get(output,
         offset_t + base_t,
         offset_y + base_y,
         offset_x + base_x) = sum;
}

bool try_convolution2(const DeviceMatrix3D::Ptr& video,
                      const DeviceMatrix3D::Ptr& kernel,
                      DeviceMatrix3D::Ptr& output)
{
    if ((kernel->dim_x != kernel->dim_y) ||
        (kernel->dim_x != kernel->dim_t)) {
        return false;
    }

    if (!((output->dim_x % 4 == 0) &&
          (output->dim_y % 4 == 0) &&
          (output->dim_t % 4 == 0))) {
        // Only support a SWATH_SIZE of 4 for now.
        return false;
    }

    if (!convolution_kernel_set(kernel)) {
        return false;
    }

    dim3 dimBlock(16, kernel->dim_y);
    dim3 dimGrid(output->dim_x / 4,
                 (output->dim_y / 4) *
                 (output->dim_t / 4));

    switch (kernel->dim_x) {
    case 5:
        do_convolution2<5,4><<<dimGrid, dimBlock>>>
            (*video, *kernel, *output);
        return true;
    case 7:
        do_convolution2<7,4><<<dimGrid, dimBlock>>>
            (*video, *kernel, *output);
        return true;
    case 9:
        do_convolution2<9,4><<<dimGrid, dimBlock>>>
            (*video, *kernel, *output);
        return true;
    case 11:
        do_convolution2<11,4><<<dimGrid, dimBlock>>>
            (*video, *kernel, *output);
        return true;
    }
    return false;
}


////////////////////////////////////////////////////////////
// convolution3
////////////////////////////////////////////////////////////


template <int KERN_SIZE, int DATA_SIZE, int SWATH_SIZE>
__device__ void
do_convolution3_load(float data[KERN_SIZE][DATA_SIZE],
                     DeviceMatrix3D video,
                     int offset_x, int offset_t,
                     int y)
{
    // Write the next set of data into the buffer.  We make use of the
    // fact that kernel.dim_y == kernel.dim_t == blockDim.y to load
    // all t values in parallel.
    data[threadIdx.y][threadIdx.x] =
        *get(video,
             offset_t + threadIdx.y,
             y,
             offset_x + threadIdx.x);

    // Get the trim
    if (threadIdx.x < DATA_SIZE - SWATH_SIZE) {
        data[threadIdx.y][threadIdx.x + SWATH_SIZE] =
            *get(video,
                 offset_t + threadIdx.y,
                 y,
                 offset_x + SWATH_SIZE + threadIdx.x);
    }
}

template <int KERN_SIZE, int DATA_SIZE>
__device__ void
do_convolution3_consume(float data[KERN_SIZE][DATA_SIZE],
                        float kern[KERN_SIZE][KERN_SIZE][KERN_SIZE],
                        DeviceMatrix3D output,
                        unsigned int offset_x, unsigned int offset_t,
                        int y, int& kern_y, float& sum)
{
    if (kern_y >= 0) {
        // Read the data we wrote last time and increment partial sums
        for (int t = 0; t < KERN_SIZE; t++) {
            for (int x = 0; x < KERN_SIZE; x++) {
                sum += data[t][threadIdx.x + x] *
                    kern[KERN_SIZE-t-1][KERN_SIZE-kern_y-1]
                    [KERN_SIZE-x-1];
            }
        }
    }

    kern_y++;

    if (kern_y == KERN_SIZE) {
        // Write out the solution
        *get(output, offset_t, y-KERN_SIZE+1, offset_x + threadIdx.x)
            = sum;
        sum = 0;
        kern_y = 0;
    }
}

/**
 * We expected to be called with dimBlock(16, KERN_SIZE, 1).
 * Each block will handle a 16 x output.dim_y x 1 piece of the output.
 */
template< int KERN_SIZE >
__global__ void do_convolution3(DeviceMatrix3D video,
                                DeviceMatrix3D kernel,
                                DeviceMatrix3D output)
{
    __shared__ float kern[KERN_SIZE][KERN_SIZE][KERN_SIZE];
    for (unsigned int t=0; t < KERN_SIZE; t++)
    {
        if (threadIdx.x < KERN_SIZE)
        {
            kern[t][threadIdx.y][threadIdx.x] =
                *get(kernel, t, threadIdx.y, threadIdx.x);
        }
    }

    const int SWATH_SIZE = 16;

    const unsigned int offset_x = SWATH_SIZE * blockIdx.x;
    const unsigned int offset_t = blockIdx.y;

    const int DATA_SIZE = SWATH_SIZE + KERN_SIZE - 1;
    __shared__ float data[2][KERN_SIZE][DATA_SIZE];

    // This variable indexes the kernel as we slide the image through.
    // We start out in the negative zone to avoid writing out results
    // outside the "valid" zone.  The values here should range from
    // -(KERN_SIZE-1):0
    int kern_y = threadIdx.y - KERN_SIZE + 1;

    // This tracks the partial sum
    float sum = 0;

    do_convolution3_load<KERN_SIZE, DATA_SIZE, SWATH_SIZE>
        (data[0],
         video, offset_x, offset_t,
         0);

    // Carve a swath in y.
    for (int y=0; y < (video.dim_y-1); y++) {
        //for (int y=0; y < 9; y++) {
        __syncthreads();

        // Flag for ping-ponging
        int read_buffer = y % 2;

        do_convolution3_consume<KERN_SIZE, DATA_SIZE>
            (data[read_buffer], kern,
             output, offset_x, offset_t,
             y, kern_y, sum);

        // Load the next frame -- hopefully the compiler will move the
        // independent load earlier in the loop.
        do_convolution3_load<KERN_SIZE, DATA_SIZE, SWATH_SIZE>
            (data[!read_buffer],
             video, offset_x, offset_t,
             y+1);
    }

    {
        // One last write
        int y = video.dim_y-1;
        int read_buffer = y % 2;

        __syncthreads();
        do_convolution3_consume<KERN_SIZE, DATA_SIZE>
            (data[read_buffer], kern,
             output, offset_x, offset_t,
             y, kern_y, sum);
    }
}

bool try_convolution3(const DeviceMatrix3D::Ptr& video,
                      const DeviceMatrix3D::Ptr& kernel,
                      DeviceMatrix3D::Ptr& output)
{
    if ((kernel->dim_x != kernel->dim_y) ||
        (kernel->dim_x != kernel->dim_t)) {
        return false;
    }

    if (output->dim_x % 16 != 0) {
        return false;
    }

    dim3 dimBlock(16, kernel->dim_y);
    dim3 dimGrid(output->dim_x / 16, output->dim_t);
    switch (kernel->dim_x) {
    case 5:
        do_convolution3<5><<<dimGrid, dimBlock>>>
            (*video, *kernel, *output);
        return true;
    case 7:
        do_convolution3<7><<<dimGrid, dimBlock>>>
            (*video, *kernel, *output);
        return true;
    case 9:
        do_convolution3<9><<<dimGrid, dimBlock>>>
            (*video, *kernel, *output);
        return true;
    case 11:
        do_convolution3<11><<<dimGrid, dimBlock>>>
            (*video, *kernel, *output);
        return true;
    }
    return false;
}


////////////////////////////////////////////////////////////
// convolution4
////////////////////////////////////////////////////////////


template <int KERN_SIZE, int DATA_SIZE>
__device__ void
do_convolution4_consume(float data[KERN_SIZE][DATA_SIZE],
                        DeviceMatrix3D output,
                        unsigned int offset_x, unsigned int offset_t,
                        int y, int& kern_y, float& sum)
{
    if (kern_y >= 0)
    {
        // Read the data we wrote last time and increment partial sums
        for (int t = 0; t < KERN_SIZE; t++)
        {
            for (int x = 0; x < KERN_SIZE; x++)
            {
                sum += data[t][threadIdx.x + x] *
                    convolution_kernel_get(KERN_SIZE-t-1, KERN_SIZE-kern_y-1,
                                           KERN_SIZE-x-1);
            }
        }
    }

    kern_y++;

    if (kern_y == KERN_SIZE)
    {
        // Write out the solution
        // *get(output, offset_x + threadIdx.x, y-KERN_SIZE+1, offset_t)
        //     = 10+threadIdx.y;

        // *get(output, offset_x + threadIdx.x, 0, offset_t)
        //     = threadIdx.y;

        *get(output, offset_t, y-KERN_SIZE+1, offset_x + threadIdx.x)
            = sum;
        sum = 0;
        kern_y = 0;
    }
}

/**
 * This function is like do_convolution3() except that we use constant
 * memory to hold the kernel.
 *
 * @note This function uses do_convolution3_load() as the algorithm is
 * exactly the same.
 */
template< int KERN_SIZE >
__global__ void do_convolution4(DeviceMatrix3D video,
                                DeviceMatrix3D kernel,
                                DeviceMatrix3D output)
{
    const int SWATH_SIZE = 16;

    const unsigned int offset_x = SWATH_SIZE * blockIdx.x;
    const unsigned int offset_t = blockIdx.y;

    const int DATA_SIZE = SWATH_SIZE + KERN_SIZE - 1;
    __shared__ float data[2][KERN_SIZE][DATA_SIZE];

    // This variable indexes the kernel as we slide the image through.
    // We start out in the negative zone to avoid writing out results
    // outside the "valid" zone.  The values here should range from
    // -(KERN_SIZE-1):0
    int kern_y = threadIdx.y - KERN_SIZE + 1;

    // This tracks the partial sum
    float sum = 0;

    do_convolution3_load<KERN_SIZE, DATA_SIZE, SWATH_SIZE>
        (data[0],
         video, offset_x, offset_t,
         0);

    // Carve a swath in y.
    for (int y=0; y < (video.dim_y-1); y++) {
        //for (int y=0; y < 9; y++) {
        __syncthreads();

        // Flag for ping-ponging
        int read_buffer = y % 2;

        do_convolution4_consume<KERN_SIZE, DATA_SIZE>
            (data[read_buffer],
             output, offset_x, offset_t,
             y, kern_y, sum);

        // Load the next frame -- hopefully the compiler will move the
        // independent load earlier in the loop.
        do_convolution3_load<KERN_SIZE, DATA_SIZE, SWATH_SIZE>
            (data[!read_buffer],
             video, offset_x, offset_t,
             y+1);
    }

    {
        // One last write
        int y = video.dim_y-1;
        int read_buffer = y % 2;

        __syncthreads();
        do_convolution4_consume<KERN_SIZE, DATA_SIZE>
            (data[read_buffer],
             output, offset_x, offset_t,
             y, kern_y, sum);
    }
}

bool try_convolution4(const DeviceMatrix3D::Ptr& video,
                      const DeviceMatrix3D::Ptr& kernel,
                      DeviceMatrix3D::Ptr& output)
{
    if ((kernel->dim_x != kernel->dim_y) ||
        (kernel->dim_x != kernel->dim_t)) {
        return false;
    }

    if (output->dim_x % 16 != 0) {
        return false;
    }

    if (!convolution_kernel_set(kernel)) {
        return false;
    }

    dim3 dimBlock(16, kernel->dim_y);
    dim3 dimGrid(output->dim_x / 16, output->dim_t);
    switch (kernel->dim_x) {
    case 5:
        do_convolution4<5><<<dimGrid, dimBlock>>>
            (*video, *kernel, *output);
        return true;
    case 7:
        do_convolution4<7><<<dimGrid, dimBlock>>>
            (*video, *kernel, *output);
        return true;
    case 9:
        do_convolution4<9><<<dimGrid, dimBlock>>>
            (*video, *kernel, *output);
        return true;
    case 11:
        do_convolution4<11><<<dimGrid, dimBlock>>>
            (*video, *kernel, *output);
        return true;
    case 13:
        do_convolution4<13><<<dimGrid, dimBlock>>>
            (*video, *kernel, *output);
        return true;
    case 15:
        do_convolution4<15><<<dimGrid, dimBlock>>>
            (*video, *kernel, *output);
        return true;
    }
    return false;
}


////////////////////////////////////////////////////////////
// convolution5
////////////////////////////////////////////////////////////

/**
 * This is just like do_convolution2, except that we deliberately
 * transpose the memory access to show that uncoalesed memory accesses
 * hurt.
 */
template< int KERN_SIZE, int SWATH_SIZE >
__global__ void do_convolution5(DeviceMatrix3D video,
                                DeviceMatrix3D kernel,
                                DeviceMatrix3D output)
{
    const int DATA_SIZE = SWATH_SIZE + KERN_SIZE - 1;
    __shared__ float data[DATA_SIZE][DATA_SIZE][DATA_SIZE];

    const unsigned int block_y = blockIdx.y % (output.dim_y / SWATH_SIZE);
    const unsigned int block_t = blockIdx.y / (output.dim_y / SWATH_SIZE);
    const unsigned int offset_x = blockIdx.x * SWATH_SIZE;
    const unsigned int offset_y = block_y * SWATH_SIZE;
    const unsigned int offset_t = block_t * SWATH_SIZE;

    for (unsigned int t=0; t < DATA_SIZE; t++)
    {
        for (unsigned int y=0; y < DATA_SIZE; y += KERN_SIZE)
        {
            const unsigned int my_y = y + threadIdx.y;
            if (my_y < DATA_SIZE)
            {
                for (unsigned int x=0; x < DATA_SIZE; x += 16)
                {
                    const unsigned int my_x = x + threadIdx.x;
                    if (my_x < DATA_SIZE) {
                        data[my_x][my_y][t] =
                            *get(video, offset_t + my_x,
                                        offset_y + my_y,
                                        offset_x + t);
                    }
                }
            }
        }
    }

    __syncthreads();

    /**
     * @todo put this whole thing in a loop in case SWATH_SIZE ever
     * gets bigger than 4.  Actually, we count on the fact hat
     * SWATH_SIZE * SWATH_SIZE = 16.
     */
    const int base_x = threadIdx.x % 4;
    const int base_t = threadIdx.x / 4;
    const int base_y = threadIdx.y;

    /**
     * @warning We depend on blockDim.y == KERN_SIZE >= SWATH_SIZE
     * @todo Remove dependency between KERN_SIZE and SWATH_SIZE
     */
    if (base_y >= SWATH_SIZE) {
        return;
    }

    float sum = 0;
    for (unsigned int t=0; t < KERN_SIZE; t++) {
        for (unsigned int y=0; y < KERN_SIZE; y++) {
            for (unsigned int x=0; x < KERN_SIZE; x++) {
                sum += data[base_t + t][base_y + y][base_x + x] *
                    convolution_kernel_get(KERN_SIZE-t-1, KERN_SIZE-y-1,
                                           KERN_SIZE-x-1);
            }
        }
    }
    // Write out the solution
    *get(output,
         offset_t + base_t,
         offset_y + base_y,
         offset_x + base_x) = sum;
}


bool try_convolution5(const DeviceMatrix3D::Ptr& video,
                      const DeviceMatrix3D::Ptr& kernel,
                      DeviceMatrix3D::Ptr& output)
{
    if ((kernel->dim_x != kernel->dim_y) ||
        (kernel->dim_x != kernel->dim_t)) {
        return false;
    }

    if (!((output->dim_x % 4 == 0) &&
          (output->dim_y % 4 == 0) &&
          (output->dim_t % 4 == 0))) {
        // Only support a SWATH_SIZE of 4 for now.
        return false;
    }

    if (!convolution_kernel_set(kernel)) {
        return false;
    }

    dim3 dimBlock(16, kernel->dim_y);
    dim3 dimGrid(output->dim_x / 4,
                 (output->dim_y / 4) *
                 (output->dim_t / 4));

    switch (kernel->dim_x) {
    case 5:
        do_convolution5<5,4><<<dimGrid, dimBlock>>>
            (*video, *kernel, *output);
        return true;
    case 7:
        do_convolution5<7,4><<<dimGrid, dimBlock>>>
            (*video, *kernel, *output);
        return true;
    case 9:
        do_convolution5<9,4><<<dimGrid, dimBlock>>>
            (*video, *kernel, *output);
        return true;
    case 11:
        do_convolution5<11,4><<<dimGrid, dimBlock>>>
            (*video, *kernel, *output);
        return true;
    }
    return false;
}


////////////////////////////////////////////////////////////
// convolve3d - convolution driver
////////////////////////////////////////////////////////////


// stolen from http://en.wikibooks.org/wiki/C_Programming/Pointers_and_arrays
#define NUM_ELEM(x) (sizeof (x) / sizeof (*(x)))

//! An internal debugging flag
static unsigned int debug_algorithm_used;

typedef bool (*try_convolution_algorithm) (const DeviceMatrix3D::Ptr&,
                                           const DeviceMatrix3D::Ptr&,
                                           DeviceMatrix3D::Ptr&);
//! The list of convolution algorithms
static const try_convolution_algorithm convolution_algorithm[] = {
    &try_convolution0, &try_convolution1,
    &try_convolution2, &try_convolution3,
    &try_convolution4, &try_convolution5
};

//! The order to try the convolution algorithms
/**
 * As usual, every problem in computer science can be solved by
 * another layer of indirection.  We introduce this layer so that we
 * can keep the numbering of the algorithms stable.
 */
static const unsigned int convolution_order[] =
    {4,3,2,0,1,5};

/**
 * @note We perform a 'valid' (as opposed to 'same', or 'full')
 * convolution, and thus we expect that output->dim_x = video->dim.x -
 * kernel->dim.x + 1.
 *
 * @warning The caller is responsible for making sure that the output
 * is the right size.
 */
void convolve3d(const DeviceMatrix3D::Ptr& video,
                const DeviceMatrix3D::Ptr& kernel,
                DeviceMatrix3D::Ptr& output)
{
    for (int i = 0; i < NUM_ELEM(convolution_order); i++) {
        unsigned int to_try = convolution_order[i];
        if (convolution_algorithm[to_try](video, kernel, output)) {
            debug_algorithm_used = to_try;
            return;
        }
    }
    throw_runtime_error("Unable to find convolution algorithm");
}


/**
 * Like convolve3d(), (The same notes and warnings apply.)  However,
 * we only try one algorithm before giving up.
 */
void convolve3d(const DeviceMatrix3D::Ptr& video,
                const DeviceMatrix3D::Ptr& kernel,
                DeviceMatrix3D::Ptr& output,
                unsigned int algorithm)
{
    assert(algorithm < NUM_ELEM(convolution_algorithm));
    if (convolution_algorithm[algorithm](video, kernel, output)) {
            debug_algorithm_used = algorithm;
            return;
        }

    throw_runtime_error("Unable to find convolution algorithm");
}

/**
 * @warning This function is not threadsafe and all.  Use at your own
 * risk.
 */
unsigned int debug_convolution_algorithm_used()
{
    return debug_algorithm_used;
}


////////////////////////////////////////////////////////////
// Helpers for complex numbers
////////////////////////////////////////////////////////////


__device__ float2 complex_multiply(float2 a, float2 b)
{
    return make_float2(a.x * b.x - a.y * b.y,
                       a.x * b.y + a.y * b.x);
}

/**
 * Returns a pointer to a float2 element of a matrix
 *
 * @note We internally multipy the x by 2
 */
__device__ float2* get2(const DeviceMatrix3D& mat,
                        unsigned int t,
                        unsigned int y,
                        unsigned int x)
{
    return (float2*)(&mat.data[t * mat.pitch_t + y * mat.pitch_y + 2*x]);
}

/**
 * Because we can't overload a float2...
 */
__device__ void increment(float2& self, float2 other)
{
    self.x += other.x;
    self.y += other.y;
}

/**
 * Because we can't overload a float2...
 */
__device__ void scale(float2& self, float multiplier)
{
    self.x *= multiplier;
    self.y *= multiplier;
}


////////////////////////////////////////////////////////////
// convolution_complex_t0
////////////////////////////////////////////////////////////


/**
 * The brain-dead general case
 */
__global__ void do_convolution_complex_t0(DeviceMatrix3D video,
                                          DeviceMatrix3D kernel,
                                          float scale_val,
                                          DeviceMatrix3D output)
{
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    if ((x < output.dim_x/2) && (y < output.dim_y)) {
        for (int t=0; t < output.dim_t; t++) {
            float2 sum = make_float2(0,0);
            for (int s=0; s < kernel.dim_t; s++) {
                increment(sum,
                  complex_multiply(*get2(video, t+s, y, x),
                                   *get2(kernel, kernel.dim_t-s-1, y,x )));
            }
            scale(sum, scale_val);
            *get2(output,t,y,x) = sum;
        }
    }
}

/**
 * This version will always succeed.
 */
bool try_convolution_complex_t0(const DeviceMatrix3D::Ptr& video,
                                const DeviceMatrix3D::Ptr& kernel,
                                float scale,
                                DeviceMatrix3D::Ptr& output)
{
    dim3 dimBlock(16,16);
    dim3 dimGrid((output->dim_x-1) / 16 + 1,
                 (output->dim_y-1) / 16 + 1);
    do_convolution_complex_t0<<<dimGrid, dimBlock>>>
        (*video, *kernel, scale, *output);
    return true;
}


////////////////////////////////////////////////////////////
// convolution_complex_t1
////////////////////////////////////////////////////////////

/**
 * @note We assume that video.dim_x == kernel.dim_x and video.dim_y =
 * kernel.dim_y.
 *
 * @warning video, kernel, and output must be packed matrices
 *
 * @warning video.pitch_t must be divisible by 2 * SWATH_YX.  (It
 * would just be SWATH_YX except that we're dealing with complex
 * values which take up 2 reals.)
 *
 * @note We expected to be called with a blockDim of (SWATH_YX,
 * KERN_SIZE) and a gridDim of (video.pitch_t / (2*SWATH_YX)).  This
 * should give us a thread for each pixel in one frame of video.
 */
template< int SWATH_YX, int KERN_SIZE>
__global__ void do_convolution_complex_t1(DeviceMatrix3D video,
                                          DeviceMatrix3D kernel,
                                          float scale_val,
                                          DeviceMatrix3D output)
{
    __shared__ float2 kern[KERN_SIZE][SWATH_YX];

    const unsigned int offset_yx = SWATH_YX * blockIdx.x + threadIdx.x;
    const unsigned int offset_t  = threadIdx.y;

    // Abuse the offsets because we have packed matrices.  We
    // essentially drop the problem back to 2D.
    kern[offset_t][threadIdx.x] = *get2(kernel, offset_t, 0, offset_yx);

    __shared__ float2 data[2][SWATH_YX];

    // Flag for ping-ponging
    int read_buffer = 0;

    // This variable indexes the kernel as we slide the image through.
    // The values here should range from 0:(KERN_SIZE-1)
    int kern_t = offset_t;

    // This tracks the partial sum
    float2 sum = make_float2(0,0);

    // Prime the buffer
    data[read_buffer][threadIdx.x] = *get2(video, 0, 0, offset_yx);

    // Carve a swath in t.
    for (unsigned int t=0; t < video.dim_t; t++) {
        // Hoist the reading up higher.  We will read the value that
        // we want to use next.
        float2 load_value;
        if (threadIdx.y == 0) {
            if ((t+1) < video.dim_t) {
                load_value = *get2(video, t+1, 0, offset_yx);
            }
        }

        increment(sum, complex_multiply(data[read_buffer][threadIdx.x],
                                        kern[KERN_SIZE - kern_t -1]
                                            [threadIdx.x]));
        kern_t++;
        if (kern_t == KERN_SIZE) {
            int out_t = t - KERN_SIZE + 1;
            if (0 <= out_t) {
                // Write out the solution
                scale(sum, scale_val);
                *get2(output, out_t, 0, offset_yx) = sum;
            }
            sum = make_float2(0,0);
            kern_t = 0;
        }

        if (threadIdx.y == 0) {
            data[!read_buffer][threadIdx.x] = load_value;
        }

        read_buffer = !read_buffer;

        // Prevent the next iteration of the loop from scribbling on
        // values we are trying to read
        __syncthreads();
    }
}

inline bool is_packed(const DeviceMatrix3D::Ptr& mat)
{
    return ((mat->pitch_t == mat->dim_y * mat->pitch_y)
            && (mat->pitch_y == mat->dim_x));

}

bool try_convolution_complex_t1(const DeviceMatrix3D::Ptr& video,
                                const DeviceMatrix3D::Ptr& kernel,
                                float scale,
                                DeviceMatrix3D::Ptr& output)
{
    // Make sure that all the matrices are packed
    if ((!is_packed(video)) || (!is_packed(kernel)) || (!is_packed(output))) {
        return false;
    }

    const int SWATH_SIZE = 16;
    // Make sure that we're good on our swath size
    if ( (output->pitch_t % (2*SWATH_SIZE)) != 0 ) {
        return false;
    }

    dim3 dimBlock(SWATH_SIZE, kernel->dim_t);
    dim3 dimGrid(video->pitch_t / (2*SWATH_SIZE));
    switch (kernel->dim_t) {
    case 5:
        do_convolution_complex_t1<SWATH_SIZE, 5><<<dimGrid, dimBlock>>>
            (*video, *kernel, scale, *output);
        return true;
    case 7:
        do_convolution_complex_t1<SWATH_SIZE, 7><<<dimGrid, dimBlock>>>
            (*video, *kernel, scale, *output);
        return true;
    case 9:
        do_convolution_complex_t1<SWATH_SIZE, 9><<<dimGrid, dimBlock>>>
            (*video, *kernel, scale, *output);
        return true;
    case 11:
        do_convolution_complex_t1<SWATH_SIZE, 11><<<dimGrid, dimBlock>>>
            (*video, *kernel, scale, *output);
        return true;
    }
    return false;
}


////////////////////////////////////////////////////////////
// convolve_complex_t - convolution driver
////////////////////////////////////////////////////////////


typedef bool (*try_convolution_complex_t) (const DeviceMatrix3D::Ptr&,
                                           const DeviceMatrix3D::Ptr&,
                                           float,
                                           DeviceMatrix3D::Ptr&);
//! A list of algorithms to try
static const try_convolution_complex_t convolution_complex_t[] = {
    &try_convolution_complex_t0,
    &try_convolution_complex_t1
};

/**
 * Perform a 1D (complex) convolution in time.  We expect that
 * video->dim_x == kernel->dim_x, and video->dim_y == kernel->dim_y,
 * and output->dim_t = (video->dim_t - kernel->dim_y + 1).
 *
 * @warning The caller is responsible for making sure that the output
 * is the right size.
 */
void convolve_complex_t(const DeviceMatrix3D::Ptr& video,
                        const DeviceMatrix3D::Ptr& kernel,
                        float scale,
                        DeviceMatrix3D::Ptr& output)
{
    for (int i = NUM_ELEM(convolution_complex_t)-1; i >= 0; i--) {
        if (convolution_complex_t[i](video, kernel, scale, output)) {
            debug_algorithm_used = i;
            return;
        }
    }
    throw_runtime_error("Unable to find convolution algorithm");
}


/**
 * Like convolve_complex_t(), (The same notes and warnings apply.)
 * However, we only try one algorithm before giving up.
 */
void convolve_complex_t_specific(const DeviceMatrix3D::Ptr& video,
                                 const DeviceMatrix3D::Ptr& kernel,
                                 float scale,
                                 DeviceMatrix3D::Ptr& output,
                                 unsigned int algorithm)
{
    assert(algorithm < NUM_ELEM(convolution_complex_t));
    if (convolution_complex_t[algorithm](video, kernel, scale, output)) {
            debug_algorithm_used = algorithm;
            return;
        }

    throw_runtime_error("Unable to find convolution algorithm");
}
