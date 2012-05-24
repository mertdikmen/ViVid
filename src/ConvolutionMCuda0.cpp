#include "DeviceMatrix.hpp"
#include "exceptions.hpp"
#include <vector_types.h>


void do_convolution0__MCUDA_kernel(struct DeviceMatrix3D video, struct DeviceMatrix3D kernel, struct DeviceMatrix3D output, dim3 blockIdx, dim3 blockDim, dim3 gridDim)
{
dim3 threadIdx;
int __threadIndex;
for (((threadIdx.z=0), (__threadIndex=0)); threadIdx.z<blockDim.z; threadIdx.z ++ )
{
for (threadIdx.y=0; threadIdx.y<blockDim.y; threadIdx.y ++ )
{
for (threadIdx.x=0; threadIdx.x<blockDim.x; ((threadIdx.x ++ ), (__threadIndex ++ )))
{
unsigned int out_x;
unsigned int yt;
unsigned int out_y;
unsigned int out_t;
unsigned int x, y, t;
float sum;
out_x=((16*blockIdx.x)+threadIdx.x);
yt=((16*blockIdx.y)+threadIdx.y);
out_y=(yt%output.dim_y);
out_t=(yt/output.dim_y);
sum=0;
if (( ! (((out_x<output.dim_x)&&(out_y<output.dim_y))&&(out_t<output.dim_t))))
{
continue ;
}

for (x=0; x<kernel.dim_x; x ++ )
{
for (y=0; y<kernel.dim_y; y ++ )
{
for (t=0; t<kernel.dim_t; t ++ )
{


sum+=(( * ( & ( & video)->data[((((out_t+t)*( & video)->pitch_t)+((out_y+y)*( & video)->pitch_y))+(out_x+x))]))*( * ( & ( & kernel)->data[(((((kernel.dim_t-t)-1)*( & kernel)->pitch_t)+(((kernel.dim_y-y)-1)*( & kernel)->pitch_y))+((kernel.dim_x-x)-1))])));
}

}

}


( * ( & ( & output)->data[(((out_t*( & output)->pitch_t)+(out_y*( & output)->pitch_y))+out_x)]))=sum;
/*
__MCUDA_THREAD_BODY
*/
}

}

/*
__MCUDA_OUTER_LOOP
*/
}

}

void do_convolution0(struct DeviceMatrix3D video, struct DeviceMatrix3D kernel, struct DeviceMatrix3D output, dim3 gridDim, dim3 blockDim)
{
// y can't be unsigned in OpenMP 2.5 -- I guess it would be ok for
// OpenMP 3.0
int y;
dim3 blockIdx;
blockIdx.z=0;
#pragma omp parallel for private(y, blockIdx)
for (y=0; y<(int)gridDim.y; y ++ )
{
blockIdx.y = y;
for (blockIdx.x=0; blockIdx.x<gridDim.x; blockIdx.x ++ )
{
do_convolution0__MCUDA_kernel(video, kernel, output, blockIdx, blockDim, gridDim);
}

}

}

bool try_convolution0_mcuda(const MCudaMatrix3D::Ptr& video,
                            const MCudaMatrix3D::Ptr& kernel,
                            MCudaMatrix3D::Ptr& output)
{
    unsigned int yt = output->dim_y * output->dim_t;

    dim3 dimBlock(16, 16);
    dim3 dimGrid((output->dim_x-1) / 16 + 1,
                 (yt-1) / 16 + 1);
    do_convolution0(*video, *kernel, *output, dimGrid, dimBlock);
    return true;
}
