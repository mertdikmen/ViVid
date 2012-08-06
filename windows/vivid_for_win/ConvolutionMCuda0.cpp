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


/**

OPENCL
 
 
**/

void do_convolution0__M_cl_kernel(struct DeviceMatrixCL3D video,
								  struct DeviceMatrixCL3D kernel,
								  struct DeviceMatrixCL3D output, 
								  size_t * blockIdx, 
								  size_t * blockDim, 
								  size_t * gridDim)
{
	size_t  threadIdx[3];
	int __threadIndex;
	for (((threadIdx[2]=0), (__threadIndex=0)); threadIdx[2]<blockDim[2]; threadIdx[2] ++ )
	{
		for (threadIdx[0]=0; threadIdx[0]<blockDim[0]; threadIdx[0] ++ )
		{
			for (threadIdx[1]=0; threadIdx[1]<blockDim[1]; ((threadIdx[1] ++ ), (__threadIndex ++ )))
			{
				unsigned int out_x;
				unsigned int yt;
				unsigned int out_y;
				unsigned int out_t;
				unsigned int x, y, t;
				float sum;
				out_x=((16*blockIdx[1])+threadIdx[1]);
				yt=((16*blockIdx[0])+threadIdx[0]);
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


void do_convolution0_cl(struct DeviceMatrixCL3D video, 
						struct DeviceMatrixCL3D kernel, 
						struct DeviceMatrixCL3D output,
						size_t *gridDim, size_t *local_work_size)
{
	// y can't be unsigned in OpenMP 2.5 -- I guess it would be ok for
	// OpenMP 3.0
	int y;

	size_t blockIdx[3] = {0,0,0}; 
	
#pragma omp parallel for private(y, blockIdx)
	for (y=0; y<(int)gridDim[0]; y ++ )
	{
		blockIdx[0] = y;
		for (blockIdx[1]=0; blockIdx[0]<gridDim[1]; blockIdx[0] ++ )
		{
			do_convolution0__M_cl_kernel(video, kernel, output, blockIdx,
										 local_work_size, gridDim);
		}
		
	}
	
}

bool try_convolution0_m_cl(const MCLMatrix3D::Ptr& video,
                            const MCLMatrix3D::Ptr& kernel,
                            MCLMatrix3D::Ptr& output)
{
    unsigned int yt = output->dim_y * output->dim_t;
	
	size_t local_work_size[3] = {16,16,1}; 
	
	int grid_ry = (yt-1) / 16 + 1;
    int grid_cx = (output->dim_x-1) / 16 + 1;
	
	
	const int n_blocks_x = grid_ry* local_work_size[0];
	const int n_blocks_y = grid_cx* local_work_size[1];
    
	size_t global_work_size[3] = {n_blocks_x, n_blocks_y,1};	
	
	size_t grid_size[3] = {grid_cx, grid_ry,1};
	
	

    do_convolution0_cl(*video, *kernel, *output, grid_size, global_work_size);
    return true;
}