#include "DeviceMatrix.hpp"
#include "exceptions.hpp"
#include <vector_types.h>
#include <string.h>
#include <omp.h>
#include "OpenCLKernels.hpp"
static const unsigned int CONSTANT_KERNEL_SIZE = 4096;
static float constant_kernel[CONSTANT_KERNEL_SIZE];

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

    memcpy(constant_kernel, kernel->data,
           kernel->dim_t * kernel->dim_y
           * kernel->dim_x * sizeof(float));

    return true;
}


template <int kern_size>
void do_convolution4__MCUDA_kernel(struct DeviceMatrix3D video, struct DeviceMatrix3D kernel, struct DeviceMatrix3D output, dim3 blockIdx, dim3 blockDim, dim3 gridDim)
{
dim3 threadIdx;
int __threadIndex;
unsigned int offset_x[512];
unsigned int offset_t[512];
float data[2][kern_size][((16+kern_size)-1)];
int kern_y[512];
float sum[512];
unsigned int y[512];
int SNtHfkVP[512];
for (((threadIdx.z=0), (__threadIndex=0)); threadIdx.z<blockDim.z; threadIdx.z ++ )
{
for (threadIdx.y=0; threadIdx.y<blockDim.y; threadIdx.y ++ )
{
for (threadIdx.x=0; threadIdx.x<blockDim.x; ((threadIdx.x ++ ), (__threadIndex ++ )))
{
offset_x[__threadIndex]=(16*blockIdx.x);
offset_t[__threadIndex]=blockIdx.y;
kern_y[__threadIndex]=((threadIdx.y-kern_size)+1);
sum[__threadIndex]=0;
{

data[0][threadIdx.y][threadIdx.x]=( * ( & ( & video)->data[((((offset_t[__threadIndex]+threadIdx.y)*( & video)->pitch_t)+(0*( & video)->pitch_y))+(offset_x[__threadIndex]+threadIdx.x))]));
if ((threadIdx.x<(((16+kern_size)-1)-16)))
{

data[0][threadIdx.y][(threadIdx.x+16)]=( * ( & ( & video)->data[((((offset_t[__threadIndex]+threadIdx.y)*( & video)->pitch_t)+(0*( & video)->pitch_y))+((offset_x[__threadIndex]+16)+threadIdx.x))]));
}

}

y[__threadIndex]=0;
SNtHfkVP[__threadIndex]=y[__threadIndex]<(video.dim_y-1);
/*
__MCUDA_THREAD_BODY
*/
}

}

/*
__MCUDA_OUTER_LOOP
*/
}

while (SNtHfkVP[0])
{
int read_buffer[512];
for (((threadIdx.z=0), (__threadIndex=0)); threadIdx.z<blockDim.z; threadIdx.z ++ )
{
for (threadIdx.y=0; threadIdx.y<blockDim.y; threadIdx.y ++ )
{
for (threadIdx.x=0; threadIdx.x<blockDim.x; ((threadIdx.x ++ ), (__threadIndex ++ )))
{
read_buffer[__threadIndex]=(y[__threadIndex]%2);
/*
__MCUDA_THREAD_BODY
*/
}

}

/*
__MCUDA_OUTER_LOOP
*/
}

for (((threadIdx.z=0), (__threadIndex=0)); threadIdx.z<blockDim.z; threadIdx.z ++ )
{
for (threadIdx.y=0; threadIdx.y<blockDim.y; threadIdx.y ++ )
{
for (threadIdx.x=0; threadIdx.x<blockDim.x; ((threadIdx.x ++ ), (__threadIndex ++ )))
{
{
if ((( * ( & kern_y[__threadIndex]))>=0))
{
int t, x;
for (t=0; t<kern_size; t ++ )
{
for (x=0; x<kern_size; x ++ )
{
( * ( & sum[__threadIndex]))+=(data[read_buffer[__threadIndex]][t][(threadIdx.x+x)]*constant_kernel[((((((kern_size-t)-1)*kern_size)+((kern_size-( * ( & kern_y[__threadIndex])))-1))*kern_size)+((kern_size-x)-1))]);
}

}

}

( * ( & kern_y[__threadIndex])) ++ ;
if ((( * ( & kern_y[__threadIndex]))==kern_size))
{

( * ( & ( & output)->data[(((offset_t[__threadIndex]*( & output)->pitch_t)+(((y[__threadIndex]-kern_size)+1)*( & output)->pitch_y))+(offset_x[__threadIndex]+threadIdx.x))]))=( * ( & sum[__threadIndex]));
sum[__threadIndex]=0;
kern_y[__threadIndex]=0;
}

}

{

data[( ! read_buffer[__threadIndex])][threadIdx.y][threadIdx.x]=( * ( & ( & video)->data[((((offset_t[__threadIndex]+threadIdx.y)*( & video)->pitch_t)+((y[__threadIndex]+1)*( & video)->pitch_y))+(offset_x[__threadIndex]+threadIdx.x))]));
if ((threadIdx.x<(((16+kern_size)-1)-16)))
{

data[( ! read_buffer[__threadIndex])][threadIdx.y][(threadIdx.x+16)]=( * ( & ( & video)->data[((((offset_t[__threadIndex]+threadIdx.y)*( & video)->pitch_t)+((y[__threadIndex]+1)*( & video)->pitch_y))+((offset_x[__threadIndex]+16)+threadIdx.x))]));
}

}

y[__threadIndex] ++ ;
SNtHfkVP[__threadIndex]=y[__threadIndex]<(video.dim_y-1);
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

{
int read_buffer[512];
for (((threadIdx.z=0), (__threadIndex=0)); threadIdx.z<blockDim.z; threadIdx.z ++ )
{
for (threadIdx.y=0; threadIdx.y<blockDim.y; threadIdx.y ++ )
{
for (threadIdx.x=0; threadIdx.x<blockDim.x; ((threadIdx.x ++ ), (__threadIndex ++ )))
{
y[__threadIndex]=(video.dim_y-1);
read_buffer[__threadIndex]=(y[__threadIndex]%2);
/*
__MCUDA_THREAD_BODY
*/
}

}

/*
__MCUDA_OUTER_LOOP
*/
}

for (((threadIdx.z=0), (__threadIndex=0)); threadIdx.z<blockDim.z; threadIdx.z ++ )
{
for (threadIdx.y=0; threadIdx.y<blockDim.y; threadIdx.y ++ )
{
for (threadIdx.x=0; threadIdx.x<blockDim.x; ((threadIdx.x ++ ), (__threadIndex ++ )))
{
{
if ((( * ( & kern_y[__threadIndex]))>=0))
{
int t, x;
for (t=0; t<kern_size; t ++ )
{
for (x=0; x<kern_size; x ++ )
{
( * ( & sum[__threadIndex]))+=(data[read_buffer[__threadIndex]][t][(threadIdx.x+x)]*constant_kernel[((((((kern_size-t)-1)*kern_size)+((kern_size-( * ( & kern_y[__threadIndex])))-1))*kern_size)+((kern_size-x)-1))]);
}

}

}

( * ( & kern_y[__threadIndex])) ++ ;
if ((( * ( & kern_y[__threadIndex]))==kern_size))
{

( * ( & ( & output)->data[(((offset_t[__threadIndex]*( & output)->pitch_t)+(((y[__threadIndex]-kern_size)+1)*( & output)->pitch_y))+(offset_x[__threadIndex]+threadIdx.x))]))=( * ( & sum[__threadIndex]));
sum[__threadIndex]=0;
kern_y[__threadIndex]=0;
}

}

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

}

template <int kern_size>
void do_convolution4(struct DeviceMatrix3D video, struct DeviceMatrix3D kernel, struct DeviceMatrix3D output, dim3 gridDim, dim3 blockDim)
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
do_convolution4__MCUDA_kernel<kern_size>(video, kernel, output, blockIdx, blockDim, gridDim);
}

}

}

bool try_convolution4_mcuda(const MCudaMatrix3D::Ptr& video,
                            const MCudaMatrix3D::Ptr& kernel,
                            MCudaMatrix3D::Ptr& output)
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
        do_convolution4<5>(*video, *kernel, *output, dimGrid, dimBlock);
        return true;
    case 7:
        do_convolution4<7>(*video, *kernel, *output, dimGrid, dimBlock);
            (*video, *kernel, *output);
        return true;
    case 9:
        do_convolution4<9>(*video, *kernel, *output, dimGrid, dimBlock);
            (*video, *kernel, *output);
        return true;
    case 11:
        do_convolution4<11>(*video, *kernel, *output, dimGrid, dimBlock);
        return true;
    case 13:
        do_convolution4<13>(*video, *kernel, *output, dimGrid, dimBlock);
        return true;
    case 15:
        do_convolution4<15>(*video, *kernel, *output, dimGrid, dimBlock);
        return true;
    }
    return false;
}



/**
 
 OPENCL
 
 
 **/




/**
 * This function sets the constant convolution kernel if possible.  It
 * returns true if successful.
 */
inline bool convolution_kernel_set_cl(const DeviceMatrixCL3D::Ptr kernel)
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
	/*  CUDA_CALL(cudaMemcpyToSymbol
	 (constant_kernel, kernel->data,
	 kernel->dim_t * kernel->dim_y
	 * kernel->dim_x * sizeof(float),
	 0, cudaMemcpyDeviceToDevice));
	 */
	
	TheContext* tc = new TheContext();
	cl_context GPUContext = tc->getMyContext()->getContextCL();
	cl_device_id cdDevice = tc->getMyContext()->getDeviceCL();
	
	MyKernels *kernels = new MyKernels(GPUContext,cdDevice);
	
	
	
	cl_int err;
	
	cl_mem constant_kernel =  clCreateBuffer(GPUContext, CL_MEM_READ_ONLY, kernel->dim_t * kernel->dim_y
											 * kernel->dim_x * sizeof(float),     
											 NULL, &err);
	
	err |= clEnqueueWriteBuffer(tc->getMyContext()->cqCommandQueue, constant_kernel, CL_TRUE, 0, 
								kernel->dim_t * kernel->dim_y
								* kernel->dim_x * sizeof(float), kernel->data, 0, NULL,  NULL);
	
	kernels->getMyKernels()->constant_kernel=constant_kernel;
	
    return true;
}











template <int kern_size>
void do_convolution4__M_cl_kernel(struct DeviceMatrixCL3D video,
								  struct DeviceMatrixCL3D kernel, 
								  struct DeviceMatrixCL3D output,
								  size_t * blockIdx,  
								  size_t * local_work_size,
								  size_t * grid_size)

{
	 size_t  threadIdx[3];
	int __threadIndex;
	unsigned int offset_x[512];
	unsigned int offset_t[512];
	float data[2][kern_size][((16+kern_size)-1)];
	int kern_y[512];
	float sum[512];
	unsigned int y[512];
	int SNtHfkVP[512];
	for (((threadIdx[2]=0), (__threadIndex=0)); threadIdx[2]<local_work_size[2]; threadIdx[2] ++ )
	{
		for (threadIdx[0]=0; threadIdx[0]<local_work_size[0]; threadIdx[0] ++ )
		{
			for (threadIdx[1]=0; threadIdx[1]<local_work_size[1]; ((threadIdx[1] ++ ), (__threadIndex ++ )))
			{
				offset_x[__threadIndex]=(16*blockIdx[1]);
				offset_t[__threadIndex]=blockIdx[1];
				kern_y[__threadIndex]=((threadIdx[0]-kern_size)+1);
				sum[__threadIndex]=0;
				{
					
					data[0][threadIdx[0]][threadIdx[1]]=( * ( & ( & video)->data[((((offset_t[__threadIndex]+threadIdx[0])*( & video)->pitch_t)+(0*( & video)->pitch_y))+(offset_x[__threadIndex]+threadIdx[1]))]));
					if ((threadIdx[1]<(((16+kern_size)-1)-16)))
					{
						
						data[0][threadIdx[0]][(threadIdx[1]+16)]=( * ( & ( & video)->data[((((offset_t[__threadIndex]+threadIdx[0])*( & video)->pitch_t)+(0*( & video)->pitch_y))+((offset_x[__threadIndex]+16)+threadIdx[1]))]));
					}
					
				}
				
				y[__threadIndex]=0;
				SNtHfkVP[__threadIndex]=y[__threadIndex]<(video.dim_y-1);
				/*
				 __MCUDA_THREAD_BODY
				 */
			}
			
		}
		
		/*
		 __MCUDA_OUTER_LOOP
		 */
	}
	
	while (SNtHfkVP[0])
	{
		int read_buffer[512];
		for (((threadIdx[2]=0), (__threadIndex=0)); threadIdx[2]<local_work_size[2]; threadIdx[2] ++ )
		{
			for (threadIdx[0]=0; threadIdx[0]<local_work_size[0]; threadIdx[0] ++ )
			{
				for (threadIdx[1]=0; threadIdx[1]<local_work_size[1]; ((threadIdx[1] ++ ), (__threadIndex ++ )))
				{
					read_buffer[__threadIndex]=(y[__threadIndex]%2);
					/*
					 __MCUDA_THREAD_BODY
					 */
				}
				
			}
			
			/*
			 __MCUDA_OUTER_LOOP
			 */
		}
		
		for (((threadIdx[2]=0), (__threadIndex=0)); threadIdx[2]<local_work_size[2]; threadIdx[2] ++ )
		{
			for (threadIdx[0]=0; threadIdx[0]<local_work_size[0]; threadIdx[0] ++ )
			{
				for (threadIdx[1]=0; threadIdx[1]<local_work_size[1]; ((threadIdx[1] ++ ), (__threadIndex ++ )))
				{
					{
						if ((( * ( & kern_y[__threadIndex]))>=0))
						{
							int t, x;
							for (t=0; t<kern_size; t ++ )
							{
								for (x=0; x<kern_size; x ++ )
								{
									( * ( & sum[__threadIndex]))+=(data[read_buffer[__threadIndex]][t][(threadIdx[1]+x)]*constant_kernel[((((((kern_size-t)-1)*kern_size)+((kern_size-( * ( & kern_y[__threadIndex])))-1))*kern_size)+((kern_size-x)-1))]);
								}
								
							}
							
						}
						
						( * ( & kern_y[__threadIndex])) ++ ;
						if ((( * ( & kern_y[__threadIndex]))==kern_size))
						{
							
							( * ( & ( & output)->data[(((offset_t[__threadIndex]*( & output)->pitch_t)+(((y[__threadIndex]-kern_size)+1)*( & output)->pitch_y))+(offset_x[__threadIndex]+threadIdx[1]))]))=( * ( & sum[__threadIndex]));
							sum[__threadIndex]=0;
							kern_y[__threadIndex]=0;
						}
						
					}
					
					{
						
						data[( ! read_buffer[__threadIndex])][threadIdx[0]][threadIdx[1]]=( * ( & ( & video)->data[((((offset_t[__threadIndex]+threadIdx[0])*( & video)->pitch_t)+((y[__threadIndex]+1)*( & video)->pitch_y))+(offset_x[__threadIndex]+threadIdx[1]))]));
						if ((threadIdx[1]<(((16+kern_size)-1)-16)))
						{
							
							data[( ! read_buffer[__threadIndex])][threadIdx[0]][(threadIdx[1]+16)]=( * ( & ( & video)->data[((((offset_t[__threadIndex]+threadIdx[0])*( & video)->pitch_t)+((y[__threadIndex]+1)*( & video)->pitch_y))+((offset_x[__threadIndex]+16)+threadIdx[1]))]));
						}
						
					}
					
					y[__threadIndex] ++ ;
					SNtHfkVP[__threadIndex]=y[__threadIndex]<(video.dim_y-1);
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
	
	{
		int read_buffer[512];
		for (((threadIdx[2]=0), (__threadIndex=0)); threadIdx[2]<local_work_size[2]; threadIdx[2] ++ )
		{
			for (threadIdx[0]=0; threadIdx[0]<local_work_size[0]; threadIdx[0] ++ )
			{
				for (threadIdx[1]=0; local_work_size[1]; ((threadIdx[1] ++ ), (__threadIndex ++ )))
				{
					y[__threadIndex]=(video.dim_y-1);
					read_buffer[__threadIndex]=(y[__threadIndex]%2);
					/*
					 __MCUDA_THREAD_BODY
					 */
				}
				
			}
			
			/*
			 __MCUDA_OUTER_LOOP
			 */
		}
		
		for (((threadIdx[2]=0), (__threadIndex=0)); threadIdx[2]<local_work_size[2]; threadIdx[2] ++ )
		{
			for (threadIdx[0]=0; threadIdx[0]<local_work_size[0]; threadIdx[0] ++ )
			{
				for (threadIdx[1]=0;threadIdx[1]<local_work_size[1]; ((threadIdx[1] ++ ), (__threadIndex ++ )))
				{
					{
						if ((( * ( & kern_y[__threadIndex]))>=0))
						{
							int t, x;
							for (t=0; t<kern_size; t ++ )
							{
								for (x=0; x<kern_size; x ++ )
								{
									( * ( & sum[__threadIndex]))+=(data[read_buffer[__threadIndex]][t][(threadIdx[1]+x)]*constant_kernel[((((((kern_size-t)-1)*kern_size)+((kern_size-( * ( & kern_y[__threadIndex])))-1))*kern_size)+((kern_size-x)-1))]);
								}
								
							}
							
						}
						
						( * ( & kern_y[__threadIndex])) ++ ;
						if ((( * ( & kern_y[__threadIndex]))==kern_size))
						{
							
							( * ( & ( & output)->data[(((offset_t[__threadIndex]*( & output)->pitch_t)+(((y[__threadIndex]-kern_size)+1)*( & output)->pitch_y))+(offset_x[__threadIndex]+threadIdx[1]))]))=( * ( & sum[__threadIndex]));
							sum[__threadIndex]=0;
							kern_y[__threadIndex]=0;
						}
						
					}
					
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
	
}











template <int kern_size>
void do_convolution4_cl(struct DeviceMatrixCL3D video, 
						struct DeviceMatrixCL3D kernel, 
						struct DeviceMatrixCL3D output,
						 size_t * local_work_size, 
						size_t * grid_size)
{
	// y can't be unsigned in OpenMP 2.5 -- I guess it would be ok for
	// OpenMP 3.0
	int y;
//	dim3 blockIdx;
//	blockIdx.z=0;
	 size_t  blockIdx[3];
#pragma omp parallel for private(y, blockIdx)
	for (y=0; y<(int)grid_size[0]; y ++ )
	{
		blockIdx[0] = y;
		for (blockIdx[1]=0; blockIdx[1]<grid_size[1]; blockIdx[1] ++ )
		{
			do_convolution4__M_cl_kernel<kern_size>(video, kernel, 
													output, blockIdx, 
													local_work_size, 
													 grid_size);
		}
		
	}
	
}




bool try_convolution4_m_cl(const MCLMatrix3D::Ptr& video,
                            const MCLMatrix3D::Ptr& kernel,
                            MCLMatrix3D::Ptr& output)
{
    if ((kernel->dim_x != kernel->dim_y) ||
        (kernel->dim_x != kernel->dim_t)) {
        return false;
    }
	
    if (output->dim_x % 16 != 0) {
        return false;
    }
	
    if (!convolution_kernel_set_cl(kernel)) {
        return false;
    }
	
 //   dim3 dimBlock(16, kernel->dim_y);
 //   dim3 dimGrid(output->dim_x / 16, output->dim_t);
	
	 size_t local_work_size[3] = {kernel->dim_y,16,1}; 
	
	int grid_ry = output->dim_t;
    int grid_cx = output->dim_x / 16;
	
	
	const int n_blocks_x = grid_ry* local_work_size[0];
	const int n_blocks_y = grid_cx* local_work_size[1];
    
	 size_t global_work_size[3] = {n_blocks_x, n_blocks_y,1};	
	
	 size_t grid_size[3] = {grid_cx, grid_ry,1};
	
  
    switch (kernel->dim_x) {
		case 5:
			do_convolution4_cl<5>(*video, *kernel, *output, local_work_size, grid_size);
			return true;
		case 7:
			do_convolution4_cl<7>(*video, *kernel, *output, local_work_size, grid_size);
            (*video, *kernel, *output);
			return true;
		case 9:
			do_convolution4_cl<9>(*video, *kernel, *output, local_work_size, grid_size);
            (*video, *kernel, *output);
			return true;
		case 11:
			do_convolution4_cl<11>(*video, *kernel, *output, local_work_size, grid_size);
			return true;
		case 13:
			do_convolution4_cl<13>(*video, *kernel, *output, local_work_size, grid_size);
			return true;
		case 15:
			do_convolution4_cl<15>(*video, *kernel, *output, global_work_size, grid_size);
			return true;
    }
    return false;
}

