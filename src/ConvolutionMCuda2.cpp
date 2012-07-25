#include "ConvolutionMCuda.hpp"
#include <string.h>
#include "OpenCLKernels.hpp"
struct dim3
{
unsigned int x, y, z;
};

static const int CONSTANT_KERNEL_SIZE = 4096;
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

template<int KERN_SIZE, int SWATH_SIZE>
void do_convolution2__MCUDA_kernel(struct DeviceMatrix3D video, struct DeviceMatrix3D kernel, struct DeviceMatrix3D output, dim3 blockIdx, dim3 blockDim, dim3 gridDim)
{
dim3 threadIdx;
int __threadIndex;
float data[((KERN_SIZE+SWATH_SIZE)-1)][((KERN_SIZE+SWATH_SIZE)-1)][((KERN_SIZE+SWATH_SIZE)-1)];
unsigned int block_y;
unsigned int block_t;
unsigned int offset_x[512];
unsigned int offset_y[512];
unsigned int offset_t[512];
unsigned int t[512];
unsigned int yy[512];
unsigned int xx[512];
float * addr[512];
int base_x;
int base_t;
int base_y;
for (((threadIdx.z=0), (__threadIndex=0)); threadIdx.z<blockDim.z; threadIdx.z ++ )
{
for (threadIdx.y=0; threadIdx.y<blockDim.y; threadIdx.y ++ )
{
for (threadIdx.x=0; threadIdx.x<blockDim.x; ((threadIdx.x ++ ), (__threadIndex ++ )))
{
block_y=(blockIdx.y%(output.dim_y/SWATH_SIZE));
block_t=(blockIdx.y/(output.dim_y/SWATH_SIZE));
offset_x[__threadIndex]=(blockIdx.x*SWATH_SIZE);
offset_y[__threadIndex]=(block_y*SWATH_SIZE);
offset_t[__threadIndex]=(block_t*SWATH_SIZE);
for (t[__threadIndex]=0; t[__threadIndex]<((KERN_SIZE+SWATH_SIZE)-1); t[__threadIndex] ++ )
{
for (yy[__threadIndex]=0; yy[__threadIndex]<((KERN_SIZE+SWATH_SIZE)-1); yy[__threadIndex]+=KERN_SIZE)
{
unsigned int my_y;
my_y=(yy[__threadIndex]+threadIdx.y);
if ((my_y<((KERN_SIZE+SWATH_SIZE)-1)))
{
for (xx[__threadIndex]=0; xx[__threadIndex]<((KERN_SIZE+SWATH_SIZE)-1); xx[__threadIndex]+=16)
{
unsigned int my_x;
my_x=(xx[__threadIndex]+threadIdx.x);
if ((my_x<((KERN_SIZE+SWATH_SIZE)-1)))
{
float * retval;
{
retval=( & video.data[((((offset_t[__threadIndex]+t[__threadIndex])*video.pitch_t)+((offset_y[__threadIndex]+my_y)*video.pitch_y))+(offset_x[__threadIndex]+my_x))]);
}

addr[__threadIndex]=retval;
data[t[__threadIndex]][my_y][my_x]=( * addr[__threadIndex]);
}

}

}

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

for (((threadIdx.z=0), (__threadIndex=0)); threadIdx.z<blockDim.z; threadIdx.z ++ )
{
for (threadIdx.y=0; threadIdx.y<blockDim.y; threadIdx.y ++ )
{
for (threadIdx.x=0; threadIdx.x<blockDim.x; ((threadIdx.x ++ ), (__threadIndex ++ )))
{
base_x=(threadIdx.x%4);
base_t=(threadIdx.x/4);
base_y=threadIdx.y;
if (( ! (base_y>=SWATH_SIZE)))
{
float sum;
float * retval;
sum=0;
for (t[__threadIndex]=0; t[__threadIndex]<KERN_SIZE; t[__threadIndex] ++ )
{
for (yy[__threadIndex]=0; yy[__threadIndex]<KERN_SIZE; yy[__threadIndex] ++ )
{
for (xx[__threadIndex]=0; xx[__threadIndex]<KERN_SIZE; xx[__threadIndex] ++ )
{
sum+=(data[(base_t+t[__threadIndex])][(base_y+yy[__threadIndex])][(base_x+xx[__threadIndex])]*constant_kernel[((((((KERN_SIZE-t[__threadIndex])-1)*KERN_SIZE)+((KERN_SIZE-yy[__threadIndex])-1))*KERN_SIZE)+((KERN_SIZE-xx[__threadIndex])-1))]);
}

}

}

{
retval=( & output.data[((((offset_t[__threadIndex]+base_t)*output.pitch_t)+((offset_y[__threadIndex]+base_y)*output.pitch_y))+(offset_x[__threadIndex]+base_x))]);
}

addr[__threadIndex]=retval;
( * addr[__threadIndex])=sum;
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

template<int KERN_SIZE, int SWATH_SIZE>
void do_convolution2(struct DeviceMatrix3D video, struct DeviceMatrix3D kernel, struct DeviceMatrix3D output, dim3 gridDim, dim3 blockDim)
{
int y;
dim3 blockIdx;
blockIdx.z=0;
#pragma omp parallel for private(y, blockIdx)
for (y=0; y<gridDim.y; y ++ )
{
blockIdx.y=y;
for (blockIdx.x=0; blockIdx.x<gridDim.x; blockIdx.x ++ )
{
    do_convolution2__MCUDA_kernel<KERN_SIZE, SWATH_SIZE>(video, kernel, output, blockIdx, blockDim, gridDim);
}

}

}

bool try_convolution2_mcuda(const MCudaMatrix3D::Ptr& video,
                            const MCudaMatrix3D::Ptr& kernel,
                            MCudaMatrix3D::Ptr& output)
{
struct dim3 dimBlock;
struct dim3 dimGrid;
dimBlock.x=16;
dimBlock.y=kernel->dim_y;
dimBlock.z=1;
dimGrid.x=(output->dim_x/4);
dimGrid.y=(output->dim_y/4) * (output->dim_t/4);
dimGrid.z=1;
if (((kernel->dim_x!=kernel->dim_y)||(kernel->dim_x!=kernel->dim_t)))
{
return false;
}

    if (!convolution_kernel_set(kernel)) {
        return false;
    }

if (( ! ((((output->dim_x%4)==0)&&((output->dim_y%4)==0))&&((output->dim_t%4)==0))))
{
return false;
}

    switch (kernel->dim_x) {
    case 5:
        do_convolution2<5,4>(*video, *kernel, *output, dimGrid, dimBlock);
        return true;
    case 7:
        do_convolution2<7,4>(*video, *kernel, *output, dimGrid, dimBlock);
        return true;
    case 9:
        do_convolution2<9,4>(*video, *kernel, *output, dimGrid, dimBlock);
        return true;
    case 11:
        do_convolution2<11,4>(*video, *kernel, *output, dimGrid, dimBlock);
        return true;
    case 13:
        do_convolution2<13,4>(*video, *kernel, *output, dimGrid, dimBlock);
        return true;
    case 15:
        do_convolution2<15,4>(*video, *kernel, *output, dimGrid, dimBlock);
        return true;
    }
    return false;
}


/*
 
 OPENCL
 
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




template<int KERN_SIZE, int SWATH_SIZE>
void do_convolution2__M_cl_kernel(struct DeviceMatrixCL3D video,
								  struct DeviceMatrixCL3D kernel,
								  struct DeviceMatrixCL3D output,
								  size_t* blockIdx, 
								  size_t* blockDim,
								  size_t* gridDim)
{
	size_t threadIdx[3];
	int __threadIndex;
	float data[((KERN_SIZE+SWATH_SIZE)-1)][((KERN_SIZE+SWATH_SIZE)-1)][((KERN_SIZE+SWATH_SIZE)-1)];
	unsigned int block_y;
	unsigned int block_t;
	unsigned int offset_x[512];
	unsigned int offset_y[512];
	unsigned int offset_t[512];
	unsigned int t[512];
	unsigned int yy[512];
	unsigned int xx[512];
	float * addr[512];
	int base_x;
	int base_t;
	int base_y;
	for (((threadIdx[2]=0), (__threadIndex=0)); threadIdx[2]<blockDim[2]; threadIdx[2] ++ )
	{
		for (threadIdx[0]=0; threadIdx[0]<blockDim[0]; threadIdx[0] ++ )
		{
			for (threadIdx[1]=0; threadIdx[1]<blockDim[1]; ((threadIdx[1] ++ ), (__threadIndex ++ )))
			{
				block_y=(blockIdx[0]%(output.dim_y/SWATH_SIZE));
				block_t=(blockIdx[0]/(output.dim_y/SWATH_SIZE));
				offset_x[__threadIndex]=(blockIdx[1]*SWATH_SIZE);
				offset_y[__threadIndex]=(block_y*SWATH_SIZE);
				offset_t[__threadIndex]=(block_t*SWATH_SIZE);
				for (t[__threadIndex]=0; t[__threadIndex]<((KERN_SIZE+SWATH_SIZE)-1); t[__threadIndex] ++ )
				{
					for (yy[__threadIndex]=0; yy[__threadIndex]<((KERN_SIZE+SWATH_SIZE)-1); yy[__threadIndex]+=KERN_SIZE)
					{
						unsigned int my_y;
						my_y=(yy[__threadIndex]+threadIdx[0]);
						if ((my_y<((KERN_SIZE+SWATH_SIZE)-1)))
						{
							for (xx[__threadIndex]=0; xx[__threadIndex]<((KERN_SIZE+SWATH_SIZE)-1); xx[__threadIndex]+=16)
							{
								unsigned int my_x;
								my_x=(xx[__threadIndex]+threadIdx[1]);
								if ((my_x<((KERN_SIZE+SWATH_SIZE)-1)))
								{
									float * retval;
									{
										retval=( & video.data[((((offset_t[__threadIndex]+t[__threadIndex])*video.pitch_t)+((offset_y[__threadIndex]+my_y)*video.pitch_y))+(offset_x[__threadIndex]+my_x))]);
									}
									
									addr[__threadIndex]=retval;
									data[t[__threadIndex]][my_y][my_x]=( * addr[__threadIndex]);
								}
								
							}
							
						}
						
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
	
	for (((threadIdx[2]=0), (__threadIndex=0)); threadIdx[2]<blockDim[2]; threadIdx[2] ++ )
	{
		for (threadIdx[0]=0; threadIdx[0]<blockDim[0]; threadIdx[0] ++ )
		{
			for (threadIdx[1]=0; threadIdx[1]<blockDim[1]; ((threadIdx[1] ++ ), (__threadIndex ++ )))
			{
				base_x=(threadIdx[1]%4);
				base_t=(threadIdx[1]/4);
				base_y=threadIdx[0];
				if (( ! (base_y>=SWATH_SIZE)))
				{
					float sum;
					float * retval;
					sum=0;
					for (t[__threadIndex]=0; t[__threadIndex]<KERN_SIZE; t[__threadIndex] ++ )
					{
						for (yy[__threadIndex]=0; yy[__threadIndex]<KERN_SIZE; yy[__threadIndex] ++ )
						{
							for (xx[__threadIndex]=0; xx[__threadIndex]<KERN_SIZE; xx[__threadIndex] ++ )
							{
								sum+=(data[(base_t+t[__threadIndex])][(base_y+yy[__threadIndex])][(base_x+xx[__threadIndex])]*constant_kernel[((((((KERN_SIZE-t[__threadIndex])-1)*KERN_SIZE)+((KERN_SIZE-yy[__threadIndex])-1))*KERN_SIZE)+((KERN_SIZE-xx[__threadIndex])-1))]);
							}
							
						}
						
					}
					
					{
						retval=( & output.data[((((offset_t[__threadIndex]+base_t)*output.pitch_t)+((offset_y[__threadIndex]+base_y)*output.pitch_y))+(offset_x[__threadIndex]+base_x))]);
					}
					
					addr[__threadIndex]=retval;
					( * addr[__threadIndex])=sum;
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





template<int KERN_SIZE, int SWATH_SIZE>
void do_convolution2_cl(struct DeviceMatrixCL3D video, 
						struct DeviceMatrixCL3D kernel,
						struct DeviceMatrixCL3D output,
						size_t * gridDim, size_t* blockDim)
{
	int y;
	size_t blockIdx[3];
	blockIdx[2]=0;
#pragma omp parallel for private(y, blockIdx)
	for (y=0; y<gridDim[0]; y ++ )
	{
		blockIdx[0]=y;
		for (blockIdx[1]=0; blockIdx[1]<gridDim[1]; blockIdx[1]++ )
		{
			do_convolution2__M_cl_kernel<KERN_SIZE, SWATH_SIZE>(video,
																kernel,
																output,
																blockIdx,
																blockDim,
																gridDim);
		}
		
	}
	
}






bool try_convolution2_m_cl(const MCLMatrix3D::Ptr& video,
                            const MCLMatrix3D::Ptr& kernel,
                            MCLMatrix3D::Ptr& output)
{
	
	size_t dimBlock[3] = {kernel->dim_y,16,1}; 
	
	int grid_ry = (output->dim_x/4);
    int grid_cx = (output->dim_y/4) * (output->dim_t/4);
	
	
	const int n_blocks_x = grid_ry* dimBlock[0];
	const int n_blocks_y = grid_cx* dimBlock[1];
    
	size_t global_work_size[3] = {n_blocks_x, n_blocks_y,1};	
	
	size_t dimGrid[3] = {grid_cx, grid_ry,1};

	

	if (((kernel->dim_x!=kernel->dim_y)||(kernel->dim_x!=kernel->dim_t)))
	{
		return false;
	}
	
    if (!convolution_kernel_set_cl(kernel)) {
        return false;
    }
	
	if (( ! ((((output->dim_x%4)==0)&&((output->dim_y%4)==0))&&((output->dim_t%4)==0))))
	{
		return false;
	}
	
    switch (kernel->dim_x) {
		case 5:
			do_convolution2_cl<5,4>(*video, *kernel, *output, dimGrid, dimBlock);
			return true;
		case 7:
			do_convolution2_cl<7,4>(*video, *kernel, *output, dimGrid, dimBlock);
			return true;
		case 9:
			do_convolution2_cl<9,4>(*video, *kernel, *output, dimGrid, dimBlock);
			return true;
		case 11:
			do_convolution2_cl<11,4>(*video, *kernel, *output, dimGrid, dimBlock);
			return true;
		case 13:
			do_convolution2_cl<13,4>(*video, *kernel, *output, dimGrid, dimBlock);
			return true;
		case 15:
			do_convolution2_cl<15,4>(*video, *kernel, *output, dimGrid, dimBlock);
			return true;
    }
    return false;
}