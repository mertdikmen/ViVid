// -*- mode: c++; c-basic-offset: 4 -*-

#include "Convolution.hpp"
#include "OpenCLKernels.hpp"
#include "exceptions.hpp"

DeviceMatrixCL3D::Ptr convolve3d_cl(const DeviceMatrixCL3D::Ptr& video,
                               const DeviceMatrixCL3D::Ptr& kernel)
{
    DeviceMatrixCL3D::Ptr retval
        = makeDeviceMatrixCL3D(video->dim_t - kernel->dim_t + 1,
                             video->dim_y - kernel->dim_y + 1,
                             video->dim_x - kernel->dim_x + 1);

    convolve3d_cl(video, kernel, retval);
    return retval;

}

DeviceMatrixCL3D::Ptr convolve3d_specific_cl(const DeviceMatrixCL3D::Ptr& video,
                                        const DeviceMatrixCL3D::Ptr& kernel,
                                        int algorithm)
{
    DeviceMatrixCL3D::Ptr retval
        = makeDeviceMatrixCL3D(video->dim_t - kernel->dim_t + 1,
                             video->dim_y - kernel->dim_y + 1,
                             video->dim_x - kernel->dim_x + 1);

    convolve3d_cl(video, kernel, retval, algorithm);
    return retval;
}


////////////////////////////////////////////////////////////
// Constant memory convolution kernel
////////////////////////////////////////////////////////////

static const int CONSTANT_KERNEL_SIZE = 4096;
//__device__ __constant__  float constant_kernel[CONSTANT_KERNEL_SIZE];

// We use a #define here because we need to access the implicit KERN_SIZE
#define convolution_kernel_get_cl(t,y,x) \
constant_kernel[((((t) * KERN_SIZE) + (y)) * KERN_SIZE) + (x)]

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
/*__global__ void do_convolution0(DeviceMatrixCL3D video,
                                DeviceMatrixCL3D kernel,
                                DeviceMatrixCL3D output)

*/




cl_int parameters_doConvolution0(cl_kernel theKernel,const DeviceMatrixCL3D::Ptr& video,
								 const DeviceMatrixCL3D::Ptr& kernel,
								 DeviceMatrixCL3D::Ptr& output,cl_mem constant_kernel){
	cl_int err=0;
	
	err |= clSetKernelArg(theKernel, 0, sizeof (cl_mem), &video->dataMatrix);
    err |= clSetKernelArg(theKernel, 1, sizeof (int), &video->dim_x);
    err |= clSetKernelArg(theKernel, 2, sizeof (int), &video->dim_y);
    err |= clSetKernelArg(theKernel, 3, sizeof (int), &video->dim_t);
	err |= clSetKernelArg(theKernel, 4, sizeof (int), &video->pitch_y);
    err |= clSetKernelArg(theKernel, 5, sizeof (int), &video->pitch_t);
	err |= clSetKernelArg(theKernel, 6, sizeof (cl_mem), &kernel->dataMatrix);
    err |= clSetKernelArg(theKernel, 7, sizeof (int), &kernel->dim_x);
    err |= clSetKernelArg(theKernel, 8, sizeof (int), &kernel->dim_y);
    err |= clSetKernelArg(theKernel, 9, sizeof (int), &kernel->dim_t);
	err |= clSetKernelArg(theKernel, 10, sizeof (int), &kernel->pitch_y);
    err |= clSetKernelArg(theKernel, 11, sizeof (int), &kernel->pitch_t);
	err |= clSetKernelArg(theKernel, 12, sizeof (cl_mem), &output->dataMatrix);
    err |= clSetKernelArg(theKernel, 13, sizeof (int), &output->dim_x);
    err |= clSetKernelArg(theKernel, 14, sizeof (int), &output->dim_y);
    err |= clSetKernelArg(theKernel, 15, sizeof (int), &output->dim_t);
	err |= clSetKernelArg(theKernel, 16, sizeof (int), &output->pitch_y);
    err |= clSetKernelArg(theKernel, 17, sizeof (int), &output->pitch_t);
	err |= clSetKernelArg(theKernel, 18, sizeof (cl_mem), &constant_kernel);
	
	return err;
}



bool try_convolution0_cl(const DeviceMatrixCL3D::Ptr& video,
                      const DeviceMatrixCL3D::Ptr& kernel,
                      DeviceMatrixCL3D::Ptr& output)
{
    unsigned int yt = output->dim_y * output->dim_t;
	
    if (!convolution_kernel_set_cl(kernel)) {
        return false;
    }
	const size_t local_work_size[2] = {16, 16}; 
	
	int grid_ry =(yt-1) / 16 + 1;
    int grid_cx = (output->dim_x-1) / 16 + 1;
	
	const int n_blocks_x =  grid_ry* local_work_size[0];
	const int n_blocks_y = grid_cx* local_work_size[1];

	const size_t global_work_size[2] = {n_blocks_x, n_blocks_y};

    
	
	TheContext* tc = new TheContext();
	
    cl_context GPUContext = tc->getMyContext()->getContextCL();
    cl_device_id cdDevice = tc->getMyContext()->getDeviceCL();
	
    // Creates the program
    // Uses NVIDIA helper functions to get the code string and it's size (in bytes)
  	
	MyKernels *kernels = new MyKernels(GPUContext,cdDevice);
	
	cl_kernel theKernel= kernels->getDoConvolution0();
	
	cl_int err;
	err=0;
	
    err =  parameters_doConvolution0(theKernel, video,kernel, 
									 output,
									 kernels->getMyKernels()->constant_kernel);	
  	
    if (err != CL_SUCCESS) {
        printf("Error: Failed to set kernel arguments 3! %d\n", err);
        exit(1);
    }
	
	
	err = clEnqueueNDRangeKernel(tc->getMyContext()->cqCommandQueue, 
								 theKernel, 2, NULL, 
								 global_work_size, local_work_size, 0, NULL, NULL);
	
    if (err) {
        printf("Error: Failed to execute kernel! %d\n", err);
        exit(1);
    }
	
	
	
	
	/*dim3 dimBlock(16, 16);
    dim3 dimGrid((output->dim_x-1) / 16 + 1,
                 (yt-1) / 16 + 1);
    do_convolution0<<<dimGrid, dimBlock>>>(*video, *kernel, *output);
	 */
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
/*__global__ void do_convolution1(DeviceMatrixCL3D video,
                                DeviceMatrixCL3D kernel,
                                DeviceMatrixCL3D output)*/
/**
 * This version should succeed most of the time.  The only problem
 * might be that the kernel is too large.
 */
bool try_convolution1_cl(const DeviceMatrixCL3D::Ptr& video,
                      const DeviceMatrixCL3D::Ptr& kernel,
                      DeviceMatrixCL3D::Ptr& output)
{
    if (kernel->dim_x * kernel->dim_y > 512) {
        // We can't launch that many threads in a block.  (We may want
        // to rethink why we want a kernel that big...)
        return false;
    }
	
    if (!convolution_kernel_set_cl(kernel)) {
        return false;
    }
	
	const size_t local_work_size[2] = {kernel->dim_y, kernel->dim_x}; 
	
	int grid_ry =output->dim_y * output->dim_t;
    int grid_cx = output->dim_x;
	
	const int n_blocks_x =  grid_ry* local_work_size[0];
	const int n_blocks_y = grid_cx* local_work_size[1];
	
	const size_t global_work_size[2] = {n_blocks_x, n_blocks_y};
	
	TheContext* tc = new TheContext();
	
    cl_context GPUContext = tc->getMyContext()->getContextCL();
    cl_device_id cdDevice = tc->getMyContext()->getDeviceCL();
	
    // Creates the program
    // Uses NVIDIA helper functions to get the code string and it's size (in bytes)
  	
	MyKernels *kernels = new MyKernels(GPUContext,cdDevice);
	
	cl_kernel theKernel= kernels->getDoConvolution1();
	
	cl_int err;
	err=0;
	
    err =  parameters_doConvolution0(theKernel, video,kernel, 
									 output,
									 kernels->getMyKernels()->constant_kernel);	
  	
    if (err != CL_SUCCESS) {
        printf("Error: Failed to set kernel arguments 3! %d\n", err);
        exit(1);
    }
	
	
	err = clEnqueueNDRangeKernel(tc->getMyContext()->cqCommandQueue, 
								 theKernel, 2, NULL, 
								 global_work_size, local_work_size, 0, NULL, NULL);
	
    if (err) {
        printf("Error: Failed to execute kernel! %d\n", err);
        exit(1);
    }
	
   /* dim3 dimBlock(kernel->dim_x, kernel->dim_y);
    dim3 dimGrid(output->dim_x, output->dim_y * output->dim_t);
    do_convolution1<<<dimGrid, dimBlock>>>(*video, *kernel, *output);*/
    return true;
}







////////////////////////////////////////////////////////////
// convolution2
////////////////////////////////////////////////////////////




/**
 * We expected to be called with dimBlock(16, KERN_SIZE).  We also
 * need KERN_SIZE == kernel->dim_x == kernel->dim_y == kernel->dim_t,
 * and for the dimensions of output to be a multiple of SWATH_SIZE.
 *
 * @warning We need SWATH_SIZE to be 4.  See note below
 */
/*template< int KERN_SIZE, int SWATH_SIZE >
__global__ void do_convolution2(DeviceMatrixCL3D video,
                                DeviceMatrixCL3D kernel,
                                DeviceMatrixCL3D output)
*/

cl_int parameters_doConvolution2(cl_kernel theKernel,const DeviceMatrixCL3D::Ptr& video,
								 const DeviceMatrixCL3D::Ptr& kernel,
								 DeviceMatrixCL3D::Ptr& output,cl_mem constant_kernel,
								 const int KS,const int SS){
	cl_int err=0;
	
	err |= clSetKernelArg(theKernel, 0, sizeof (cl_mem), &video->dataMatrix);
    err |= clSetKernelArg(theKernel, 1, sizeof (int), &video->dim_x);
    err |= clSetKernelArg(theKernel, 2, sizeof (int), &video->dim_y);
    err |= clSetKernelArg(theKernel, 3, sizeof (int), &video->dim_t);
	err |= clSetKernelArg(theKernel, 4, sizeof (int), &video->pitch_y);
    err |= clSetKernelArg(theKernel, 5, sizeof (int), &video->pitch_t);
	err |= clSetKernelArg(theKernel, 6, sizeof (cl_mem), &kernel->dataMatrix);
    err |= clSetKernelArg(theKernel, 7, sizeof (int), &kernel->dim_x);
    err |= clSetKernelArg(theKernel, 8, sizeof (int), &kernel->dim_y);
    err |= clSetKernelArg(theKernel, 9, sizeof (int), &kernel->dim_t);
	err |= clSetKernelArg(theKernel, 10, sizeof (int), &kernel->pitch_y);
    err |= clSetKernelArg(theKernel, 11, sizeof (int), &kernel->pitch_t);
	err |= clSetKernelArg(theKernel, 12, sizeof (cl_mem), &output->dataMatrix);
    err |= clSetKernelArg(theKernel, 13, sizeof (int), &output->dim_x);
    err |= clSetKernelArg(theKernel, 14, sizeof (int), &output->dim_y);
    err |= clSetKernelArg(theKernel, 15, sizeof (int), &output->dim_t);
	err |= clSetKernelArg(theKernel, 16, sizeof (int), &output->pitch_y);
    err |= clSetKernelArg(theKernel, 17, sizeof (int), &output->pitch_t);
	err |= clSetKernelArg(theKernel, 18, sizeof (cl_mem), &constant_kernel);
	err |= clSetKernelArg(theKernel, 19, sizeof (int), &KS);
	err |= clSetKernelArg(theKernel, 20, sizeof (int), &SS);
	
	return err;
}

bool try_convolution2_cl(const DeviceMatrixCL3D::Ptr& video,
                      const DeviceMatrixCL3D::Ptr& kernel,
                      DeviceMatrixCL3D::Ptr& output)
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
	
    if (!convolution_kernel_set_cl(kernel)) {
        return false;
    }
	
	const size_t local_work_size[2] = {kernel->dim_y, 16}; 
	
	int grid_ry =(output->dim_y / 4) *
	(output->dim_t / 4);
    int grid_cx = output->dim_x / 4;
	
	const int n_blocks_x =  grid_ry* local_work_size[0];
	const int n_blocks_y = grid_cx* local_work_size[1];
	
	const size_t global_work_size[2] = {n_blocks_x, n_blocks_y};
	
	TheContext* tc = new TheContext();
	
    cl_context GPUContext = tc->getMyContext()->getContextCL();
    cl_device_id cdDevice = tc->getMyContext()->getDeviceCL();
	
    // Creates the program
    // Uses NVIDIA helper functions to get the code string and it's size (in bytes)
  	
	MyKernels *kernels = new MyKernels(GPUContext,cdDevice);
	
	cl_kernel theKernel;
	
	cl_int err;
	err=0;
	
   	
  /*  dim3 dimBlock(16, kernel->dim_y);
    dim3 dimGrid(output->dim_x / 4,
                 (output->dim_y / 4) *
                 (output->dim_t / 4));
	*/
	bool switchDone = false;
    switch (kernel->dim_x) {
		case 5:
			theKernel= kernels->getDoConvolution2_8();
			err =  parameters_doConvolution2(theKernel, video,kernel, 
											 output,
											 kernels->getMyKernels()->constant_kernel,5,4);	
			
			
			switchDone = true;
			break;
		case 7:
			theKernel= kernels->getDoConvolution2_10();
			err =  parameters_doConvolution2(theKernel, video,kernel, 
											 output,
											 kernels->getMyKernels()->constant_kernel,7,4);	
			switchDone =true;
			break;
		case 9:
			theKernel= kernels->getDoConvolution2_12();
			err =  parameters_doConvolution2(theKernel, video,kernel, 
											 output,
											 kernels->getMyKernels()->constant_kernel,9,4);	
			switchDone = true;
			break;
		case 11:
			theKernel= kernels->getDoConvolution2_14();
			err =  parameters_doConvolution2(theKernel, video,kernel, 
											 output,
											 kernels->getMyKernels()->constant_kernel,11,4);	
			switchDone = true;
			
			break;
	 
    }
	
	if (err != CL_SUCCESS) {
        printf("Error: Failed to set kernel arguments 3! %d\n", err);
        exit(1);
    }
	
	
	err = clEnqueueNDRangeKernel(tc->getMyContext()->cqCommandQueue, 
								 theKernel, 2, NULL, 
								 global_work_size, local_work_size, 0, NULL, NULL);
	
    if (err) {
        printf("Error: Failed to execute kernel! %d\n", err);
        exit(1);
    }
	
    return switchDone;
}







////////////////////////////////////////////////////////////
// convolution3
////////////////////////////////////////////////////////////
cl_int parameters_doConvolution3(cl_kernel theKernel,const DeviceMatrixCL3D::Ptr& video,
								 const DeviceMatrixCL3D::Ptr& kernel,
								 DeviceMatrixCL3D::Ptr& output,cl_mem constant_kernel,
								 const int KS){
	cl_int err=0;
	
	err |= clSetKernelArg(theKernel, 0, sizeof (cl_mem), &video->dataMatrix);
    err |= clSetKernelArg(theKernel, 1, sizeof (int), &video->dim_x);
    err |= clSetKernelArg(theKernel, 2, sizeof (int), &video->dim_y);
    err |= clSetKernelArg(theKernel, 3, sizeof (int), &video->dim_t);
	err |= clSetKernelArg(theKernel, 4, sizeof (int), &video->pitch_y);
    err |= clSetKernelArg(theKernel, 5, sizeof (int), &video->pitch_t);
	err |= clSetKernelArg(theKernel, 6, sizeof (cl_mem), &kernel->dataMatrix);
    err |= clSetKernelArg(theKernel, 7, sizeof (int), &kernel->dim_x);
    err |= clSetKernelArg(theKernel, 8, sizeof (int), &kernel->dim_y);
    err |= clSetKernelArg(theKernel, 9, sizeof (int), &kernel->dim_t);
	err |= clSetKernelArg(theKernel, 10, sizeof (int), &kernel->pitch_y);
    err |= clSetKernelArg(theKernel, 11, sizeof (int), &kernel->pitch_t);
	err |= clSetKernelArg(theKernel, 12, sizeof (cl_mem), &output->dataMatrix);
    err |= clSetKernelArg(theKernel, 13, sizeof (int), &output->dim_x);
    err |= clSetKernelArg(theKernel, 14, sizeof (int), &output->dim_y);
    err |= clSetKernelArg(theKernel, 15, sizeof (int), &output->dim_t);
	err |= clSetKernelArg(theKernel, 16, sizeof (int), &output->pitch_y);
    err |= clSetKernelArg(theKernel, 17, sizeof (int), &output->pitch_t);
	err |= clSetKernelArg(theKernel, 18, sizeof (cl_mem), &constant_kernel);
	err |= clSetKernelArg(theKernel, 19, sizeof (int), &KS);

	
	return err;
}
bool try_convolution3_cl(const DeviceMatrixCL3D::Ptr& video,
                      const DeviceMatrixCL3D::Ptr& kernel,
                      DeviceMatrixCL3D::Ptr& output)
{
    if ((kernel->dim_x != kernel->dim_y) ||
        (kernel->dim_x != kernel->dim_t)) {
        return false;
    }
	
    if (output->dim_x % 16 != 0) {
        return false;
    }
	
	const size_t local_work_size[2] = {kernel->dim_y, 16}; 
	
	int grid_ry =output->dim_t;
    int grid_cx = output->dim_x / 16;
	
	const int n_blocks_x =  grid_ry* local_work_size[0];
	const int n_blocks_y = grid_cx* local_work_size[1];
	
	const size_t global_work_size[2] = {n_blocks_x, n_blocks_y};
	
	
	
 /*   dim3 dimBlock(16, kernel->dim_y);
    dim3 dimGrid(output->dim_x / 16, output->dim_t);
  */
    TheContext* tc = new TheContext();
	
    cl_context GPUContext = tc->getMyContext()->getContextCL();
    cl_device_id cdDevice = tc->getMyContext()->getDeviceCL();
	
    // Creates the program
    // Uses NVIDIA helper functions to get the code string and it's size (in bytes)
  	
	MyKernels *kernels = new MyKernels(GPUContext,cdDevice);
	
	cl_kernel theKernel;
	
	cl_int err;
	err=0;

	bool switchDone=false;
	switch (kernel->dim_x) {
		case 5:
			theKernel= kernels->getDoConvolution3();
			err =  parameters_doConvolution3(theKernel, video,kernel, 
											 output,
											 kernels->getMyKernels()->constant_kernel,5);	
			
			
			switchDone = true;
			
			break;		case 7:
			theKernel= kernels->getDoConvolution3_7();
			err =  parameters_doConvolution3(theKernel, video,kernel, 
											 output,
											 kernels->getMyKernels()->constant_kernel,7);
			switchDone = true;
			
			break;		case 9:
			theKernel= kernels->getDoConvolution3_9();
			err =  parameters_doConvolution3(theKernel, video,kernel, 
											 output,
											 kernels->getMyKernels()->constant_kernel,9);
			switchDone = true;
			
			break;		case 11:
			theKernel= kernels->getDoConvolution3_11();
			err =  parameters_doConvolution3(theKernel, video,kernel, 
											 output,
											 kernels->getMyKernels()->constant_kernel,11);
			switchDone = true;
			
			break;
    }
	
   	if (err != CL_SUCCESS) {
        printf("Error: Failed to set kernel arguments 3! %d\n", err);
        exit(1);
    }
	
	
	err = clEnqueueNDRangeKernel(tc->getMyContext()->cqCommandQueue, 
								 theKernel, 2, NULL, 
								 global_work_size, local_work_size, 0, NULL, NULL);
	
    if (err) {
        printf("Error: Failed to execute kernel! %d\n", err);
        exit(1);
    }
	
    return switchDone;
}




////////////////////////////////////////////////////////////
// convolution4
////////////////////////////////////////////////////////////


bool try_convolution4_cl(const DeviceMatrixCL3D::Ptr& video,
                      const DeviceMatrixCL3D::Ptr& kernel,
                      DeviceMatrixCL3D::Ptr& output)
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
	
	const size_t local_work_size[2] = {kernel->dim_y, 16}; 
	
	int grid_ry =output->dim_t;
    int grid_cx = output->dim_x / 16;
	
	const int n_blocks_x =  grid_ry* local_work_size[0];
	const int n_blocks_y = grid_cx* local_work_size[1];
	
	const size_t global_work_size[2] = {n_blocks_x, n_blocks_y};
	
	TheContext* tc = new TheContext();
	
    cl_context GPUContext = tc->getMyContext()->getContextCL();
    cl_device_id cdDevice = tc->getMyContext()->getDeviceCL();
	
    // Creates the program
    // Uses NVIDIA helper functions to get the code string and it's size (in bytes)
  	
	MyKernels *kernels = new MyKernels(GPUContext,cdDevice);
	
	cl_kernel theKernel;
	
	cl_int err;
	err=0;
	
	bool switchDone=false;
	
   /* dim3 dimBlock(16, kernel->dim_y);
    dim3 dimGrid(output->dim_x / 16, output->dim_t);
	*/
    switch (kernel->dim_x) {
		case 5:
			theKernel= kernels->getDoConvolution4();
			//there are the same arguments as convolution3
			err =  parameters_doConvolution3(theKernel, video,kernel, 
											 output,
											 kernels->getMyKernels()->constant_kernel,5);	
			switchDone = true;
			
			break;
		case 7:
			theKernel= kernels->getDoConvolution4_7();
			//there are the same arguments as convolution3
			err =  parameters_doConvolution3(theKernel, video,kernel, 
											 output,
											 kernels->getMyKernels()->constant_kernel,5);	
			switchDone = true;
			
			break;
		case 9:
			theKernel= kernels->getDoConvolution4_9();
			//there are the same arguments as convolution3
			err =  parameters_doConvolution3(theKernel, video,kernel, 
											 output,
											 kernels->getMyKernels()->constant_kernel,kernel->dim_x);	

			switchDone = true;
			
			break;
		case 11:
			theKernel= kernels->getDoConvolution4_11();
			//there are the same arguments as convolution3
			err =  parameters_doConvolution3(theKernel, video,kernel, 
											 output,
											 kernels->getMyKernels()->constant_kernel,kernel->dim_x);	
			switchDone = true;
			
			break;
		case 13:
			theKernel= kernels->getDoConvolution4_13();
			//there are the same arguments as convolution3
			err =  parameters_doConvolution3(theKernel, video,kernel, 
											 output,
											 kernels->getMyKernels()->constant_kernel,kernel->dim_x);	
			switchDone = true;
			
			break;
		case 15:
			theKernel= kernels->getDoConvolution4_15();
			//there are the same arguments as convolution3
			err =  parameters_doConvolution3(theKernel, video,kernel, 
											 output,
											 kernels->getMyKernels()->constant_kernel,kernel->dim_x);	

			switchDone = true;
			
			break;
    }
	if (err != CL_SUCCESS) {
        printf("Error: Failed to set kernel arguments 3! %d\n", err);
        exit(1);
    }
	
	
	err = clEnqueueNDRangeKernel(tc->getMyContext()->cqCommandQueue, 
								 theKernel, 2, NULL, 
								 global_work_size, local_work_size, 0, NULL, NULL);
	
    if (err) {
        printf("Error: Failed to execute kernel! %d\n", err);
        exit(1);
    }
	
    return switchDone;}



////////////////////////////////////////////////////////////
// convolution5
////////////////////////////////////////////////////////////

/**
 * This is just like do_convolution2, except that we deliberately
 * transpose the memory access to show that uncoalesed memory accesses
 * hurt.
 */


bool try_convolution5_cl(const DeviceMatrixCL3D::Ptr& video,
                      const DeviceMatrixCL3D::Ptr& kernel,
                      DeviceMatrixCL3D::Ptr& output)
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
	
    if (!convolution_kernel_set_cl(kernel)) {
        return false;
    }
	
	const size_t local_work_size[2] = {kernel->dim_y, 16}; 
	
	int grid_ry =(output->dim_y / 4) *
	(output->dim_t / 4);
    int grid_cx =  output->dim_x / 4;
	
	const int n_blocks_x =  grid_ry* local_work_size[0];
	const int n_blocks_y = grid_cx* local_work_size[1];
	
	const size_t global_work_size[2] = {n_blocks_x, n_blocks_y};
	
	TheContext* tc = new TheContext();
	
    cl_context GPUContext = tc->getMyContext()->getContextCL();
    cl_device_id cdDevice = tc->getMyContext()->getDeviceCL();
	
    // Creates the program
    // Uses NVIDIA helper functions to get the code string and it's size (in bytes)
  	
	MyKernels *kernels = new MyKernels(GPUContext,cdDevice);
	
	cl_kernel theKernel;
	
	cl_int err;
	err=0;
	
	bool switchDone=false;
	
	
/*	
    dim3 dimBlock(16, kernel->dim_y);
    dim3 dimGrid(output->dim_x / 4,
                 (output->dim_y / 4) *
                 (output->dim_t / 4));
	*/
    switch (kernel->dim_x) {
		case 5:
			theKernel= kernels->getDoConvolution5();
			err =  parameters_doConvolution2(theKernel, video,kernel, 
											 output,
											 kernels->getMyKernels()->constant_kernel,5,4);	
			switchDone = true;
			break;
		case 7:
			theKernel= kernels->getDoConvolution5_7();
			err =  parameters_doConvolution2(theKernel, video,kernel, 
											 output,
											 kernels->getMyKernels()->constant_kernel,7,4);	
			switchDone = true;
			break;
			
		case 9:
			theKernel= kernels->getDoConvolution5_9();
			err =  parameters_doConvolution2(theKernel, video,kernel, 
											 output,
											 kernels->getMyKernels()->constant_kernel,9,4);	
			switchDone = true;
			break;
		case 11:
			theKernel= kernels->getDoConvolution5_11();
			err =  parameters_doConvolution2(theKernel, video,kernel, 
											 output,
											 kernels->getMyKernels()->constant_kernel,11,4);	
			switchDone = true;
			break;
    }
 
	if (err != CL_SUCCESS) {
        printf("Error: Failed to set kernel arguments 3! %d\n", err);
        exit(1);
    }
	
	
	err = clEnqueueNDRangeKernel(tc->getMyContext()->cqCommandQueue, 
								 theKernel, 2, NULL, 
								 global_work_size, local_work_size, 0, NULL, NULL);
	
    if (err) {
        printf("Error: Failed to execute kernel! %d\n", err);
        exit(1);
    }
	
    return switchDone;
}




////////////////////////////////////////////////////////////
// convolve3d - convolution driver
////////////////////////////////////////////////////////////


// stolen from http://en.wikibooks.org/wiki/C_Programming/Pointers_and_arrays
#define NUM_ELEM(x) (sizeof (x) / sizeof (*(x)))


static unsigned int debug_algorithm_used_cl;

typedef bool (*try_convolution_algorithm_cl) (const DeviceMatrixCL3D::Ptr&,
                                           const DeviceMatrixCL3D::Ptr&,
                                           DeviceMatrixCL3D::Ptr&);
//! The list of convolution algorithms
static const try_convolution_algorithm_cl convolution_algorithm_cl[] = {
    &try_convolution0_cl, &try_convolution1_cl,
    &try_convolution2_cl, &try_convolution3_cl,
    &try_convolution4_cl, &try_convolution5_cl
};


//! The order to try the convolution algorithms
/**
 * As usual, every problem in computer science can be solved by
 * another layer of indirection.  We introduce this layer so that we
 * can keep the numbering of the algorithms stable.
 */
static const unsigned int convolution_order_cl[] =
{4,3,2,0,1,5};

/**
 * @note We perform a 'valid' (as opposed to 'same', or 'full')
 * convolution, and thus we expect that output->dim_x = video->dim.x -
 * kernel->dim.x + 1.
 *
 * @warning The caller is responsible for making sure that the output
 * is the right size.
 */

void convolve3d_cl(const DeviceMatrixCL3D::Ptr& video,
                const DeviceMatrixCL3D::Ptr& kernel,
                DeviceMatrixCL3D::Ptr& output)
{
   for (int i = 0; i < NUM_ELEM(convolution_order_cl); i++) {
        unsigned int to_try = convolution_order_cl[i];
        if (convolution_algorithm_cl[to_try](video, kernel, output)) {
            debug_algorithm_used_cl = to_try;
            return;
        }
    }
     throw_runtime_error("Unable to find convolution algorithm");
}


/**
 * Like convolve3d(), (The same notes and warnings apply.)  However,
 * we only try one algorithm before giving up.
 */
void convolve3d_cl(const DeviceMatrixCL3D::Ptr& video,
                const DeviceMatrixCL3D::Ptr& kernel,
                DeviceMatrixCL3D::Ptr& output,
                unsigned int algorithm)
{
    assert(algorithm < NUM_ELEM(convolution_algorithm_cl));
    if (convolution_algorithm_cl[algorithm](video, kernel, output)) {
		debug_algorithm_used_cl = algorithm;
		return;
	}
	
    throw_runtime_error("Unable to find convolution algorithm");
}

/**
 * @warning This function is not threadsafe and all.  Use at your own
 * risk.
 */
unsigned int debug_convolution_algorithm_used_cl()
{
    return debug_algorithm_used_cl;
}




////////////////////////////////////////////////////////////
// convolution_complex_t0
////////////////////////////////////////////////////////////

cl_int parameters_doConvolutionComplex(cl_kernel theKernel,
									   const DeviceMatrixCL3D::Ptr& video,
								 const DeviceMatrixCL3D::Ptr& kernel,
								 DeviceMatrixCL3D::Ptr& output,
									   const float scale){
	cl_int err=0;
	
	err |= clSetKernelArg(theKernel, 0, sizeof (cl_mem), &video->dataMatrix);
    err |= clSetKernelArg(theKernel, 1, sizeof (int), &video->dim_x);
    err |= clSetKernelArg(theKernel, 2, sizeof (int), &video->dim_y);
    err |= clSetKernelArg(theKernel, 3, sizeof (int), &video->dim_t);
	err |= clSetKernelArg(theKernel, 4, sizeof (int), &video->pitch_y);
    err |= clSetKernelArg(theKernel, 5, sizeof (int), &video->pitch_t);
	err |= clSetKernelArg(theKernel, 6, sizeof (cl_mem), &kernel->dataMatrix);
    err |= clSetKernelArg(theKernel, 7, sizeof (int), &kernel->dim_x);
    err |= clSetKernelArg(theKernel, 8, sizeof (int), &kernel->dim_y);
    err |= clSetKernelArg(theKernel, 9, sizeof (int), &kernel->dim_t);
	err |= clSetKernelArg(theKernel, 10, sizeof (int), &kernel->pitch_y);
    err |= clSetKernelArg(theKernel, 11, sizeof (int), &kernel->pitch_t);
	err |= clSetKernelArg(theKernel, 12, sizeof (cl_mem), &output->dataMatrix);
    err |= clSetKernelArg(theKernel, 13, sizeof (int), &output->dim_x);
    err |= clSetKernelArg(theKernel, 14, sizeof (int), &output->dim_y);
    err |= clSetKernelArg(theKernel, 15, sizeof (int), &output->dim_t);
	err |= clSetKernelArg(theKernel, 16, sizeof (int), &output->pitch_y);
    err |= clSetKernelArg(theKernel, 17, sizeof (int), &output->pitch_t);
	err |= clSetKernelArg(theKernel, 18, sizeof (float), &scale);
	
	return err;
}
/**
 * The brain-dead general case
 */

/**
 * This version will always succeed.
 */
bool try_convolution_complex_t0_cl(const DeviceMatrixCL3D::Ptr& video,
                                const DeviceMatrixCL3D::Ptr& kernel,
                                float scale,
                                DeviceMatrixCL3D::Ptr& output)
{
	const size_t local_work_size[2] = {16, 16}; 
	
	int grid_ry =(output->dim_y-1) / 16 + 1;
    int grid_cx =  (output->dim_x-1) / 16 + 1;
	
	const int n_blocks_x =  grid_ry* local_work_size[0];
	const int n_blocks_y = grid_cx* local_work_size[1];
	
	const size_t global_work_size[2] = {n_blocks_x, n_blocks_y};
	
	TheContext* tc = new TheContext();
	
    cl_context GPUContext = tc->getMyContext()->getContextCL();
    cl_device_id cdDevice = tc->getMyContext()->getDeviceCL();
	
    // Creates the program
    // Uses NVIDIA helper functions to get the code string and it's size (in bytes)
  	
	MyKernels *kernels = new MyKernels(GPUContext,cdDevice);
	
	cl_kernel theKernel= kernels->getDoConvolutionComplexT0();
	
	cl_int err;
	err=0;
	
    err =  parameters_doConvolutionComplex(theKernel, video,kernel, 
									 output,
									 scale);	
  	
    if (err != CL_SUCCESS) {
        printf("Error: Failed to set kernel arguments 3! %d\n", err);
        exit(1);
    }
	
	
	err = clEnqueueNDRangeKernel(tc->getMyContext()->cqCommandQueue, 
								 theKernel, 2, NULL, 
								 global_work_size, local_work_size, 0, NULL, NULL);
	
    if (err) {
        printf("Error: Failed to execute kernel! %d\n", err);
        exit(1);
    }
	
	
	
//getDoConvolutionComplexT0
	
    /*dim3 dimBlock(16,16);
    dim3 dimGrid((output->dim_x-1) / 16 + 1,
                 (output->dim_y-1) / 16 + 1);
    do_convolution_complex_t0<<<dimGrid, dimBlock>>>
	(*video, *kernel, scale, *output);*/
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

inline bool is_packed_cl(const DeviceMatrixCL3D::Ptr& mat)
{
    return ((mat->pitch_t == mat->dim_y * mat->pitch_y)
            && (mat->pitch_y == mat->dim_x));
	
}

bool try_convolution_complex_t1_cl(const DeviceMatrixCL3D::Ptr& video,
                                const DeviceMatrixCL3D::Ptr& kernel,
                                float scale,
                                DeviceMatrixCL3D::Ptr& output)
{
    // Make sure that all the matrices are packed
    if ((!is_packed_cl(video)) || (!is_packed_cl(kernel)) || (!is_packed_cl(output))) {
        return false;
    }
	
    const int SWATH_SIZE = 16;
    // Make sure that we're good on our swath size
    if ( (output->pitch_t % (2*SWATH_SIZE)) != 0 ) {
        return false;
    }
	
	const size_t local_work_size[2] = {kernel->dim_t, SWATH_SIZE}; 
	
	int grid_ry =1;
    int grid_cx =  video->pitch_t / (2*SWATH_SIZE);
	
	const int n_blocks_x =  grid_ry* local_work_size[0];
	const int n_blocks_y = grid_cx* local_work_size[1];
	
	const size_t global_work_size[2] = {n_blocks_x, n_blocks_y};
	
	TheContext* tc = new TheContext();
	
    cl_context GPUContext = tc->getMyContext()->getContextCL();
    cl_device_id cdDevice = tc->getMyContext()->getDeviceCL();
	
	MyKernels *kernels = new MyKernels(GPUContext,cdDevice);
	
		
	cl_int err;
	err=0;
	cl_kernel theKernel;
		
   
	bool switchDone = false;
    /*dim3 dimBlock(SWATH_SIZE, kernel->dim_t);
    dim3 dimGrid(video->pitch_t / (2*SWATH_SIZE));*/
    switch (kernel->dim_t) {
		case 5:
			theKernel= kernels->getDoConvolutionComplexT1_5();

			switchDone = true;
		case 7:
			theKernel= kernels->getDoConvolutionComplexT1_7();
			switchDone = true;
		case 9:
			theKernel= kernels->getDoConvolutionComplexT1_9();
			switchDone = true;
		case 11:
			theKernel= kernels->getDoConvolutionComplexT1_11();
			switchDone = true;
    }
	
	err =  parameters_doConvolutionComplex(theKernel, video,kernel, 
										   output,
										   scale);
	
    if (err != CL_SUCCESS) {
        printf("Error: Failed to set kernel arguments 3! %d\n", err);
        exit(1);
    }
	
	
	err = clEnqueueNDRangeKernel(tc->getMyContext()->cqCommandQueue, 
								 theKernel, 2, NULL, 
								 global_work_size, local_work_size, 0, NULL, NULL);
	
    if (err) {
        printf("Error: Failed to execute kernel! %d\n", err);
        exit(1);
    }
	
    return switchDone;
}




////////////////////////////////////////////////////////////
// convolve_complex_t - convolution driver
////////////////////////////////////////////////////////////


typedef bool (*try_convolution_complex_t_cl) (const DeviceMatrixCL3D::Ptr&,
                                           const DeviceMatrixCL3D::Ptr&,
                                           float,
                                           DeviceMatrixCL3D::Ptr&);
//! A list of algorithms to try
static const try_convolution_complex_t_cl convolution_complex_t_cl[] = {
    &try_convolution_complex_t0_cl,
    &try_convolution_complex_t1_cl
};

/**
 * Perform a 1D (complex) convolution in time.  We expect that
 * video->dim_x == kernel->dim_x, and video->dim_y == kernel->dim_y,
 * and output->dim_t = (video->dim_t - kernel->dim_y + 1).
 *
 * @warning The caller is responsible for making sure that the output
 * is the right size.
 */
void convolve_complex_t_cl(const DeviceMatrixCL3D::Ptr& video,
                        const DeviceMatrixCL3D::Ptr& kernel,
                        float scale,
                        DeviceMatrixCL3D::Ptr& output)
{
    for (int i = NUM_ELEM(convolution_complex_t_cl)-1; i >= 0; i--) {
        if (convolution_complex_t_cl[i](video, kernel, scale, output)) {
            debug_algorithm_used_cl = i;
            return;
        }
    }
    throw_runtime_error("Unable to find convolution algorithm");
}


/**
 * Like convolve_complex_t(), (The same notes and warnings apply.)
 * However, we only try one algorithm before giving up.
 */
void convolve_complex_t_specific_cl(const DeviceMatrixCL3D::Ptr& video,
                                 const DeviceMatrixCL3D::Ptr& kernel,
                                 float scale,
                                 DeviceMatrixCL3D::Ptr& output,
                                 unsigned int algorithm)
{
    assert(algorithm < NUM_ELEM(convolution_complex_t_cl));
    if (convolution_complex_t_cl[algorithm](video, kernel, scale, output)) {
		debug_algorithm_used_cl = algorithm;
		return;
	}
	
    throw_runtime_error("Unable to find convolution algorithm");
}
