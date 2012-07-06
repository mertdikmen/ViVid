#include "PairwiseDistance.hpp"
#include "PairwiseDistanceLocal.hpp"
#include "DeviceMatrixOpenCL.hpp"


DeviceMatrix::Ptr pwdist_cuda( const DeviceMatrix::Ptr& features_train,
                               const DeviceMatrix::Ptr& features_test){

    DeviceMatrix::Ptr out = makeDeviceMatrix(features_train->height,
                                             features_test->height);
    pwdist_generic(features_train.get(), features_test.get(), out.get(), EUCLIDEAN);
    return out;
}






DeviceMatrix::Ptr pwdot_cuda( const DeviceMatrix::Ptr& features_train,
                               const DeviceMatrix::Ptr& features_test){

    DeviceMatrix::Ptr out = makeDeviceMatrix(features_train->height,
                                             features_test->height);
    pwdist_generic(features_train.get(), features_test.get(), out.get(), DOTPRODUCT);
    return out;
}






DeviceMatrix::Ptr pwabsdot_cuda( const DeviceMatrix::Ptr& features_train,
                               const DeviceMatrix::Ptr& features_test){

    DeviceMatrix::Ptr out = makeDeviceMatrix(features_train->height,
                                             features_test->height);
    pwdist_generic(features_train.get(), features_test.get(), out.get(), ABSDOTPRODUCT);
    return out;
}




DeviceMatrix::Ptr pwchisq_cuda( const DeviceMatrix::Ptr& features_train,
                              const DeviceMatrix::Ptr& features_test){

    DeviceMatrix::Ptr out = makeDeviceMatrix(features_train->height,
                                             features_test->height);
    pwdist_generic(features_train.get(), features_test.get(), out.get(), CHISQUARED);
    return out;
}




DeviceMatrix::Ptr pwcityblock_cuda( const DeviceMatrix::Ptr& features_train,
                                    const DeviceMatrix::Ptr& features_test){

    DeviceMatrix::Ptr out = makeDeviceMatrix(features_train->height,
                                             features_test->height);
    pwdist_generic(features_train.get(), features_test.get(), out.get(), CITYBLOCK);
    return out;
}




DeviceMatrix::Ptr argmin_cuda(const DeviceMatrix::Ptr& matrix)
{
    DeviceMatrix::Ptr out = makeDeviceMatrix(matrix->height, 1);
    argmin_cuda_local(matrix.get(), out.get());
    return out;
}

DeviceMatrix::Ptr argmax_cuda(const DeviceMatrix::Ptr& matrix)
{
    DeviceMatrix::Ptr out = makeDeviceMatrix(matrix->height, 1);
    argmax_cuda_local(matrix.get(), out.get());
    return out;
}

DeviceMatrix::Ptr min_cuda(const DeviceMatrix::Ptr& matrix)
{
    DeviceMatrix::Ptr out = makeDeviceMatrix(matrix->height, 1);
    min_cuda_local(matrix.get(), out.get());
    return out;
}

DeviceMatrix::Ptr max_cuda(const DeviceMatrix::Ptr& matrix)
{
    DeviceMatrix::Ptr out = makeDeviceMatrix(matrix->height, 1);
    max_cuda_local(matrix.get(), out.get());
    return out;
}




/**
 
 
 OpenCL function
 
 **/



DeviceMatrixCL::Ptr pwdist_cl( const DeviceMatrixCL::Ptr& features_train,
							  const DeviceMatrixCL::Ptr& features_test){
	
    DeviceMatrixCL::Ptr out = makeDeviceMatrixCL(features_train->height,
												 features_test->height);
	
    pwdist_genericCL(features_train.get(), features_test.get(), out.get(), EUCLIDEAN);
    return out;
}


DeviceMatrixCL::Ptr pwdot_cl( const DeviceMatrixCL::Ptr& features_train,
							 const DeviceMatrixCL::Ptr& features_test){
	
    DeviceMatrixCL::Ptr out = makeDeviceMatrixCL(features_train->height,
												 features_test->height);
    pwdist_genericCL(features_train.get(), features_test.get(), out.get(), DOTPRODUCT);
    return out;
}

DeviceMatrixCL::Ptr pwabsdot_cl( const DeviceMatrixCL::Ptr& features_train,
								const DeviceMatrixCL::Ptr& features_test){
	
    DeviceMatrixCL::Ptr out = makeDeviceMatrixCL(features_train->height,
												 features_test->height);
    pwdist_genericCL(features_train.get(), features_test.get(), out.get(), ABSDOTPRODUCT);
    return out;
}


DeviceMatrixCL::Ptr pwchisq_cl( const DeviceMatrixCL::Ptr& features_train,
							   const DeviceMatrixCL::Ptr& features_test){
	
    DeviceMatrixCL::Ptr out = makeDeviceMatrixCL(features_train->height,
												 features_test->height);
    pwdist_genericCL(features_train.get(), features_test.get(), out.get(), CHISQUARED);
    return out;
}




DeviceMatrixCL::Ptr pwcityblock_cuda( const DeviceMatrixCL::Ptr& features_train,
								   const DeviceMatrixCL::Ptr& features_test){
	
    DeviceMatrixCL::Ptr out = makeDeviceMatrixCL(features_train->height,
                                             features_test->height);
    pwdist_genericCL(features_train.get(), features_test.get(), out.get(), CITYBLOCK);
    return out;
}




/**


This code was in the cu file of the cuda version

**/



static const unsigned int BLOCK_SIZE = 16;

void pwdist_genericCL( const DeviceMatrixCL* features_train,
					  const DeviceMatrixCL* features_test,
					  DeviceMatrixCL* output,
					  int type) {
	
    //dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    //int grid_ry = (features_train->height-1) / dimBlock.y + 1;
    //int grid_cx = (features_test->height-1) / dimBlock.x + 1;
    //dim3 dimGrid(grid_cx, grid_ry);
	
	// pairwiseDistanceKernelGeneric<<<dimGrid, dimBlock>>>(*features_train,*features_test,*output,type);
    //cudaThreadSynchronize();
	
	
	
    TheContext * tc = new TheContext();
	
    cl_context GPUContext = tc->getMyContext()->getContextCL();
    cl_device_id cdDevice = tc->getMyContext()->getDeviceCL();
	
	
	
	// Creates the program
	// Uses NVIDIA helper functions to get the code string and it's size (in bytes)
	size_t src_size = 0;
	
	
	
	char *program_source = load_program_source("/home/antigm/vivid/src/PairwiseDistance.cl");
	if (program_source == NULL) {
        printf("Error: Failed to read the OpenCL kernel: kernel.cl\n");
        exit(-1);
	}
	cl_int err;
	cl_program program_list[32]; 
	program_list[0] = clCreateProgramWithSource(GPUContext, 1, (const char **) &program_source, NULL, &err);
	if (!program_list[0]) {
		printf("Error: Failed to create compute program for device %d!\n", 0);
		printf("************\n%s\n************\n", program_source);
	}
	
	// Build the program executable
	//
	const char * options = "";
	err = clBuildProgram(program_list[0], 0, NULL, options, NULL, NULL);
	if (err != CL_SUCCESS) {
		size_t len;
		char buffer[2048];
		
		printf("Error: Failed to build program executable for device %d!\n",err);
		clGetProgramBuildInfo(program_list[0], cdDevice, CL_PROGRAM_BUILD_LOG, sizeof (buffer), buffer, &len);
		printf("%s\n", buffer);
		
	}
	
	
	
	cl_kernel kernel_list[32];
	
	
	
	kernel_list[0] = clCreateKernel(program_list[0], "pairwiseDistanceKernelGeneric", &err);
	if (!kernel_list[0] || err != CL_SUCCESS) {
		printf("Error: Failed to create compute kernel for device %d!\n", 0);
		exit(1);
	}
	err=0;
	
	err |=   clSetKernelArg(kernel_list[0], 0, sizeof (cl_mem), &features_train->dataMatrix);
	err |=   clSetKernelArg(kernel_list[0], 1, sizeof (int), &features_train->width);
	err|=  clSetKernelArg(kernel_list[0], 2, sizeof (int), &features_train->height);
	err |= clSetKernelArg(kernel_list[0],3, sizeof (int), &features_train->pitch);
	
	
	err |=   clSetKernelArg(kernel_list[0], 4, sizeof (cl_mem), &features_test->dataMatrix);
	err |=   clSetKernelArg(kernel_list[0], 5, sizeof (int), &features_test->width);
	err|=  clSetKernelArg(kernel_list[0], 6, sizeof (int), &features_test->height);
	err |= clSetKernelArg(kernel_list[0],7, sizeof (int), &features_test->pitch);
	
	err |=   clSetKernelArg(kernel_list[0], 8, sizeof (cl_mem), &output->dataMatrix);
	err |=   clSetKernelArg(kernel_list[0], 9, sizeof (int), &output->width);
	err|=  clSetKernelArg(kernel_list[0], 10, sizeof (int), &output->height);
	err |= clSetKernelArg(kernel_list[0],11, sizeof (int), &output->pitch);
	
	err |= clSetKernelArg(kernel_list[0], 12, sizeof (int), &type);
	err |= clSetKernelArg(kernel_list[0], 13, sizeof (int), &BLOCK_SIZE);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to set kernel arguments 3! %d\n", err);
        exit(1);
    }
	
	
	
	
	/*
    err |= clSetKernelArg(kernel_list[0], 1, sizeof (DeviceMatrixCL), &features_train);	
	if (err != CL_SUCCESS) {
        printf("Error: Failed to set kernel arguments 1! %d\n", err);
        exit(1);
    }
    err |= clSetKernelArg(kernel_list[0], 2, sizeof (DeviceMatrixCL), &features_train);
	
	if (err != CL_SUCCESS) {
        printf("Error: Failed to set kernel arguments 2! %d\n", err);
        exit(1);
    }
    err |= clSetKernelArg(kernel_list[0], 3, sizeof (int), &type);
	
    if (err != CL_SUCCESS) {
        printf("Error: Failed to set kernel arguments 3! %d\n", err);
        exit(1);
    }
	*/
		const size_t local_work_size[2]={BLOCK_SIZE,BLOCK_SIZE}; 
		const size_t global_work_size[2]={ceil((float)features_train->width/BLOCK_SIZE)*BLOCK_SIZE,ceil((float)features_train->height/BLOCK_SIZE)*BLOCK_SIZE}; 
		
	
	err = clEnqueueNDRangeKernel(tc->getMyContext()->cqCommandQueue, kernel_list[0], 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
    if (err) {
        printf("Error: Failed to execute kernel! %d\n", err);
        exit(1);
    }
}
