//******************************************************************* 
// Demo OpenCL application to compute a simple vector addition 
// computation between 2 arrays on the GPU // ****************************************************************** 
#include <stdio.h>
#include <stdlib.h> 
#include <CL/cl.h>
#include <cuda_runtime.h>
// OpenCL source code
const char* OpenCLSource[] = { 
	"__kernel void VectorAdd(__global int* c, __global int* a,__global int* b)",
	"{",
	"	// Index of the elements to add \n", 
	"	unsigned int n = get_global_id(0);", 
	"	// Sum the n’th element of vectors a and b and store in c \n", 
	"	c[n] = a[n] + b[n];",
	"}"
};

// Some interesting data for the vectors
int InitialData1[20] = {37,50,54,50,56,0,43,43,74,71,32,36,16,43,56,100,50,25,15,17}; 
int InitialData2[20] = {35,51,54,58,55,32,36,69,27,39,35,40,16,44,55,14,58,75,18,15};

// Number of elements in the vectors to be added
#define SIZE 2048

// Main function 
// ********************************************************************* 
int main(int argc, char **argv)
{
	int HostVector1[SIZE], HostVector2[SIZE];
	cl_int errorcode;

	int deviceCount= 0;
	cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

	// Initialize with some interesting repeating data
	for(int c = 0; c < SIZE; c++) {
		HostVector1[c] = InitialData1[c%20]; 
		HostVector2[c] = InitialData2[c%20];
	}

	//Get an OpenCL platform
	cl_platform_id cpPlatform; 
	if (clGetPlatformIDs(1, &cpPlatform, NULL) != CL_SUCCESS) {
		printf("clGetPlatformIDs error\n");
	}
	
	// Get a GPU device
	cl_device_id cdDevice; 
	if (clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &cdDevice, NULL) != CL_SUCCESS) {
		printf("clGetDeviceIDs error\n");
	}

	// Create a context to run OpenCL on our CUDA-enabled NVIDIA GPU
	//cl_context GPUContext = clCreateContextFromType(0, CL_DEVICE_TYPE_GPU, NULL, NULL, &errorcode); 
	cl_context GPUContext = clCreateContext(0, 1, &cdDevice, NULL, NULL, &errorcode); 
	if (GPUContext == NULL) {
		printf("clCreateContextFromType error: ");
		if (errorcode == CL_INVALID_PLATFORM) printf("invalid platform\n");
		if (errorcode == CL_INVALID_VALUE) printf("invalid value\n");
		if (errorcode == CL_DEVICE_NOT_AVAILABLE) printf("device not available\n");
		if (errorcode == CL_DEVICE_NOT_FOUND) printf("device not found\n");
		if (errorcode == CL_OUT_OF_HOST_MEMORY) printf("out of host memory\n");
		if (errorcode == CL_INVALID_DEVICE_TYPE) printf("invalid device type\n");
		exit(1);
	}

	// Create a command-queue on the GPU device
	cl_command_queue cqCommandQueue = clCreateCommandQueue(GPUContext, cdDevice, 0, NULL);
	if (cqCommandQueue == NULL) {
		printf("clCreateCommandQueue error\n");
	}

	// Allocate GPU memory for source vectors AND initialize from CPU memory
	cl_mem GPUVector1 = clCreateBuffer(GPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * SIZE, HostVector1, NULL);
	cl_mem GPUVector2 = clCreateBuffer(GPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * SIZE, HostVector2, NULL);

	// Allocate output memory on GPU
	cl_mem GPUOutputVector = clCreateBuffer(GPUContext, CL_MEM_WRITE_ONLY, sizeof(int) * SIZE, NULL, NULL);
	if (GPUVector1 == NULL || GPUVector2 == NULL || GPUOutputVector == NULL) {
		printf("clCreateBuffer error\n");
	}

	// Create OpenCL program with source code
	cl_program OpenCLProgram = clCreateProgramWithSource(GPUContext, 7, OpenCLSource, NULL, NULL);
	if (OpenCLProgram == NULL) {
		printf("clCreateProgramWithSource error\n");
	}

	// Build the program (OpenCL JIT compilation)
	if (clBuildProgram(OpenCLProgram, 0, NULL, NULL, NULL, NULL) != CL_SUCCESS) {
		printf("clBuildProgram erro\n");
	} 

	// Create a handle to the compiled OpenCL function (Kernel)
	cl_kernel OpenCLVectorAdd = clCreateKernel(OpenCLProgram, "VectorAdd", NULL);
	if (OpenCLVectorAdd == NULL) {
		printf("clCreateKernel error\n");
	}

	// In the next step we associate the GPU memory with the Kernel arguments
	if (clSetKernelArg(OpenCLVectorAdd, 0, sizeof(cl_mem), (void*)&GPUOutputVector) != CL_SUCCESS || 
	clSetKernelArg(OpenCLVectorAdd, 1, sizeof(cl_mem), (void*)&GPUVector1) != CL_SUCCESS || 
	clSetKernelArg(OpenCLVectorAdd, 2, sizeof(cl_mem), (void*)&GPUVector2) != CL_SUCCESS) {
		printf("clSetKernelArg error\n");
	}

	// Launch the Kernel on the GPU 
	size_t WorkSize[1] = {SIZE}; 
	// one dimensional Range 
	if (clEnqueueNDRangeKernel(cqCommandQueue, OpenCLVectorAdd, 1, NULL, WorkSize, NULL, 0, NULL, NULL) != CL_SUCCESS) {
		printf("clEnqueueNDRangeKernel error \n");
	}

	// Copy the output in GPU memory back to CPU memory
	int HostOutputVector[SIZE]; 
	if (clEnqueueReadBuffer(cqCommandQueue, GPUOutputVector, CL_TRUE, 0, SIZE * sizeof(int), HostOutputVector, 0, NULL, NULL) != CL_SUCCESS) {
		printf("clEnqueueReadBuffer error\n");
	}

	// Cleanup
	clReleaseKernel(OpenCLVectorAdd); 
	clReleaseProgram(OpenCLProgram); 
	clReleaseCommandQueue(cqCommandQueue); 
	clReleaseContext(GPUContext); 
	clReleaseMemObject(GPUVector1); 
	clReleaseMemObject(GPUVector2); 
	clReleaseMemObject(GPUOutputVector);

	// Print out the results
	for (int Rows = 0; Rows < (SIZE/20); Rows++, printf("\t")){ 
		for(int c = 0; c <20; c++){
			printf("%c",(char)HostOutputVector[Rows * 20 + c]); 
		}
	}
	printf("\n\nThe End\n\n");
	while(1);
	return 0;
}