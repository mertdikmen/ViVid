#pragma once
#include "CL/cl.h"
#include <stdio.h>
#include <iostream>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <error.h>
#include "OpenCLTypes.hpp"

#define NUM_MAX_KERNELS 50

static char* load_program_source(const char *filename) 
{
	struct stat statbuf;
	FILE *fh;
	char *source;

	fh = fopen(filename, "r");
	if (fh == 0)
		return 0;

	stat(filename, &statbuf);
	source = (char *) malloc(statbuf.st_size + 1);
	fread(source, statbuf.st_size, 1, fh);
	source[statbuf.st_size] = '\0';

	return source;
}

class ViVidCLKernels
{
public:
	cl_mem  c_FilterBank;

	ViVidCLKernels(cl_context context, vivid::DeviceType device_type);
	void createKernel(cl_context, const char* kernel, const char* path, int indice);

	cl_kernel getPairwiseDistanceKernel(){ return kernel_list[0];}
	cl_kernel getArgminKernel(){ return kernel_list[1];}
	cl_kernel getArgmaxKernel(){ return kernel_list[2];}
	cl_kernel getMinKernel(){ return kernel_list[3];}
	cl_kernel getMaxKernel(){ return kernel_list[4];}
	cl_kernel getBlockWiseDistanceKernel(){ return kernel_list[5];}
	cl_kernel getBlockWiseFilterKernel(){return kernel_list[6];}
	cl_kernel getCellHistogramKernel(){ return kernel_list[7];}
	cl_kernel getCellHistogramKernel1(){ return kernel_list[8];}
	cl_kernel getCellHistogramKernel2(){ return kernel_list[9];}
	cl_kernel getCellHistogramKernel3(){ return kernel_list[10];}

private:
	cl_kernel kernel_list[NUM_MAX_KERNELS];
	bool kernel_ready[NUM_MAX_KERNELS];
	cl_program program_list[NUM_MAX_KERNELS]; 
	cl_device_id cdDevice_K;
	cl_mem constant_kernel;
};

/*
struct theKernels {
	cl_kernel kernel_list[50];
	cl_program program_list[50]; 
	vivid::ContexOpenCl* my_context;
	cl_device_id cdDevice_K;
	cl_mem  c_FilterBank;
	cl_mem constant_kernel;

	theKernels(vivid::ContexOpenCl* context)
	{
		my_context = context;
	
		if (my_context->getDeviceType() == vivid::DEVICE_GPU)
			createKernel("pairwiseDistanceKernel","../../../src/E_PairwiseDistance.cl",0);
		else if (my_context->getDeviceType() == vivid::DEVICE_CPU)
			createKernel("pairwiseDistanceKernel","../../../src/CPU_PairwiseDistance.cl",0);

		createKernel("argminKernel","../../../src/argminKernel.cl",1);
		createKernel("argmaxKernel","../../../src/argmaxKernel.cl",2);
		createKernel("minKernel","../../../src/minKernel.cl",3);
		createKernel("maxKernel","../../../src/maxKernel.cl",4);
		
		if (my_context->getDeviceType() == vivid::DEVICE_GPU)
			createKernel("blockwise_distance_kernel","../../../src/E_blockwise_distance_kernel.cl",5);
		else if (my_context->getDeviceType() == vivid::DEVICE_CPU)
			createKernel("blockwise_distance_kernel","../../../src/CPU_blockwise_distance_kernel.cl",5);
		
		createKernel("blockwise_filter_kernel","../../../src/blockwise_filter_kernel.cl",6);
		createKernel("cell_histogram_kernel","../../../src/cell_histogram_kernel.cl",7);
		createKernel("cellHistogramKernel1","../../../src/cellHistogramKernel1.cl",8);
		createKernel("cellHistogramKernel2","../../../src/cellHistogramKernel2.cl",9);
		createKernel("cellHistogramKernel3","../../../src/cellHistogramKernel3.cl",10);
	}

	void createKernel(const char* kernel, const char* path, int indice){
        char full_path[256];
		sprintf(full_path, "%s", path);

		char *program_source = load_program_source(full_path);
		if (program_source == NULL) {
			printf("Error: Failed to read the OpenCL kernel: %s\n",path);
			exit(-1);
		}
		cl_int err;

		program_list[indice] = clCreateProgramWithSource(my_context->getContextCL(), 1, (const char **) &program_source, NULL, &err);
		if (!program_list[indice]) {
			printf("Error: Failed to create compute program for device %d Kernel: (%s)!\n", indice,kernel);
			printf("************\n%s\n************\n", program_source);
		}

		// Build the program executable
		const char * options = "-cl-fast-relaxed-math";
		err = clBuildProgram(program_list[indice], 0, NULL, options, NULL, NULL);
		if (err != CL_SUCCESS) {
			size_t len;
			char buffer[10000];

			printf("Error: Failed to build program executable for device %d kernel: (%s)!\n",err,kernel);
			cl_int get_err=clGetProgramBuildInfo(program_list[indice], cdDevice_K, CL_PROGRAM_BUILD_LOG, sizeof (buffer), buffer, &len);
			printf("%d %s\n", get_err, buffer);

		}

		kernel_list[indice] = clCreateKernel(program_list[indice], kernel, &err);
		if (!kernel_list[indice] || err != CL_SUCCESS) {
			printf("Error: Failed to create compute kernel for device %d Kernel: (%s)!\n", indice, full_path);
			exit(1);
		}
	}
};
*/



/*
class MyKernels{
public:
	static theKernels*  My_Kernels;
	static theKernels*  My_Kernels_TMP;
	MyKernels(cl_context GPUContext_K1, cl_device_id cdDevice_K1);
	MyKernels(cl_context GPUContext_K1, cl_device_id cdDevice_K1,int cpu);
	void MyKernelsOff(){ My_Kernels=NULL; }

	theKernels * getMyKernels(){
		return My_Kernels;
	}

	cl_kernel getPairwiseDistanceKernel(){ return My_Kernels->kernel_list[0];}
	cl_kernel getArgminKernel(){ return My_Kernels->kernel_list[1];}
	cl_kernel getArgmaxKernel(){ return My_Kernels->kernel_list[2];}
	cl_kernel getMinKernel(){ return My_Kernels->kernel_list[3];}
	cl_kernel getMaxKernel(){ return My_Kernels->kernel_list[4];}
	cl_kernel getBlockWiseDistanceKernel(){ return My_Kernels->kernel_list[5];}
	cl_kernel getBlockWiseFilterKernel(){return My_Kernels->kernel_list[6];}
	cl_kernel getCellHistogramKernel(){ return My_Kernels->kernel_list[7];}
	cl_kernel getCellHistogramKernel1(){ return My_Kernels->kernel_list[8];}
	cl_kernel getCellHistogramKernel2(){ return My_Kernels->kernel_list[9];}
	cl_kernel getCellHistogramKernel3(){ return My_Kernels->kernel_list[10];}

	~MyKernels(){};
};

class MyKernels_CPU{
public:
	static theKernels*  My_Kernels;
	MyKernels_CPU(cl_context GPUContext_K1, cl_device_id cdDevice_K1);

	theKernels * getMyKernels(){return My_Kernels;}
	cl_kernel getPairwiseDistanceKernel(){return My_Kernels->kernel_list[0];}
	cl_kernel getArgminKernel(){return My_Kernels->kernel_list[1];}
	cl_kernel getArgmaxKernel(){return My_Kernels->kernel_list[2];}
	cl_kernel getMinKernel(){return My_Kernels->kernel_list[3];}
	cl_kernel getMaxKernel(){return My_Kernels->kernel_list[4];}
	cl_kernel getBlockWiseDistanceKernel(){return My_Kernels->kernel_list[5];}
	cl_kernel getBlockWiseFilterKernel(){return My_Kernels->kernel_list[6];}
	cl_kernel getCellHistogramKernel(){return My_Kernels->kernel_list[7];}
	cl_kernel getCellHistogramKernel1(){return My_Kernels->kernel_list[8];}
	cl_kernel getCellHistogramKernel2(){return My_Kernels->kernel_list[9];}

	~MyKernels_CPU(){};
};

*/
