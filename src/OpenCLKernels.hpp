/*
 *  OpenCLKernels.hpp
 *  
 *
 *  Created by Antonio García Martín on 09/07/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */


#include "CL/cl.h"
#include <stdio.h>
#include <iostream>


#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <error.h>


static char * load_program_source(const char *filename) {
	
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




struct theKernels {
	
	cl_kernel kernel_list[32];
	cl_program program_list[32]; 
	cl_context GPUContext_K;
	cl_device_id cdDevice_K;
	theKernels(cl_context GPUContext,cl_device_id cdDevice){
		GPUContext_K = GPUContext;
		cdDevice_K   = cdDevice;
		createKernel("pairwiseDistanceKernelGeneric","../src/PairwiseDistance.cl",0);
		createKernel("argminKernel","../src/argminKernel.cl",1);
		createKernel("argmaxKernel","../src/argmaxKernel.cl",2);
		createKernel("minKernel","../src/minKernel.cl",2);
		createKernel("maxKernel","../src/maxKernel.cl",3);
	}
	
	void createKernel(const char * kernel,const char * ruta,int indice){
		
	//	TheContext* tc = new TheContext();
		
	//	cl_context GPUContext_K = tc->getMyContext()->getContextCL();
	//	cl_device_id cdDevice_K = tc->getMyContext()->getDeviceCL();
		
		// Creates the program
		// Uses NVIDIA helper functions to get the code string and it's size (in bytes)
		size_t src_size = 0;
		
		
		char *program_source = load_program_source(ruta);
		if (program_source == NULL) {
			printf("Error: Failed to read the OpenCL kernel $s: kernel.cl\n",kernel);
			exit(-1);
		}
		cl_int err;
		
		program_list[indice] = clCreateProgramWithSource(GPUContext_K, 1, (const char **) &program_source, NULL, &err);
		if (!program_list[indice]) {
			printf("Error: Failed to create compute program for device %d Kernel: (%s)!\n", indice,kernel);
			printf("************\n%s\n************\n", program_source);
		}
		
		// Build the program executable
		const char * options = "";
		err = clBuildProgram(program_list[indice], 0, NULL, options, NULL, NULL);
		if (err != CL_SUCCESS) {
			size_t len;
			char buffer[2048];
			
			printf("Error: Failed to build program executable for device %d kernel: (%s)!\n",err,kernel);
			clGetProgramBuildInfo(program_list[indice], cdDevice_K, CL_PROGRAM_BUILD_LOG, sizeof (buffer), buffer, &len);
			printf("%s\n", buffer);
			
		}
		
		
		
		kernel_list[indice] = clCreateKernel(program_list[indice], kernel, &err);
		if (!kernel_list[indice] || err != CL_SUCCESS) {
			printf("Error: Failed to create compute kernel for device %d Kernel: (%s)!\n", indice,kernel);
			exit(1);
		}
		
	}
	
	

	
}  ;



class MyKernels{
	
 	
public:
	static theKernels *  My_Kernels;
	
	MyKernels(cl_context GPUContext_K1,
			  cl_device_id cdDevice_K1);
	theKernels * getMyKernels();
	cl_kernel getPairwiseDistanceKernel(){
		return My_Kernels->kernel_list[0];
	}
	
	
	cl_kernel getArgminKernel(){
		return My_Kernels->kernel_list[1];
	}
	
	
	cl_kernel getArgmaxKernel(){
		return My_Kernels->kernel_list[2];
	}
	
	
	cl_kernel getMinKernel(){
		return My_Kernels->kernel_list[3];
	}
	
	
	cl_kernel getMaxKernel(){
		return My_Kernels->kernel_list[4];
	}
	
	~MyKernels(){};
};



