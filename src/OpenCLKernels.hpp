#pragma once
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
#include <sys/types.h>
#include <sys/stat.h>
#include <error.h>

static char* load_program_source(const char *filename) {

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
	cl_kernel kernel_list[50];
	cl_program program_list[50]; 
	cl_context GPUContext_K;
	cl_device_id cdDevice_K;
	cl_mem  c_FilterBank;
	cl_mem constant_kernel;
	theKernels(cl_context GPUContext,cl_device_id cdDevice){
		GPUContext_K = GPUContext;
		cdDevice_K   = cdDevice;
		createKernel("pairwiseDistanceKernelGeneric","../../../src/PairwiseDistance.cl",0);
		createKernel("argminKernel","../../../src/argminKernel.cl",1);
		createKernel("argmaxKernel","../../../src/argmaxKernel.cl",2);
		createKernel("minKernel","../../../src/minKernel.cl",3);
		createKernel("maxKernel","../../../src/maxKernel.cl",4);
		createKernel("blockwise_distance_kernel","../../../src/blockwise_distance_kernel.cl",5);
		createKernel("blockwise_filter_kernel","../../../src/blockwise_filter_kernel.cl",6);
		createKernel("cell_histogram_kernel","../../../src/cell_histogram_kernel.cl",7);
		createKernel("cellHistogramKernel1","../../../src/cellHistogramKernel1.cl",8);
		createKernel("cellHistogramKernel2","../../../src/cellHistogramKernel2.cl",9);
		createKernel("do_convolution0","../../../src/do_convolution0.cl",10);
		createKernel("do_convolution1","../../../src/do_convolution1.cl",11);
		createKernel("do_convolution2_8","../../../src/do_convolution2.cl",12);
		createKernel("do_convolution2_10","../../../src/do_convolution2_10.cl",13);
		createKernel("do_convolution2_12","../../../src/do_convolution2_12.cl",14);
		createKernel("do_convolution2_14","../../../src/do_convolution2_14.cl",15);
		createKernel("do_convolution3","../../../src/do_convolution3.cl",16);
		createKernel("do_convolution3_7","../../../src/do_convolution3_7.cl",17);
		createKernel("do_convolution3_9","../../../src/do_convolution3_9.cl",18);
		createKernel("do_convolution3_11","../../../src/do_convolution3_11.cl",19);
		createKernel("do_convolution4","../../../src/do_convolution4.cl",20);
		createKernel("do_convolution4_7","../../../src/do_convolution4_7.cl",21);
		createKernel("do_convolution4_9","../../../src/do_convolution4_9.cl",22);
		createKernel("do_convolution4_11","../../../src/do_convolution4_11.cl",23);
		createKernel("do_convolution4_13","../../../src/do_convolution4_13.cl",24);
		createKernel("do_convolution4_15","../../../src/do_convolution4_15.cl",25);
		createKernel("do_convolution5","../../../src/do_convolution5.cl",26);
		createKernel("do_convolution5_7","../../../src/do_convolution5_7.cl",27);
		createKernel("do_convolution5_9","../../../src/do_convolution5_9.cl",28);
		createKernel("do_convolution5_11","../../../src/do_convolution5_11.cl",29);
		createKernel("do_convolution_complex_t0","../../../src/do_convolution_complex_t0.cl",30);
		createKernel("do_convolution_complex_t1_5","../../../src/do_convolution_complex_t1.cl",31);
		createKernel("do_convolution_complex_t1_7","../../../src/do_convolution_complex_t1_7.cl",32);
		createKernel("do_convolution_complex_t1_9","../../../src/do_convolution_complex_t1_9.cl",33);
		createKernel("do_convolution_complex_t1_11","../../../src/do_convolution_complex_t1_11.cl",34);
	}

	void createKernel(const char * kernel,const char * ruta,int indice){

		//	TheContext* tc = new TheContext();

		//	cl_context GPUContext_K = tc->getMyContext()->getContextCL();
		//	cl_device_id cdDevice_K = tc->getMyContext()->getDeviceCL();

		// Creates the program
		// Uses NVIDIA helper functions to get the code string and it's size (in bytes)
		//size_t src_size = 0;

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
};

class MyKernels{
public:
	static theKernels*  My_Kernels;
	static theKernels*  My_Kernels_TMP;
	MyKernels(cl_context GPUContext_K1,
		cl_device_id cdDevice_K1);

	MyKernels(cl_context GPUContext_K1,
		cl_device_id cdDevice_K1,int cpu);

	void MyKernelsOff(){
		My_Kernels=NULL;
	}

	theKernels * getMyKernels(){
		return My_Kernels;
	}

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

	cl_kernel getBlockWiseDistanceKernel(){
		return My_Kernels->kernel_list[5];
	}

	cl_kernel getBlockWiseFilterKernel(){
		return My_Kernels->kernel_list[6];
	}

	cl_kernel getCellHistogramKernel(){
		return My_Kernels->kernel_list[7];
	}

	cl_kernel getCellHistogramKernel1(){
		return My_Kernels->kernel_list[8];
	}

	cl_kernel getCellHistogramKernel2(){
		return My_Kernels->kernel_list[9];
	}

	cl_kernel getDoConvolution0(){
		return My_Kernels->kernel_list[10];
	}

	cl_kernel getDoConvolution1(){
		return My_Kernels->kernel_list[11];
	}

	cl_kernel getDoConvolution2_8(){
		return My_Kernels->kernel_list[12];
	}
	cl_kernel getDoConvolution2_10(){
		return My_Kernels->kernel_list[13];
	}

	cl_kernel getDoConvolution2_12(){
		return My_Kernels->kernel_list[14];
	}
	cl_kernel getDoConvolution2_14(){
		return My_Kernels->kernel_list[15];
	}
	cl_kernel getDoConvolution3(){
		return My_Kernels->kernel_list[16];
	}

	cl_kernel getDoConvolution3_7(){
		return My_Kernels->kernel_list[17];
	}
	cl_kernel getDoConvolution3_9(){
		return My_Kernels->kernel_list[18];

	}
	cl_kernel getDoConvolution3_11(){
		return My_Kernels->kernel_list[19];
	}

	cl_kernel getDoConvolution4(){
		return My_Kernels->kernel_list[20];
	}
	cl_kernel getDoConvolution4_7(){
		return My_Kernels->kernel_list[21];
	}
	cl_kernel getDoConvolution4_9(){
		return My_Kernels->kernel_list[22];
	}
	cl_kernel getDoConvolution4_11(){
		return My_Kernels->kernel_list[23];
	}
	cl_kernel getDoConvolution4_13(){
		return My_Kernels->kernel_list[24];
	}
	cl_kernel getDoConvolution4_15(){
		return My_Kernels->kernel_list[25];
	}

	cl_kernel getDoConvolution5(){
		return My_Kernels->kernel_list[26];
	}

	cl_kernel getDoConvolution5_7(){
		return My_Kernels->kernel_list[27];
	}

	cl_kernel getDoConvolution5_9(){
		return My_Kernels->kernel_list[28];
	}

	cl_kernel getDoConvolution5_11(){
		return My_Kernels->kernel_list[29];
	}

	cl_kernel getDoConvolutionComplexT0(){
		return My_Kernels->kernel_list[30];
	}

	cl_kernel getDoConvolutionComplexT1_5(){
		return My_Kernels->kernel_list[31];
	}
	cl_kernel getDoConvolutionComplexT1_7(){
		return My_Kernels->kernel_list[32];
	}

	cl_kernel getDoConvolutionComplexT1_9(){
		return My_Kernels->kernel_list[33];
	}

	cl_kernel getDoConvolutionComplexT1_11(){
		return My_Kernels->kernel_list[34];
	}

	~MyKernels(){};
};



class MyKernels_CPU{
public:
	static theKernels*  My_Kernels;
	MyKernels_CPU(cl_context GPUContext_K1,
		cl_device_id cdDevice_K1);

	theKernels * getMyKernels(){
		return My_Kernels;
	}

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

	cl_kernel getBlockWiseDistanceKernel(){
		return My_Kernels->kernel_list[5];
	}

	cl_kernel getBlockWiseFilterKernel(){
		return My_Kernels->kernel_list[6];
	}

	cl_kernel getCellHistogramKernel(){
		return My_Kernels->kernel_list[7];
	}

	cl_kernel getCellHistogramKernel1(){
		return My_Kernels->kernel_list[8];
	}

	cl_kernel getCellHistogramKernel2(){
		return My_Kernels->kernel_list[9];
	}

	cl_kernel getDoConvolution0(){
		return My_Kernels->kernel_list[10];
	}

	cl_kernel getDoConvolution1(){
		return My_Kernels->kernel_list[11];
	}

	cl_kernel getDoConvolution2_8(){
		return My_Kernels->kernel_list[12];
	}
	cl_kernel getDoConvolution2_10(){
		return My_Kernels->kernel_list[13];
	}

	cl_kernel getDoConvolution2_12(){
		return My_Kernels->kernel_list[14];
	}
	cl_kernel getDoConvolution2_14(){
		return My_Kernels->kernel_list[15];
	}
	cl_kernel getDoConvolution3(){
		return My_Kernels->kernel_list[16];
	}

	cl_kernel getDoConvolution3_7(){
		return My_Kernels->kernel_list[17];
	}
	cl_kernel getDoConvolution3_9(){
		return My_Kernels->kernel_list[18];

	}
	cl_kernel getDoConvolution3_11(){
		return My_Kernels->kernel_list[19];
	}

	cl_kernel getDoConvolution4(){
		return My_Kernels->kernel_list[20];
	}
	cl_kernel getDoConvolution4_7(){
		return My_Kernels->kernel_list[21];
	}
	cl_kernel getDoConvolution4_9(){
		return My_Kernels->kernel_list[22];
	}
	cl_kernel getDoConvolution4_11(){
		return My_Kernels->kernel_list[23];
	}
	cl_kernel getDoConvolution4_13(){
		return My_Kernels->kernel_list[24];
	}
	cl_kernel getDoConvolution4_15(){
		return My_Kernels->kernel_list[25];
	}

	cl_kernel getDoConvolution5(){
		return My_Kernels->kernel_list[26];
	}

	cl_kernel getDoConvolution5_7(){
		return My_Kernels->kernel_list[27];
	}

	cl_kernel getDoConvolution5_9(){
		return My_Kernels->kernel_list[28];
	}

	cl_kernel getDoConvolution5_11(){
		return My_Kernels->kernel_list[29];
	}

	cl_kernel getDoConvolutionComplexT0(){
		return My_Kernels->kernel_list[30];
	}

	cl_kernel getDoConvolutionComplexT1_5(){
		return My_Kernels->kernel_list[31];
	}
	cl_kernel getDoConvolutionComplexT1_7(){
		return My_Kernels->kernel_list[32];
	}

	cl_kernel getDoConvolutionComplexT1_9(){
		return My_Kernels->kernel_list[33];
	}

	cl_kernel getDoConvolutionComplexT1_11(){
		return My_Kernels->kernel_list[34];
	}

	~MyKernels_CPU(){};
};
