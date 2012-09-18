/*
 *  ContextOpenCL.h
 *  
 *
 *  Created by Antonio García Martín on 28/06/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
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
using namespace std;


#include <CL/cl.h>
#include <boost/shared_ptr.hpp>


//Get an OpenCL platform
struct myContexOpenCl {
	
	cl_platform_id cpPlatform; 
	cl_device_id cdDevice; 
	cl_command_queue cqCommandQueue;
	cl_context Context;
	myContexOpenCl(int type){
		
#define NUM_ENTRIES 10

		cl_device_id cdDeviceTEST[NUM_ENTRIES];
		cl_platform_id cpPlatformTEST[NUM_ENTRIES];
		cl_uint n_devices;
		cl_uint n_platforms;

		cl_int errorcode;

		clGetPlatformIDs(NUM_ENTRIES, cpPlatformTEST, &n_platforms);

		printf("NUM PLATFORMS: %d\n", n_platforms);

	//	if (cpPlatform==NULL){
			printf("Creating platform\n");
			if (clGetPlatformIDs(1, &cpPlatform, NULL) != CL_SUCCESS) {
				printf("clGetPlatformIDs error\n");
			}
			
	//	}
		
			   size_t returned_size = 0;
    cl_char vendor_name[NUM_ENTRIES][1024];
    cl_char device_name[NUM_ENTRIES][1024];

	//	if(cdDevice==NULL){
			
			if(type==CL_DEVICE_TYPE_GPU){
				printf("Getting GPU Device\n");
				if (clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, 
							   &cdDevice, NULL) != CL_SUCCESS) {
				printf("clGetDeviceIDs con el device error\n");
				}
			
			}else{
				printf("Getting CPU device\n");
				const cl_device_type opncl_device_type[]={CL_DEVICE_TYPE_CPU,CL_DEVICE_TYPE_GPU};
				if (clGetDeviceIDs(cpPlatform, opncl_device_type[0], NUM_ENTRIES, cdDeviceTEST, &n_devices) == CL_SUCCESS)
				{
					printf("NUM DEVICES: %d\n", n_devices);
					for (int i=0; i<n_devices; i++)
					{
						clGetDeviceInfo(cdDeviceTEST[i], CL_DEVICE_NAME, sizeof (vendor_name[i]), vendor_name[i], &returned_size);
						printf("Device Vendor: %d\t%s\n", i, vendor_name[i]);
					}

				}

				if (clGetDeviceIDs(cpPlatform,  CL_DEVICE_TYPE_CPU , 1, 
							   &cdDevice, NULL) != CL_SUCCESS) {
				printf("clGetDeviceIDs con el device error\n");
				}
			}
	//	}
	
		
		
	//	if(GPUContext==NULL){
			printf("Making context\n");
			Context = clCreateContext(0, 1, &cdDevice,
														  NULL, NULL, &errorcode); 
			if (Context == NULL) {
				printf("clCreateContextFromType error: %d",errorcode);
				if (errorcode == CL_INVALID_PLATFORM) 
					printf("invalid platform\n");
				if (errorcode == CL_INVALID_VALUE) printf("invalid value\n");
				if (errorcode == CL_DEVICE_NOT_AVAILABLE) 
					printf("device not available\n");
				if (errorcode == CL_DEVICE_NOT_FOUND)
					printf("device not found\n");
				if (errorcode == CL_OUT_OF_HOST_MEMORY)
					printf("out of host memory\n");
				if (errorcode == CL_INVALID_DEVICE_TYPE) 
					printf("invalid device type\n");
				exit(1);
			}
	//	}
		
	//	if(cqCommandQueue==NULL){
			printf("Making Command Queue\n");
			cqCommandQueue = 
			clCreateCommandQueue(Context, cdDevice, 0, NULL);
			if (cqCommandQueue == NULL) {
				printf("clCreateCommandQueue error\n");
			}
	//	}
		
	
		
		
	}
	
	cl_device_id getDeviceCL(){
		
		if(cdDevice==NULL){
			if (clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, 
							   &cdDevice, NULL) != CL_SUCCESS) {
				printf("clGetDeviceIDs error\n");
			}
			
		}
		
		return cdDevice;
		
	}
	
	cl_context getContextCL(){
		cl_int errorcode;

		if(Context==NULL){
			Context = clCreateContext(0, 1, &cdDevice,
										 NULL, NULL, &errorcode); 
			if (Context == NULL) {
				printf("clCreateContextFromType error: ");
				if (errorcode == CL_INVALID_PLATFORM) 
					printf("invalid platform\n");
				if (errorcode == CL_INVALID_VALUE) printf("invalid value\n");
				if (errorcode == CL_DEVICE_NOT_AVAILABLE) 
					printf("device not available\n");
				if (errorcode == CL_DEVICE_NOT_FOUND)
					printf("device not found\n");
				if (errorcode == CL_OUT_OF_HOST_MEMORY)
					printf("out of host memory\n");
				if (errorcode == CL_INVALID_DEVICE_TYPE) 
					printf("invalid device type\n");
				exit(1);
			}
		}		
		return Context;
		
	}
	
}  ;




class TheContext{
	
 	
public:
	static myContexOpenCl *  The_Context_GPU;
	static myContexOpenCl* The_Context_CPU;
	TheContext();
	myContexOpenCl * getMyContext();
	myContexOpenCl * getMyContextCPU();
	~TheContext(){};
};



//boost::shared_ptr<myContexOpenCl> myContex  (new myContexOpenCl());
/*

cl_context getContextCL(){
	cl_int errorcode;
	if (theContextOpenCL.cpPlatform==NULL){
		if (clGetPlatformIDs(1, &theContextOpenCL.cpPlatform, NULL) != CL_SUCCESS) {
			printf("clGetPlatformIDs error\n");
		}
	
	}
	
	if(theContextOpenCL.cdDevice==NULL){
		if (clGetDeviceIDs(theContextOpenCL.cpPlatform, CL_DEVICE_TYPE_GPU, 1, 
						   &theContextOpenCL.cdDevice, NULL) != CL_SUCCESS) {
			printf("clGetDeviceIDs error\n");
		}
	}
	
	if(theContextOpenCL.GPUContext==NULL){
		theContextOpenCL.GPUContext = clCreateContext(0, 1, &theContextOpenCL.cdDevice,
											NULL, NULL, &errorcode); 
		if (theContextOpenCL.GPUContext == NULL) {
			printf("clCreateContextFromType error: ");
		if (errorcode == CL_INVALID_PLATFORM) 
			printf("invalid platform\n");
		if (errorcode == CL_INVALID_VALUE) printf("invalid value\n");
		if (errorcode == CL_DEVICE_NOT_AVAILABLE) 
			printf("device not available\n");
		if (errorcode == CL_DEVICE_NOT_FOUND)
			printf("device not found\n");
		if (errorcode == CL_OUT_OF_HOST_MEMORY)
			printf("out of host memory\n");
		if (errorcode == CL_INVALID_DEVICE_TYPE) 
			printf("invalid device type\n");
		exit(1);
		}
	}
	
	if(theContextOpenCL.cqCommandQueue==NULL){
		theContextOpenCL.cqCommandQueue = 
		clCreateCommandQueue(theContextOpenCL.GPUContext, theContextOpenCL.cdDevice, 0, NULL);
		if (theContextOpenCL.cqCommandQueue == NULL) {
			printf("clCreateCommandQueue error\n");
		}
	}
	
	return theContextOpenCL.GPUContext;

}


cl_device_id getDeviceCL(){
	
	if(theContextOpenCL.cdDevice==NULL){
		if (clGetDeviceIDs(theContextOpenCL.cpPlatform, CL_DEVICE_TYPE_GPU, 1, 
						   &theContextOpenCL.cdDevice, NULL) != CL_SUCCESS) {
			printf("clGetDeviceIDs error\n");
		}
	}
	
	return theContextOpenCL.cdDevice;

}
*/



/*static char * load_program_source(const char *filename) {
	
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
}*/


