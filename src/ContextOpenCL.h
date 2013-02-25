#ifndef CONTEXTOPENCL_H
#define CONTEXTOPENCL_H

#pragma once

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
#include <CL/cl.h>

using namespace std;

#define VIVID_CL_CONTEXT_GPU 0
#define VIVID_CL_CONTEXT_CPU 1

#define OPENCL_CALL(call) do {\
	cl_int err = call; \
	if(CL_SUCCESS!= err) { \
	printf("clGetPlatformIDs error\n");	\
	} } while (0)

namespace vivid
{
	void print_cl_error(cl_int errorcode);

	class ContexOpenCl 
	{
	public:
		cl_context Context;
		cl_command_queue cqCommandQueue;
		cl_device_id cdDevice;

		ContexOpenCl(cl_device_id _cdDevice): cdDevice(_cdDevice)
		{
			char device_name[256];
			OPENCL_CALL(clGetDeviceInfo(cdDevice, CL_DEVICE_NAME, 256, device_name, NULL));

			printf("Making the context on device %s\n", device_name);
			
			cl_int errorcode;
			
			Context = clCreateContext(0, 1, &cdDevice, NULL, NULL, &errorcode); 
			
			if (Context == NULL) 
			{
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

			printf("Making Command Queue\n");
			cqCommandQueue = clCreateCommandQueue(Context, cdDevice, 0, &errorcode);
			if (errorcode != CL_SUCCESS)
			{
				printf("clCreateCommandQueue error\n");
			}
			if (cqCommandQueue == NULL) 
			{
				printf("clCreateCommandQueue error\n");
			}
		}

		cl_device_id getDeviceCL(){
			if(cdDevice==NULL){
				printf("Device == NULL\n");
				//if (clGetDeviceIDs(cpPlatforms[0], CL_DEVICE_TYPE_GPU, 1, 
				//	&cdDevice, NULL) != CL_SUCCESS) {
				//		printf("clGetDeviceIDs error\n");
				//}
			}
			return cdDevice;
		}

		cl_context getContextCL(){
			if (Context == NULL)
			{
				printf("Context == NULL\n");
			}
			/*
			cl_int errorcode;

			if(Context==NULL){
				Context = clCreateContext(0, 1, &cdDevice, NULL, NULL, &errorcode); 
				if (Context == NULL) 
				{
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
			*/
			return Context;
		}
	};

	class TheContext
	{
	public:
		static ContexOpenCl* The_Context_GPU;
		static ContexOpenCl* The_Context_CPU;

		TheContext(std::string cpu_platform = "", std::string gpu_platform = "");
		TheContext(int target_device);
		//TheContext(int cpu);
		ContexOpenCl* getMyContext(int target_device = VIVID_CL_CONTEXT_GPU)
		{
			if (target_device == VIVID_CL_CONTEXT_GPU)
				return The_Context_GPU;
			else if (target_device == VIVID_CL_CONTEXT_CPU)
				return The_Context_CPU;
		}
		//myContexOpenCl * getMyContextCPU();

		~TheContext(){};

		cl_uint n_devices;
		cl_uint n_platforms;

	};
}

#endif

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


