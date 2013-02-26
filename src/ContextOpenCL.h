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
#include "cl_exceptions.hpp"

using namespace std;

namespace vivid
{
	enum DeviceType { DEVICE_GPU = CL_DEVICE_TYPE_GPU, DEVICE_CPU = CL_DEVICE_TYPE_CPU };

	class ContexOpenCl 
	{
	public:
		ContexOpenCl(cl_device_id _device_id, cl_platform_id _platform_id);

		cl_device_id getDeviceCL(){ return deviceId; }
		cl_context getContextCL(){ return context; }
		cl_platform_id getPlatform(){ return platformId; }
		cl_command_queue getCommandQueue() { return commandQueue; }

	private:
		cl_context context;
		cl_command_queue commandQueue;
		cl_platform_id platformId;
		cl_device_id deviceId;
	};

	class CLContextSource
	{
	public:
		CLContextSource(std::string cpu_platform = "", std::string gpu_platform = "");

		~CLContextSource(){};

		ContexOpenCl* getContext(DeviceType target_device)
		{
			switch (target_device)
			{
			case DEVICE_CPU:
				return The_Context_CPU;
			case DEVICE_GPU:
				return The_Context_GPU;
			default:
				return NULL;
			}
		}

	private:
		static ContexOpenCl* The_Context_GPU;
		static ContexOpenCl* The_Context_CPU;
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


