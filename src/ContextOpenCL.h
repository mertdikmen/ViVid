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

using namespace std;

#include <CL/opencl.h>
#include <CL/cl.h>
#include <boost/shared_ptr.hpp>


//Get an OpenCL platform
struct myContexOpenCl {
	
	cl_platform_id cpPlatform; 
	cl_device_id cdDevice; 
	cl_command_queue cqCommandQueue;
	cl_context GPUContext;
	myContexOpenCl(){
		
		cl_int errorcode;
	//	if (cpPlatform==NULL){
			printf("Creating platform\n");
			if (clGetPlatformIDs(1, &cpPlatform, NULL) != CL_SUCCESS) {
				printf("clGetPlatformIDs error\n");
			}
			
	//	}
		
	//	if(cdDevice==NULL){
			printf("Getting Device\n");
			if (clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, 
							   &cdDevice, NULL) != CL_SUCCESS) {
				printf("clGetDeviceIDs con el device error\n");
			}
			
			
	//	}
	
		
		
	//	if(GPUContext==NULL){
			printf("Making context\n");
			GPUContext = clCreateContext(0, 1, &cdDevice,
														  NULL, NULL, &errorcode); 
			if (GPUContext == NULL) {
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
	//	}
		
	//	if(cqCommandQueue==NULL){
			printf("Making Command Queue\n");
			cqCommandQueue = 
			clCreateCommandQueue(GPUContext, cdDevice, 0, NULL);
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

		if(GPUContext==NULL){
			GPUContext = clCreateContext(0, 1, &cdDevice,
										 NULL, NULL, &errorcode); 
			if (GPUContext == NULL) {
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
		return GPUContext;
		
	}
	
}  ;




class TheContext{
	
 	
public:
	static myContexOpenCl *  The_Context_Singleton;

	TheContext();
	myContexOpenCl * getMyContext();
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


