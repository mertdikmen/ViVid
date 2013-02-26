/*
*  ContextOpenCl.cpp
*  
*
*  Created by Antonio García Martín on 30/06/12.
*  Copyright 2012 __MyCompanyName__. All rights reserved.
*
*/

#include "ContextOpenCL.h"
//extern int device_use;

//Get an OpenCL platform
#define NUM_ENTRIES 10

namespace vivid
{
	void print_cl_error(cl_int errorcode)
	{
		printf("OpenCL Error: ");
		switch (errorcode)
		{
		case CL_INVALID_CONTEXT:
			printf("Invalid context\n");
			break;
		case CL_INVALID_PLATFORM:
			printf("Invalid platform\n");
			break;
		case CL_INVALID_VALUE: 
			printf("Invalid value\n");
			break;
		case CL_DEVICE_NOT_AVAILABLE:
			printf("Device not available\n");
			break;
		case CL_DEVICE_NOT_FOUND:
			printf("Device not found\n");
			break;
		case CL_OUT_OF_HOST_MEMORY:
			printf("Out of host memory\n");
			break;
		case CL_INVALID_DEVICE_TYPE:
			printf("Invalid device type\n");
			break;
		default:
			printf("OpenCL Error: Unknown error code\n");
			break;
		}
	}

	PlatformDevicePair cpu_platform_device_info;
	PlatformDevicePair gpu_platform_device_info;

	ContexOpenCl* CLContextSource::The_Context_GPU=NULL;
	ContexOpenCl* CLContextSource::The_Context_CPU=NULL;

	PlatformDevicePair setup_cl_platform(std::string platform_vendor, DeviceType device_type)
	{
		cl_platform_id cpPlatforms[NUM_ENTRIES]; 
		cl_device_id cdDevice; 
		cl_uint n_platforms;

		OPENCL_CALL(clGetPlatformIDs(NUM_ENTRIES, cpPlatforms, &n_platforms));

		char platform_vendors[NUM_ENTRIES][256];
		
		size_t param_value_size_ret;
		printf("OpenCL Platforms: %d\n", n_platforms);
		for (int i = 0; i < n_platforms; i++)
		{
			OPENCL_CALL(clGetPlatformInfo(cpPlatforms[i],CL_PLATFORM_VENDOR,256,platform_vendors[i],&param_value_size_ret));
			std::cout << "Platform " << i + 1 << ": " << platform_vendors[i] << std::endl;
		}

		for (int i = 0; i < n_platforms; i++)
		{
			if ((clGetDeviceIDs(cpPlatforms[i], device_type, 1, &cdDevice, NULL) == CL_SUCCESS) &&
				(strcmp(platform_vendor.c_str(), platform_vendors[i]) == 0))
			{
				return PlatformDevicePair(platform_vendors[i], cdDevice);
			}
		}

		return PlatformDevicePair(CL_INVALID_PLATFORM, CL_INVALID_DEVICE)
	}

	CLContextSource::CLContextSource(std::string cpu_platform, std::string gpu_platform)
	{
		if ((The_Context_GPU!=NULL) && (The_Context_CPU !=NULL))
		{ 
			//all contexts exist, bail.
			return;
		}

		cpu_platform_device_info = setup_cl_platform(cpu_platform, DEVICE_CPU);
		gpu_platform_device_info = setup_cl_platform(gpu_platform, DEVICE_GPU);

		//if (The_Context_GPU==NULL){
		//	printf("\nGetting GPU Device\n");
		//			std::cout << "Found GPU device on OpelCL platform " << i + 1 << std::endl;
		//			if ( (strcmp(gpu_platform.c_str(), "") == 0) ||
		//				 () )
		//			{
		//				std::cout << "Creating context." << std::endl;
		//				The_Context_GPU = new ContexOpenCl(cdDevice);
		//				break;
		//			}
		//			else 
		//			{
		//				std::cout << "Does not match config: " << gpu_platform.c_str() << ". Skipping." << std::endl;
		//			}
		//		}
		//	}
		//	if (The_Context_GPU == NULL)
		//	{
		//		printf("No GPU Context created\n");
		//	}
		//}

		//if (The_Context_CPU==NULL){
		//	printf("\nGetting CPU device\n");
		//	for (int i = 0; i < n_platforms; i++)
		//	{
		//		if (clGetDeviceIDs(cpPlatforms[i], CL_DEVICE_TYPE_CPU, 1, &cdDevice, NULL) == CL_SUCCESS) 
		//		{
		//			std::cout << "Found CPU device on OpelCL Platform " << i + 1 << std::endl;
		//			if ( (strcmp(cpu_platform.c_str(), "") == 0) ||
		//				 (strcmp(cpu_platform.c_str(), platform_vendors[i]) == 0) )
		//			{
		//				std::cout << "Creating context." << std::endl;
		//				The_Context_CPU = new ContexOpenCl(cdDevice);
		//				break;
		//			}
		//			else 
		//			{
		//				std::cout << "Does not match config: " << cpu_platform.c_str() << ". Skipping." << std::endl;
		//			}
		//		}
		//		else if (i == n_platforms - 1)
		//		{
		//			std::cerr << "Cannot find CPU device" << std::endl;
		//		}
		//	}
		//}
	}

	//TheContext::TheContext(int cpu)
	//{
	//	if (The_Context_GPU==NULL && cpu==0){
	//		The_Context_GPU = new myContexOpenCl( CL_DEVICE_TYPE_GPU );
	//		printf("\nGPU context done\n\n");
	//	}

	//	if (The_Context_CPU==NULL && cpu==1){
	//		The_Context_CPU= new myContexOpenCl( CL_DEVICE_TYPE_CPU );
	//		printf("\nCPU context done\n\n");
	//	}
	//}

	//myContexOpenCl * TheContext::getMyContextCPU()
	//{
	//	return  The_Context_CPU;
	//}


}