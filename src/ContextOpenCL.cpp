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
		switch (errorcode)
		{
		case -34:
			printf("OpenCL Error: Invalid Context\n");
			break;
		default:
			printf("OpenCL Error: Unknown error code\n");
			break;
		}
	}

	ContexOpenCl* TheContext::The_Context_GPU=NULL;
	ContexOpenCl* TheContext::The_Context_CPU=NULL;

	TheContext::TheContext(int target_device)
	{
		if ((target_device == VIVID_CL_CONTEXT_CPU) && (The_Context_CPU == NULL))
		{
			printf("Requested CPU context does not exist\n");
		}
		if ((target_device == VIVID_CL_CONTEXT_GPU) && (The_Context_GPU == NULL))
		{
			printf("Requested GPU context does not exist\n");
		}

	}

	TheContext::TheContext(std::string cpu_platform, std::string gpu_platform)
	{
		if ((The_Context_GPU!=NULL) && (The_Context_CPU !=NULL))
		{ 
			//all contexts exist, bail.
			return;
		}

		cl_platform_id cpPlatforms[NUM_ENTRIES]; 
		cl_device_id cdDevice; 

		OPENCL_CALL(clGetPlatformIDs(NUM_ENTRIES, cpPlatforms, &n_platforms));

		char platform_vendors[NUM_ENTRIES][256];
		size_t param_value_size_ret;
		printf("OpenCL Platforms: %d\n", n_platforms);
		for (int i = 0; i < n_platforms; i++)
		{
			OPENCL_CALL(clGetPlatformInfo(cpPlatforms[i],CL_PLATFORM_VENDOR,256,platform_vendors[i],&param_value_size_ret));
			std::cout << "Platform " << i + 1 << ": " << platform_vendors[i] << std::endl;
		}

		if (The_Context_GPU==NULL){
			printf("\nGetting GPU Device\n");
			for (int i = 0; i < n_platforms; i++)
			{
				if (clGetDeviceIDs(cpPlatforms[i], CL_DEVICE_TYPE_GPU, 1, &cdDevice, NULL) == CL_SUCCESS) 
				{
					std::cout << "Found GPU device on OpelCL platform " << i + 1 << std::endl;
					if ( (strcmp(gpu_platform.c_str(), "") == 0) ||
						 (strcmp(gpu_platform.c_str(), platform_vendors[i]) == 0) )
					{
						std::cout << "Creating context." << std::endl;
						The_Context_GPU = new ContexOpenCl(cdDevice);
						break;
					}
					else 
					{
						std::cout << "Does not match config: " << gpu_platform.c_str() << ". Skipping." << std::endl;
					}
				}
			}
			if (The_Context_GPU == NULL)
			{
				printf("No GPU Context created\n");
			}
		}

		if (The_Context_CPU==NULL){
			printf("\nGetting CPU device\n");
			for (int i = 0; i < n_platforms; i++)
			{
				if (clGetDeviceIDs(cpPlatforms[i], CL_DEVICE_TYPE_CPU, 1, &cdDevice, NULL) == CL_SUCCESS) 
				{
					std::cout << "Found CPU device on OpelCL Platform " << i + 1 << std::endl;
					if ( (strcmp(cpu_platform.c_str(), "") == 0) ||
						 (strcmp(cpu_platform.c_str(), platform_vendors[i]) == 0) )
					{
						std::cout << "Creating context." << std::endl;
						The_Context_CPU = new ContexOpenCl(cdDevice);
						break;
					}
					else 
					{
						std::cout << "Does not match config: " << cpu_platform.c_str() << ". Skipping." << std::endl;
					}
				}
				else if (i == n_platforms - 1)
				{
					std::cerr << "Cannot find CPU device" << std::endl;
				}
			}
		}
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