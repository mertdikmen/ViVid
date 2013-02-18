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
	ContexOpenCl* TheContext::The_Context_GPU=NULL;
	ContexOpenCl* TheContext::The_Context_CPU=NULL;

	TheContext::TheContext()
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
			printf("Getting GPU Device\n");
			for (int i = 0; i < n_platforms; i++)
			{
				if (clGetDeviceIDs(cpPlatforms[i], CL_DEVICE_TYPE_GPU, 1, &cdDevice, NULL) == CL_SUCCESS) 
				{
					std::cout << "Found GPU device on OpelCL platform " << i + 1 << std::endl;
					The_Context_GPU = new ContexOpenCl(cdDevice);
					break;
				}
				else if (i == n_platforms - 1)
				{
					std::cerr << "Cannot find GPU device" << std::endl;
				}
			}
		}

		if (The_Context_CPU==NULL){
			printf("Getting CPU device\n");
			for (int i = 0; i < n_platforms; i++)
			{
				if (clGetDeviceIDs(cpPlatforms[i], CL_DEVICE_TYPE_CPU, 1, &cdDevice, NULL) == CL_SUCCESS) 
				{
					std::cout << "Found CPU device on OpelCL Platform " << i + 1 << std::endl;
					The_Context_CPU = new ContexOpenCl(cdDevice);
					break;
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