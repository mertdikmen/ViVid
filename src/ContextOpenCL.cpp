#include "ContextOpenCL.h"

namespace vivid
{
	ContexOpenCl* CLContextSource::The_Context_GPU=NULL;
	ContexOpenCl* CLContextSource::The_Context_CPU=NULL;

	ContexOpenCl::ContexOpenCl(cl_device_id _device_id, cl_platform_id _platform_id): 
		deviceId(_device_id), platformId(_platform_id)
	{
			char device_name[256];
			OPENCL_CALL(clGetDeviceInfo(deviceId, CL_DEVICE_NAME, 256, device_name, NULL));

			printf("Making the context on device %s\n", device_name);
			
			cl_int errorcode;
			
			context = clCreateContext(0, 1, &deviceId, NULL, NULL, &errorcode); 
			
			if (errorcode != CL_SUCCESS)
			{
				print_cl_error(errorcode);
				exit(1);
			}

			printf("Making Command Queue\n");
			commandQueue = clCreateCommandQueue(context, deviceId, 0, &errorcode);
			if (errorcode != CL_SUCCESS)
			{
				print_cl_error(errorcode);
			}
	}

	ContexOpenCl* setup_cl_platform(std::string platform_vendor, DeviceType device_type)
	{
		cl_platform_id cpPlatforms[10]; 
		cl_device_id cdDevice; 
		cl_uint n_platforms;

		OPENCL_CALL(clGetPlatformIDs(10, cpPlatforms, &n_platforms));

		char platform_vendors[10][256];
		
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
				return new ContexOpenCl(cdDevice, cpPlatforms[i]);
			}
		}

		return NULL;
	}

	CLContextSource::CLContextSource(std::string cpu_platform, std::string gpu_platform)
	{
		if ((The_Context_GPU!=NULL) && (The_Context_CPU !=NULL))
		{ 
			return;
		}

		if (The_Context_GPU == NULL)
		{
			The_Context_GPU = setup_cl_platform(gpu_platform, DEVICE_GPU);
		}
		if (The_Context_CPU == NULL)
		{
			The_Context_CPU = setup_cl_platform(cpu_platform, DEVICE_CPU);
		}
	}
}
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
	//}

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
//}