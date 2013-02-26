#include <stdio.h>
#include "cl_exceptions.hpp"

void vivid::print_cl_error(cl_int errorcode)
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