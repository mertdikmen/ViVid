#include <stdio.h>
#include "cl_exceptions.hpp"

void vivid::print_cl_error(cl_int errorcode, char* file, int line)
{
	if (line <= 0)
	{
		printf("OpenCL Error in file: %s at line: %d. ", file, line);
	}
	else
	{
		printf("OpenCL Error: ");
	}

	switch (errorcode)
	{
	case CL_SUCCESS:                            printf("Success!");break;
	case CL_DEVICE_NOT_FOUND:                   printf("Device not found.");break;
	case CL_DEVICE_NOT_AVAILABLE:               printf("Device not available");break;
	case CL_COMPILER_NOT_AVAILABLE:             printf("Compiler not available");break;
	case CL_MEM_OBJECT_ALLOCATION_FAILURE:      printf("Memory object allocation failure");break;
	case CL_OUT_OF_RESOURCES:                   printf("Out of resources");break;
	case CL_OUT_OF_HOST_MEMORY:                 printf("Out of host memory");break;
	case CL_PROFILING_INFO_NOT_AVAILABLE:       printf("Profiling information not available");break;
	case CL_MEM_COPY_OVERLAP:                   printf("Memory copy overlap");break;
	case CL_IMAGE_FORMAT_MISMATCH:              printf("Image format mismatch");break;
	case CL_IMAGE_FORMAT_NOT_SUPPORTED:         printf("Image format not supported");break;
	case CL_BUILD_PROGRAM_FAILURE:              printf("Program build failure");break;
	case CL_MAP_FAILURE:                        printf("Map failure");break;
	case CL_INVALID_VALUE:                      printf("Invalid value");break;
	case CL_INVALID_DEVICE_TYPE:                printf("Invalid device type");break;
	case CL_INVALID_PLATFORM:                   printf("Invalid platform");break;
	case CL_INVALID_DEVICE:                     printf("Invalid device");break;
	case CL_INVALID_CONTEXT:                    printf("Invalid context");break;
	case CL_INVALID_QUEUE_PROPERTIES:           printf("Invalid queue properties");break;
	case CL_INVALID_COMMAND_QUEUE:              printf("Invalid command queue");break;
	case CL_INVALID_HOST_PTR:                   printf("Invalid host pointer");break;
	case CL_INVALID_MEM_OBJECT:                 printf("Invalid memory object");break;
	case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:    printf("Invalid image format descriptor");break;
	case CL_INVALID_IMAGE_SIZE:                 printf("Invalid image size");break;
	case CL_INVALID_SAMPLER:                    printf("Invalid sampler");break;
	case CL_INVALID_BINARY:                     printf("Invalid binary");break;
	case CL_INVALID_BUILD_OPTIONS:              printf("Invalid build options");break;
	case CL_INVALID_PROGRAM:                    printf("Invalid program");break;
	case CL_INVALID_PROGRAM_EXECUTABLE:         printf("Invalid program executable");break;
	case CL_INVALID_KERNEL_NAME:                printf("Invalid kernel name");break;
	case CL_INVALID_KERNEL_DEFINITION:          printf("Invalid kernel definition");break;
	case CL_INVALID_KERNEL:                     printf("Invalid kernel");break;
	case CL_INVALID_ARG_INDEX:                  printf("Invalid argument index");break;
	case CL_INVALID_ARG_VALUE:                  printf("Invalid argument value");break;
	case CL_INVALID_ARG_SIZE:                   printf("Invalid argument size");break;
	case CL_INVALID_KERNEL_ARGS:                printf("Invalid kernel arguments");break;
	case CL_INVALID_WORK_DIMENSION:             printf("Invalid work dimension");break;
	case CL_INVALID_WORK_GROUP_SIZE:            printf("Invalid work group size");break;
	case CL_INVALID_WORK_ITEM_SIZE:             printf("Invalid work item size");break;
	case CL_INVALID_GLOBAL_OFFSET:              printf("Invalid global offset");break;
	case CL_INVALID_EVENT_WAIT_LIST:            printf("Invalid event wait list");break;
	case CL_INVALID_EVENT:                      printf("Invalid event");break;
	case CL_INVALID_OPERATION:                  printf("Invalid operation");break;
	case CL_INVALID_GL_OBJECT:                  printf("Invalid OpenGL object");break;
	case CL_INVALID_BUFFER_SIZE:                printf("Invalid buffer size");break;
	case CL_INVALID_MIP_LEVEL:                  printf("Invalid mip-map level");break;
	default: printf("Unknown: %d", errorcode);
	}
}