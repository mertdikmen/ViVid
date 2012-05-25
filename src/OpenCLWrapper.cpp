#include <iostream>

#include "OpenCLWrapper.hpp"

using namespace boost::python;

void test_cl()
{
    std::cout << "test valid" << std::endl;
}

void export_OpenCL()
{
    /*
    cl_int error = 0;   // Used to handle error codes
    cl_platform_id platform;
    cl_context context;
    cl_command_queue queue;
    cl_device_id device;

    // Platform
    error = clGetPlatformIDs(&platform);
    assert(error == CL_SUCCESS);

    // Device
    error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    assert(error == CL_SUCCESS);
        
    // Context
    context = clCreateContext(0, 1, &device, NULL, NULL, &error);
    assert(error == CL_SUCCESS);
    
    // Command-queue
    queue = clCreateCommandQueue(context, device, 0, &error);
    assert(error == CL_SUCCESS);
    */
}


