/*
 *  ContextOpenCl.cpp
 *  
 *
 *  Created by Antonio García Martín on 30/06/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "OpenCLKernels.hpp"

theKernels *  MyKernels::My_Kernels=NULL;

MyKernels::MyKernels(cl_context GPUContext1,
        cl_device_id cdDevice1){
    //	printf("Te Context()");
    if (My_Kernels==NULL){
        //	printf("Is NULL");
        My_Kernels = new theKernels(GPUContext1,cdDevice1);
    }
}



