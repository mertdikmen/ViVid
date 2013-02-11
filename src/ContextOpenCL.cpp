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

namespace vivid
{
	myContexOpenCl *  TheContext::The_Context_GPU=NULL;
	myContexOpenCl *  TheContext::The_Context_CPU=NULL;

	TheContext::TheContext()
	{
		if (The_Context_GPU==NULL){
			The_Context_GPU = new myContexOpenCl( CL_DEVICE_TYPE_GPU );
			printf("\nGPU context done\n\n");
		}

		if (The_Context_CPU==NULL){
			The_Context_CPU = new myContexOpenCl( CL_DEVICE_TYPE_CPU );
			printf("\nCPU context done\n\n");
		}
	}

	TheContext::TheContext(int cpu)
	{
		if (The_Context_GPU==NULL && cpu==0){
			The_Context_GPU = new myContexOpenCl( CL_DEVICE_TYPE_GPU );
			printf("\nGPU context done\n\n");
		}

		if (The_Context_CPU==NULL && cpu==1){
			The_Context_CPU= new myContexOpenCl( CL_DEVICE_TYPE_CPU );
			printf("\nCPU context done\n\n");
		}
	}

	myContexOpenCl * TheContext::getMyContext()
	{
		return The_Context_GPU;
	}

	myContexOpenCl * TheContext::getMyContextCPU()
	{
		return  The_Context_CPU;
	}


}