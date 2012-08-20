/*
 *  ContextOpenCl.cpp
 *  
 *
 *  Created by Antonio García Martín on 30/06/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "ContextOpenCL.h"

myContexOpenCl *  TheContext::The_Context_GPU=NULL;
myContexOpenCl *  TheContext::The_Context_CPU=NULL;



TheContext::TheContext()
{
	if (The_Context_GPU==NULL){
	//	printf("Is NULL");
		The_Context_GPU = new myContexOpenCl( CL_DEVICE_TYPE_GPU );
		printf("\nGPU context done\n\n");
	}
	
	if (The_Context_CPU==NULL){
		// the_Contexts[0]=The_Context_Singleton;
		 The_Context_CPU= new myContexOpenCl( CL_DEVICE_TYPE_CPU );
		 printf("\nCPU context done\n\n");

	}

	

}

myContexOpenCl * TheContext::getMyContext(){
	return The_Context_CPU;
}
myContexOpenCl * TheContext::getMyContextCPU(){
	return  The_Context_CPU;
}

