/*
 *  ContextOpenCl.cpp
 *  
 *
 *  Created by Antonio García Martín on 30/06/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "ContextOpenCL.h"



myContexOpenCl *  TheContext::The_Context_Singleton=NULL;

TheContext::TheContext(){
	
	printf("Te Context()");
	
	if (The_Context_Singleton==NULL){
		printf("Is NULL");
		The_Context_Singleton = new myContexOpenCl();
	}
	
	
}

myContexOpenCl * TheContext::getMyContext(){
	return The_Context_Singleton;
}