#include "vivid.hpp"
#include "omp.h"

/*	which device to use
	CPU 0
	GPU 1
*/
int device_use; 

int main(int argc, char* argv[])
{
	std::string cpu_platform = "Intel(R) Corporation";
	std::string gpu_platform = "Intel(R) Corporation";

	//Initialize the singleton source
	vivid::CLContextSource cl_context_source(cpu_platform, gpu_platform);

	device_use = 0;
	if(argc>1)
		device_use = atoi(argv[1]);
	
	const int aheight = 500;
	const int awidth = 1000;
	const int bheight = 150;
	const int bwidth = awidth;
	const int owidth = 300;

	float* random1 = new float[awidth * aheight];
	float* random2 = new float[awidth * bheight];
	for(int i=0; i<aheight*awidth; i++) {
		random1[i] = (rand() % 100) / (float)(100);
	}
	for(int i=0; i<bheight*bwidth; i++) {
		random2[i] = (rand() % 100) / (float)(100);
	}

	//create a device matrix
	DeviceMatrixCL::Ptr dmpCL1 = makeDeviceMatrixCL(aheight, awidth, vivid::DEVICE_CPU);
	DeviceMatrixCL::Ptr dmpCL2 = makeDeviceMatrixCL(bheight, bwidth, vivid::DEVICE_CPU);

	//copy to the DeviceMatrix
	DeviceMatrixCL_copyToDevice(*dmpCL1, random1);
	DeviceMatrixCL_copyToDevice(*dmpCL2, random2);
	double tic0,tic1;
	tic0 = omp_get_wtime();
	for (int i = 0; i < 1000; i++)
	{
		DeviceMatrixCL::Ptr dmpCLpwdist = pwdist_cl(dmpCL1, dmpCL2);
	}
	clFinish(cl_context_source.getContext(vivid::DEVICE_CPU)->getCommandQueue());
	tic1 = omp_get_wtime();
	std::cout << "pwdist time: " << tic1 - tic0 << std::endl;
	float* retval = new float[aheight*awidth];
	float* retval2 = new float[bheight*bwidth];

	//copy to the DeviceMatrix
	//DeviceMatrixCL_copyFromDevice(*dmpCLpwdist, retval);


	//double tic = omp_get_wtime();
 //   for (unsigned int i=0; i<aheight; i++){
 //       #pragma omp parallel for num_threads(4)
 //       for (unsigned int j=0; j<bheight; j++){
 //           float sum = 0.0;

 //           for (int k = 0; k < awidth; k++){
 //               float dif = (random1[i*awidth+k] - random2[j*awidth+k]);
 //               sum+=dif*dif;
 //           }
 //           retval2[i*owidth+j]=sum;
 //       }
 //   }
 //   double toc = omp_get_wtime();

 //   printf("CPU time: %.6f ms\n", (toc - tic) * 1e3 );
	//bool error = false;
	//for(int i=0; i<aheight*bheight; i++)
	//	if(abs(retval[i]-retval2[i])>1e-2) 
	//	{
	//		error = true;
	//		printf("%f %f\n",retval[i], retval2[i]);
	//	}
	//if(error)
	//	printf("output error!\n");

	return 0;
}