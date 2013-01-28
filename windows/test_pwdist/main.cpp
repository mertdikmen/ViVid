#include "vivid.hpp"
#include "omp.h"

/*	which device to use
	CPU 0
	GPU 1
*/
int device_use; 

int main(int argc, char* argv[])
{
	device_use = 0;
	if(argc>1)
		device_use = atoi(argv[1]);
	
	const int aheight = 500;
	const int awidth = 1000;
	const int bheight = 300;
	const int bwidth = 1000;
	const int owidth = 300;

	float* random1 = new float[1000 * 500];
	float* random2 = new float[1000 * 300];
	for(int i=0; i<aheight*awidth; i++) {
		random1[i] = (rand() % 100) / (float)(100);
	}
	for(int i=0; i<bheight*bwidth; i++) {
		random2[i] = (rand() % 100) / (float)(100);
	}

	//create a device matrix
	DeviceMatrixCL::Ptr dmpCL1 = makeDeviceMatrixCL(500, 1000);
	DeviceMatrixCL::Ptr dmpCL2 = makeDeviceMatrixCL(300, 1000);

	//copy to the DeviceMatrix
	DeviceMatrixCL_copyToDevice(*dmpCL1, random1);
	DeviceMatrixCL_copyToDevice(*dmpCL2, random2);
	double tic0,tic1;
	tic0 = omp_get_wtime();
	DeviceMatrixCL::Ptr dmpCLpwdist = pwdist_cl(dmpCL1, dmpCL2);
	tic1 = omp_get_wtime();
	std::cout << "pwdist time: " << tic1 - tic0 << std::endl;
	float* retval = new float[500*300];
	float* retval2 = new float[500*300];

	//copy to the DeviceMatrix
	DeviceMatrixCL_copyFromDevice(*dmpCLpwdist, retval);


	double tic = omp_get_wtime();
    for (unsigned int i=0; i<aheight; i++){
        #pragma omp parallel for num_threads(4)
        for (unsigned int j=0; j<bheight; j++){
            float sum = 0.0;

            for (int k = 0; k < awidth; k++){
                float dif = (random1[i*awidth+k] - random2[j*awidth+k]);
                sum+=dif*dif;
            }
            retval2[i*owidth+j]=sum;
        }
    }
    double toc = omp_get_wtime();

    printf("CPU time: %.6f ms\n", (toc - tic) * 1e3 );
	bool error = false;
	for(int i=0; i<aheight*bheight; i++)
		if(abs(retval[i]-retval2[i])>1e-2) 
		{
			error = true;
			printf("%f %f\n",retval[i], retval2[i]);
		}
	if(error)
		printf("output error!\n");

	return 0;
}