#include "vivid.hpp"
#include <iostream>

int main(int argc, char* argv[])
{
	float* random1 = new float[1000 * 5];
	float* random2 = new float[1000 * 3];

	for (int i = 0; i < 1000 * 5; i++){
		random1[i] = float(std::rand()) / RAND_MAX;
	}

	for (int i = 0; i < 1000 * 3; i++){
		random2[i] = float(std::rand()) / RAND_MAX;
	}

	//create a device matrix
	DeviceMatrixCL::Ptr dmpCL1 = makeDeviceMatrixCL(5, 1000);
	DeviceMatrixCL::Ptr dmpCL2 = makeDeviceMatrixCL(3, 1000);

	//copy to the DeviceMatrix
	DeviceMatrixCL_copyToDevice(*dmpCL1, random1);
	DeviceMatrixCL_copyToDevice(*dmpCL2, random2);

	DeviceMatrixCL::Ptr dmpCLpwdist = pwdist_cl(dmpCL1, dmpCL2);

	float* retval_ref = new float[5*3];
	memset(retval_ref, 0, sizeof(float) * 5 * 3);

	for (int i = 0; i < 5; i++){
		for (int j = 0; j < 3; j++){
			for (int k = 0; k < 1000; k++){
				float diff = random1[i * 1000 + k] - random2[j * 1000 + k];
				retval_ref[i * 3 + j] += diff * diff;
			}
		}
	}

	float* retval = new float[5*3];

	//copy to the DeviceMatrix
	DeviceMatrixCL_copyFromDevice(*dmpCLpwdist, retval);

	

	for (int i = 0; i < 5 * 3; i++)
	{
		std::cout << retval_ref[i] << ":" << retval[i] << std::endl;
	}


	return 0;
}