#include "vivid.hpp"

int main(int argc, char* argv[])
{
	float* random1 = new float[1000 * 5];
	float* random2 = new float[1000 * 3];

	//create a device matrix
	DeviceMatrixCL::Ptr dmpCL1 = makeDeviceMatrixCL(5, 1000);
	DeviceMatrixCL::Ptr dmpCL2 = makeDeviceMatrixCL(3, 1000);

	//copy to the DeviceMatrix
	DeviceMatrixCL_copyToDevice(*dmpCL1, random1);
	DeviceMatrixCL_copyToDevice(*dmpCL2, random2);

	DeviceMatrixCL::Ptr dmpCLpwdist = pwdist_cl(dmpCL1, dmpCL2);

	float* retval = new float[5*3];

	//copy to the DeviceMatrix
	DeviceMatrixCL_copyFromDevice(*dmpCLpwdist, retval);

	return 0;
}