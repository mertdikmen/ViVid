#include "vivid.hpp"
#include <opencv2\opencv.hpp>

const char* exampleImagePath = "..\\..\\..\\media\\kewell1.jpg";

int main(int argc, char* argv[])
{
	cv::Mat exampleImage = cv::imread(exampleImagePath, 0);
	
	//convert to float
	exampleImage.convertTo(exampleImage, CV_32FC1);

	//pull the data
	float* f_imData = (float*) exampleImage.data;

	const int height = exampleImage.size().height;
	const int width = exampleImage.size().width;

	//create a device matrix
	DeviceMatrixCL::Ptr dmpCL = makeDeviceMatrixCL(height, width);

	//copy to the DeviceMatrix
	DeviceMatrixCL_copyToDevice(*dmpCL, f_imData);

	//copy back
	float* copiedBack = new float[height * width];
	DeviceMatrixCL_copyFromDevice(*dmpCL, copiedBack);

	//verify
	for (int i = 0; i < exampleImage.size().area(); i++)
	{
		assert(copiedBack[i] == f_imData[i]);
	}
	
	return 0;
}