#include <vector>
#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/ocl/ocl.hpp"

static char* exampleImageFile = "kewell1.jpg";

using namespace cv;

int main(int argc, char* argv[])
{	
	std::vector<ocl::Info> oclinfo;
	ocl::getDevice(oclinfo, ocl::CVCL_DEVICE_TYPE_GPU);

	std::cout << "OpenCL devices:" << std::endl;

	for (std::vector<ocl::Info>::iterator it = oclinfo.begin();
		it != oclinfo.end();
		it++)
	{
		for (int i = 0; i < (*it).DeviceName.size(); i++)
		{
			std::cout << (*it).DeviceName[i] << std::endl;
		}
	}

	//setting the device 1
	ocl::setDevice(oclinfo[0], 0);

	Mat image = imread(exampleImageFile);
	//cvtColor(image, image, CV_BGR2GRAY);
	
	ocl::oclMat ocl_image(image);
	ocl::oclMat ocl_sobel_filtered;

	ocl_sobel_filtered.create(ocl_image.size(), CV_MAKETYPE(ocl_image.depth(), ocl_image.channels()));
	ocl::Sobel(ocl_image, ocl_sobel_filtered, CV_8U, 1, 1, 3, 1.0, 0.0, BORDER_CONSTANT);
	cv::Mat sobel_filtered(ocl_sobel_filtered);

	imshow("test", sobel_filtered);
	waitKey();

	
	return 0;
}