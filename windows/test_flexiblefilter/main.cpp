#include <opencv2\opencv.hpp>
#include "vivid.hpp"

static char* exampleImagePath = "..\\..\\..\\media\\kewell1.jpg";

int main(int argc, char* argv[])
{
	cv::Mat exampleImage = cv::imread(exampleImagePath, 0);
	
	//convert to float
	exampleImage.convertTo(exampleImage, CV_32FC1);

	//pull the data
	float* f_imData = (float*) exampleImage.data;

	const int height = exampleImage.size().height;
	const int width = exampleImage.size().width;

	return 0;
}