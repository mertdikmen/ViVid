#include <fstream>
#include "opencv2\opencv.hpp"
#include "vivid.hpp"

#include "classifier.hpp"
#include "filterbank.hpp"

static char* exampleImagePath = "..\\..\\..\\media\\kewell1.jpg";

int main(int argc, char* argv[])
{
	cv::Mat exampleImage = cv::imread(exampleImagePath, 0);

	const int num_filters = 100;
	const int filter_dim = 3;

	FilterBank fb(filter_dim, num_filters);
	fb.set_on_device();


	return 0;
}