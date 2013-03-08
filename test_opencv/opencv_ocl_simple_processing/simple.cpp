#include <vector>
#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/ocl/ocl.hpp"

#include "omp.h"

static char* exampleImageFile = "kewell1.jpg";

using namespace cv;

#define TIMED_CALL(str, iterations, blob){\
	double tic = omp_get_wtime(); \
	for (int i = 0; i < iterations; i++){ \
	blob; \
	} \
	double toc = omp_get_wtime(); \
	printf("Time of %s: %.4f\n", str, toc - tic); }

Mat doSobel(Mat& input)
{
	Mat output;
	TIMED_CALL("Sobel CPP", 1000,
		Sobel(input, output, CV_8U, 1, 1, 3, 1.0, 0.0, BORDER_CONSTANT);
	);
	return output;
}

Mat doSobel_ocl(Mat& input)
{
	ocl::oclMat ocl_image(input);
	ocl::oclMat ocl_sobel_filtered;

	ocl_sobel_filtered.create(ocl_image.size(), CV_MAKETYPE(ocl_image.depth(), ocl_image.channels()));	
	TIMED_CALL("Sobel OpenCL", 1000,
		ocl::Sobel(ocl_image, ocl_sobel_filtered, CV_8U, 1, 1, 3, 1.0, 0.0, BORDER_CONSTANT);
	);
	cv::Mat sobel_filtered(ocl_sobel_filtered);

	return sobel_filtered;
}

void test_sobel(Mat& image)
{
	Mat sobel_ocl = doSobel_ocl(image);
	Mat sobel_c = doSobel(image);
	//imshow("Sobel OpenCL", sobel_ocl);
	//imshow("Sobel CPP", sobel_c);
	//waitKey();
}

Mat doCanny_ocl(Mat& input)
{
	ocl::oclMat ocl_image(input);
	ocl::oclMat ocl_canny_edges;

	ocl_canny_edges.create(ocl_image.size(), CV_8UC1);

	TIMED_CALL("Canny OpenCL", 100, 
			ocl::Canny(ocl_image, ocl_canny_edges, 0.1, 1.0, 3, false);
	);

	Mat canny_edges(ocl_canny_edges);

	return canny_edges;
}

void test_canny(Mat& image)
{
	Mat gray_image;
	cvtColor(image, gray_image, CV_BGR2GRAY);

	doCanny_ocl(gray_image);

}

Mat doGaussianBlur(Mat& input)
{
	Mat output;
	TIMED_CALL("Gauss Blur CPP", 1000,
			GaussianBlur(input, output, cv::Size(11,11), 5.0, 0.0, BORDER_REFLECT);
	);
	return output;
}

Mat doGaussianBlur_ocl(Mat& input)
{
	ocl::oclMat input_ocl(input);
	ocl::oclMat output_ocl;

	TIMED_CALL("Gauss Blur OpenCL", 1000,
		ocl::GaussianBlur(input_ocl, output_ocl, cv::Size(11,11), 5.0, 0.0, BORDER_REFLECT);
	);
	return Mat(output_ocl);
}

void test_gaussian_blur(Mat& image)
{
	Mat gauss_ocl =	doGaussianBlur_ocl(image);
	Mat gauss_cpp = doGaussianBlur(image);
}

Mat doHistogram_ocl(Mat& input)
{
	ocl::oclMat input_ocl(input);
	ocl::oclMat hist_ocl;

	TIMED_CALL("Histogram OpenCL", 5000, ocl::calcHist(input_ocl, hist_ocl); );
	return Mat(hist_ocl);
}

/*
OutputArray doHistogram(Mat& input)
{
	cv::OutputArray output;
	int chan = 1;
	int histSize = 256;
	const float ranges[2] = {0, 256};
	TIMED_CALL("Histogram CPP", 5000, calcHist(&input, 1, &chan, InputArray(), output, 1, &histSize, &ranges););
	return output;
}

void test_histogram(Mat& image)
{
	Mat gray_image;
	cvtColor(image, gray_image, CV_BGR2GRAY);
	Mat histogram_ocl = doHistogram_ocl(gray_image);
}
*/

Mat doHOG_ocl(Mat& image)
{
	ocl::HOGDescriptor  hog_desc_ocl = ocl::HOGDescriptor::HOGDescriptor();

	ocl::oclMat image_ocl(image);

	ocl::oclMat descriptors_ocl;

	TIMED_CALL("HOG OpenCL", 100,
	hog_desc_ocl.getDescriptors(image_ocl, Size(8,8), descriptors_ocl);
	);

	return Mat(descriptors_ocl);
}

std::vector<float> doHOG(Mat& image)
{
	HOGDescriptor hog_desc = HOGDescriptor::HOGDescriptor();

	std::vector<float> descriptors;

	TIMED_CALL("HOG CPP", 100,
			hog_desc.compute(image, descriptors, Size(8,8));
	);

	return descriptors;
}

void test_HOG(Mat& image)
{
	Mat gray_image;
	cvtColor(image, gray_image, CV_BGR2GRAY);
	doHOG_ocl(gray_image);
	doHOG(gray_image);
}

int main(int argc, char* argv[])
{	
	std::vector<ocl::Info> oclinfo;
	ocl::getDevice(oclinfo, ocl::CVCL_DEVICE_TYPE_CPU);
	
	//std::cout << "OpenCL devices:" << std::endl;

	for (std::vector<ocl::Info>::iterator it = oclinfo.begin();
		it != oclinfo.end();
		it++)
	{
		for (int i = 0; i < (*it).DeviceName.size(); i++)
		{
			std::cout << (*it).DeviceName[i] << std::endl;
		}

		break;
	}
	
	//setting the device 1
	ocl::setDevice(oclinfo[0], 0);

	Mat image = imread(exampleImageFile);
	//imshow("test",image);
	//waitKey();
	//test_sobel(image);
	//test_canny(image);
	//test_gaussian_blur(image);
	//test_histogram(image);
	test_HOG(image);
	return 0;
}