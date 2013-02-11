#include "opencv2\opencv.hpp"
#include "vivid.hpp"
#include "omp.h"
#include <windows.h>

#include "filterbank.hpp"
#include "classifier.hpp"

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
	

	static char* exampleImagePath = "..\\..\\..\\media\\kewell1.jpg";

	//create a random filterbank
	const int num_filters = 100;
	const int filter_dim = 3;

	FilterBank fb(filter_dim, num_filters);
	fb.set_on_device();

	Classifier clf(128, 64, 8, 2, num_filters);

	//load the image on device
	cv::Mat exampleImage = cv::imread(exampleImagePath, 0);
	//convert to float
	exampleImage.convertTo(exampleImage, CV_32FC1);

	if(device_use==0)
		std::cout << "running on CPU" <<std::endl;
	else
		std::cout << "running on GPU" <<std::endl;
	std::cout << "Image dimensions:" << exampleImage.size().height <<" "<< exampleImage.size().width <<std::endl;

	//pull the data
	float* f_imData = (float*) exampleImage.data;
	DeviceMatrixCL::Ptr dmpCL = makeDeviceMatrixCL(exampleImage.size().height, exampleImage.size().width);
	DeviceMatrixCL_copyToDevice(*dmpCL, f_imData);


/*	for(int i=0; i<20; i++)
	{

	DeviceMatrixCL3D::Ptr ff_im = fb.apply_cl(dmpCL);
//	tic1= omp_get_wtime();

	DeviceMatrixCL::Ptr block_histogram = cell_histogram_dense_cl(
		ff_im, num_filters, 8, 0, 0, 
		exampleImage.size().height, exampleImage.size().width);
//	tic2= omp_get_wtime();

	DeviceMatrixCL::Ptr result = clf.apply(block_histogram);
	}
	*/
	double tic0, tic1, tic2, tic3;
	tic0= omp_get_wtime();

//	for(int i=0; i<1000; i++)
	{

	DeviceMatrixCL3D::Ptr ff_im = fb.apply_cl(dmpCL);
	tic1= omp_get_wtime();

	DeviceMatrixCL::Ptr block_histogram = cell_histogram_dense_cl(
		ff_im, num_filters, 8, 0, 0, 
		exampleImage.size().height, exampleImage.size().width);
	tic2= omp_get_wtime();

	DeviceMatrixCL::Ptr result = clf.apply(block_histogram);
	}

	tic3 = omp_get_wtime();

	std::cout << "full pipeline time: " << tic3 - tic0 << std::endl;
	std::cout << "filter pipeline time: " << tic1 - tic0 << std::endl;
	std::cout << "histogram pipeline time: " << tic2 - tic1 << std::endl;
	std::cout << "classifier pipeline time: " << tic3 - tic2 << std::endl;

	return 0;
}

/*
windows timer:
LARGE_INTEGER timerFreq_;
QueryPerformanceFrequency(&timerFreq_);
LARGE_INTEGER  st;
QueryPerformanceCounter(&st);



LARGE_INTEGER  et;
QueryPerformanceCounter(&et);
std::cout << (et.QuadPart - st.QuadPart) * 1000 / timerFreq_.QuadPart << "ms" << "\n";
*/