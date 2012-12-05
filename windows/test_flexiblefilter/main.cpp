#include "opencv2\opencv.hpp"
#include "vivid.hpp"

#include <iostream>
#include <fstream>

#include "omp.h"

static char* exampleImagePath = "..\\..\\..\\media\\kewell1.jpg";

void dump_output(char* file_name, float* data, const int channels, const int height, const int width)
{
	std::ofstream test_out(file_name, std::ios_base::out);

	int index = 0;
	for (int k = 0; k < channels; k++){
		for (int j = 0; j < height; j++)
		{
			for (int i = 0; i < width; i++)
			{
				test_out << data[index++] << ", ";
			}

			test_out << std::endl;
		}
		test_out << "----------------------" << std::endl;
	}
	test_out.close();
}

#define DO_C
#define DO_CUDA
#define DO_OPENCL
#define DO_OPENCV

int main(int argc, char* argv[])
{
	cv::Mat exampleImage = cv::imread(exampleImagePath, 0);

	//convert to float
	exampleImage.convertTo(exampleImage, CV_32FC1);

	//pull the data
	float* f_imData = (float*) exampleImage.data;

	const int height = exampleImage.size().height;
	const int width = exampleImage.size().width;

	//create a random filterbank
	const int num_filters = 100;
	const int filter_dim = 3;

	float* filter_bank = new float[num_filters * filter_dim * filter_dim];

	for (int i = 0; i < num_filters * filter_dim * filter_dim; i++)
	{
		filter_bank[i] = float( std::rand() ) / RAND_MAX;
	}

	double tic, toc;

#ifdef DO_OPENCV
	cv::Mat out_temp(height, width, CV_32FC1);
	cv::Mat out_index(height, width, CV_32FC1);
	cv::Mat out_value;

	tic = omp_get_wtime();
	for (int i = 0; i < num_filters; i++)
	{
		cv::Mat filter(filter_dim, filter_dim, CV_32FC1, filter_bank + i * filter_dim * filter_dim);

		cv::matchTemplate(exampleImage, filter, out_temp, CV_TM_CCORR);

		out_temp = cv::abs(out_temp);

		if (i==0)
		{
			out_value = cv::abs(out_temp);
		}
		else 
		{
			out_value = cv::max(out_value, out_temp);
		}

	}
	toc = omp_get_wtime();

	std::cout << "OpenCV time: " << toc - tic << std::endl;

	//dump_output("test_ff_opencv.out", (float*)out_value.data, 1, height, width);
#endif

#ifdef DO_C
	//C Reference
	float* retvalC = new float[2 * height * width];
	
	tic = omp_get_wtime();
	cosine_filter(f_imData, filter_bank, height, width, filter_dim, filter_dim, num_filters, retvalC);
	toc = omp_get_wtime();

	std::cout << "C time: " << toc - tic << std::endl;

	dump_output("test_ff_c.out", retvalC, 2, height, width);
	delete[] retvalC;
#endif

#ifdef DO_CUDA
	//CUDA Reference
	DeviceMatrix::Ptr dmpCU = makeDeviceMatrix(height, width);
	
	tic = omp_get_wtime();
	
	DeviceMatrix_copyToDevice(*dmpCU, f_imData);
	set_filter_bank_cuda(filter_bank, num_filters * filter_dim * filter_dim);
	DeviceMatrix3D::Ptr retdmCU = filter_frame_cuda_3(dmpCU, num_filters, 1, FF_OPTYPE_COSINE);
	float* retvalCU = new float[height * width * 2];
	DeviceMatrix3D_copyFromDevice(*retdmCU, retvalCU);

	toc = omp_get_wtime();

	std::cout << "CUDA time: " << toc - tic << std::endl;

	dump_output("test_ff_cuda.out", retvalCU, 2, height, width);
	delete[] retvalCU;
#endif

#ifdef DO_OPENCL
	//OPENCL Reference
	DeviceMatrixCL::Ptr dmpCL = makeDeviceMatrixCL(height, width);

	tic = omp_get_wtime();

	DeviceMatrixCL_copyToDevice(*dmpCL, f_imData);
	set_filter_bank_cl(filter_bank, num_filters * filter_dim * filter_dim);
	DeviceMatrixCL3D::Ptr retdm = filter_frame_cl_3(dmpCL, num_filters, 1, FF_OPTYPE_COSINE);
	float* retvalCL = new float[height * width * 2];
	DeviceMatrixCL3D_copyFromDevice(*retdm, retvalCL);

	toc = omp_get_wtime();

	std::cout << "OpenCL time: " << toc - tic << std::endl;

	dump_output("test_ff_opencl.out", retvalCL, 2, height, width);

	delete[] retvalCL;
#endif

	delete[] filter_bank;

	return 0;
}