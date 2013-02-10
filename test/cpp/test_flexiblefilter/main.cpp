#include "opencv2\opencv.hpp"
#include "vivid.hpp"

#include <iostream>
#include <fstream>

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

	//create a random filterbank
	const int num_filters = 100;
	const int filter_dim = 3;

	float* filter_bank = new float[num_filters * filter_dim * filter_dim];

	for (int i = 0; i < num_filters * filter_dim * filter_dim; i++)
	{
		filter_bank[i] = float( std::rand() ) / RAND_MAX;
	}

	//C Reference
	float* retvalC = new float[2 * height * width];
	cosine_filter(f_imData, filter_bank, height, width, filter_dim, filter_dim, num_filters, retvalC);

	std::ofstream test_out_c("testc.out", std::ios_base::out);
	for (int j = 0; j < height; j++)
	{
		for (int i = 0; i < width; i++)
		{
			test_out_c << retvalC[j * width + i] << ", ";
		}

		test_out_c << std::endl;
	}

	test_out_c << std::endl << std::endl << std::endl;

	for (int j = 0; j < height; j++)
	{
		for (int i = 0; i < width; i++)
		{
			test_out_c << retvalC[height * width + j * width + i] << ", ";
		}
		test_out_c << std::endl;
	}
	test_out_c.close();

	//CUDA Reference
/*	DeviceMatrix::Ptr dmpCU = makeDeviceMatrix(height, width);
	DeviceMatrix_copyToDevice(*dmpCU, f_imData);
	set_filter_bank_cuda(filter_bank, num_filters * filter_dim * filter_dim);
	DeviceMatrix3D::Ptr retdmCU = filter_frame_cuda_3(dmpCU, num_filters, 1, FF_OPTYPE_COSINE);
	float* retvalCU = new float[height * width * 2];
	DeviceMatrix3D_copyFromDevice(*retdmCU, retvalCU);

	std::ofstream test_out("test.out", std::ios_base::out);
	for (int j = 0; j < height; j++)
	{
		for (int i = 0; i < width; i++)
		{
			test_out << retvalCU[j * width + i] << ", ";
		}

		test_out << std::endl;
	}

	test_out << std::endl << std::endl << std::endl;

	for (int j = 0; j < height; j++)
	{
		for (int i = 0; i < width; i++)
		{
			test_out << retvalCU[height * width + j * width + i] << ", ";
		}
		test_out << std::endl;
	}
	test_out.close();
	*/
	//OPENCL Reference
	DeviceMatrixCL::Ptr dmpCL = makeDeviceMatrixCL(height, width);
	DeviceMatrixCL_copyToDevice(*dmpCL, f_imData);
	set_filter_bank_cl(filter_bank, num_filters * filter_dim * filter_dim);
	DeviceMatrixCL3D::Ptr retdm = filter_frame_cl_3(dmpCL, num_filters, 1, FF_OPTYPE_COSINE);
	float* retval = new float[height * width * 2];
	DeviceMatrixCL3D_copyFromDevice(*retdm, retval);

	std::ofstream test_out_cl("testcl.out", std::ios_base::out);
	for (int j = 0; j < height; j++)
	{
		for (int i = 0; i < width; i++)
		{
			test_out_cl << retval[j * width + i] << ", ";
		}

		test_out_cl << std::endl;
	}

	test_out_cl << std::endl << std::endl << std::endl;

	for (int j = 0; j < height; j++)
	{
		for (int i = 0; i < width; i++)
		{
			test_out_cl << retval[height * width + j * width + i] << ", ";
		}
		test_out_cl << std::endl;
	}
	test_out_cl.close();

	delete[] filter_bank;
	delete[] retval;
	delete[] retvalC;
	//delete[] retvalCU;

	return 0;
}