#include "opencv2\opencv.hpp"
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

	//create a random filterbank
	const int num_filters = 100;
	const int filter_dim = 3;

	float* filter_bank = new float[num_filters * filter_dim * filter_dim];

	for (int i = 0; i < num_filters * filter_dim * filter_dim; i++)
	{
		filter_bank[i] = float( std::rand() ) / RAND_MAX;
	}

	//CUDA Reference
	DeviceMatrix::Ptr dmpCU = makeDeviceMatrix(height, width);
	DeviceMatrix_copyToDevice(*dmpCU, f_imData);
	set_filter_bank_cuda(filter_bank, num_filters * filter_dim * filter_dim);
	DeviceMatrix3D::Ptr retdmCU = filter_frame_cuda_3(dmpCU, num_filters, 1, FF_OPTYPE_COSINE);
	float* retvalCU = new float[height * width * 2];
	DeviceMatrix3D_copyFromDevice(*retdmCU, retvalCU);

	//OPENCL Reference
	DeviceMatrixCL::Ptr dmpCL = makeDeviceMatrixCL(height, width);
	DeviceMatrixCL_copyToDevice(*dmpCL, f_imData);
	set_filter_bank_cl(filter_bank, num_filters * filter_dim * filter_dim);
	DeviceMatrixCL3D::Ptr retdm = filter_frame_cl_3(dmpCL, num_filters, 1, FF_OPTYPE_COSINE);
	float* retval = new float[height * width * 2];
	DeviceMatrixCL3D_copyFromDevice(*retdm, retval);

	for (int i = 0; i < 10; i++)
	{
		std::cout << retval[i] << "\t" << retvalCU[i] << std::endl;
	}

	delete[] filter_bank;
	delete[] retval;

	return 0;
}