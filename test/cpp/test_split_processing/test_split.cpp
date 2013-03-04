#include <fstream>
#include "opencv2\opencv.hpp"
#include "vivid.hpp"

#include "boost/program_options.hpp"

#include "classifier.hpp"
#include "filterbank.hpp"

#include "omp.h"

namespace bpo = boost::program_options;

static char* exampleImagePath = "..\\..\\..\\media\\kewell1.jpg";

void parse_config(std::string config_filename, std::string& cpu_platform, std::string& gpu_platform)
{
	std::ifstream config_stream(config_filename.c_str(), ios_base::in);
	bpo::options_description desc("Options");
	desc.add_options()
		("CPUContext.platform", bpo::value<std::string>(&cpu_platform)->default_value("Intel(R) Corporation"), "CPU platform")
		("GPUContext.platform", bpo::value<std::string>(&gpu_platform)->default_value("NVIDIA"), "GPU platform");

	bpo::variables_map vm;

	try 
	{
		bpo::store(bpo::parse_config_file(config_stream, desc), vm);
		bpo::notify(vm);
	}
	catch(bpo::error& e)
	{
		std::cout << "Error: " << e.what() << std::endl;
		std::cout << desc << std::endl;
	}

	config_stream.close();
}

int main(int argc, char* argv[])
{
	std::string config_filename = "config_split.ini";

	std::string cpu_platform, gpu_platform;

	parse_config(config_filename, cpu_platform, gpu_platform);

	std::cout << "Config: CPU Platform: " << cpu_platform << std::endl;
	std::cout << "Config: GPU Platform: " << gpu_platform << std::endl;

	//Initialize the singleton source
	vivid::CLContextSource cl_context_source(cpu_platform, gpu_platform);

	//pull the data
	cv::Mat exampleImage = cv::imread(exampleImagePath, 0);
	cv::Mat f_exampleImage;
	exampleImage.convertTo(f_exampleImage, CV_32FC1);
	cv::resize(f_exampleImage, f_exampleImage, cv::Size(f_exampleImage.size().width * 2, f_exampleImage.size().height));

	auto width = f_exampleImage.size().width;
	auto height = f_exampleImage.size().height;

	/*float split_coeff = 0.333333f;*/
	float split_coeff = 0.33333;
	if (argc > 1)
	{
		split_coeff = atof(argv[1]);
	}

	int sub_width = width * split_coeff + 1;

	////height split
	//cv::Mat cpu_image = exampleImage(cv::Rect(0,height * split_coeff,width, height * (1.0f - split_coeff))).clone();
	//cv::Mat gpu_image = exampleImage(cv::Rect(0,0,width, height * split_coeff)).clone();

	//width split
	cv::Mat cpu_image = f_exampleImage(cv::Rect(sub_width, 0, width-sub_width, height)).clone();
	cv::Mat gpu_image = f_exampleImage(cv::Rect(0, 0, sub_width, height)).clone();

	float* cpu_image_data = (float*) _mm_malloc(cpu_image.size().area() *sizeof(float), 256);
	float* gpu_image_data = (float*) _mm_malloc(gpu_image.size().area() *sizeof(float), 256);

	//for (int i = 0; i < cpu_image.size().area(); i++)
	//{
	//	cpu_image_data[i] = cpu_image.data[i];
	//}

	//for (int i = 0; i < gpu_image.size().area(); i++)
	//{
	//	gpu_image_data[i] = gpu_image.data[i];
	//}

	printf("CPU Image\twidth: %d,\theight: %d\n", cpu_image.size().width, cpu_image.size().height);
	printf("GPU Image\twidth: %d,\theight: %d\n", gpu_image.size().width, gpu_image.size().height);

	DeviceMatrixCL::Ptr dmpCL_gpu = 
		makeDeviceMatrixCL(gpu_image.size().height, gpu_image.size().width, vivid::DEVICE_GPU);
	DeviceMatrixCL_copyToDevice(*dmpCL_gpu, (float*) gpu_image_data);

	DeviceMatrixCL::Ptr dmpCL_cpu = 
		makeDeviceMatrixCL(cpu_image.size().height, cpu_image.size().width, vivid::DEVICE_CPU);
	DeviceMatrixCL_copyToDevice(*dmpCL_cpu, (float*) cpu_image_data);

	const int num_filters = 100;
	const int filter_dim = 3;

	FilterBank fb_cpu(filter_dim, num_filters, vivid::DEVICE_CPU);
	Classifier clf_cpu(128, 64, 8, 2, num_filters, vivid::DEVICE_CPU);

	FilterBank fb_gpu(filter_dim, num_filters, vivid::DEVICE_GPU);
	Classifier clf_gpu(128, 64, 8, 2, num_filters, vivid::DEVICE_GPU);

	clFinish(cl_context_source.getContext(vivid::DEVICE_CPU)->getCommandQueue());
	clFinish(cl_context_source.getContext(vivid::DEVICE_GPU)->getCommandQueue());
	double tic = omp_get_wtime();
	for (int i = 1000; i > 0; i--)
	{
		if (i % 100 == 0) std::cout << i << std::endl;
		if (0)
		{
			DeviceMatrixCL3D::Ptr ff_im_cpu = fb_cpu.apply_cl(dmpCL_cpu);
			DeviceMatrixCL::Ptr block_histogram_cpu = cell_histogram_dense_cl(
				ff_im_cpu, num_filters, 8, 0, 0, 
				cpu_image.size().height, cpu_image.size().width);
			//DeviceMatrixCL::Ptr result_cpu = clf_cpu.apply(block_histogram_cpu);
		}
		if (1)
		{
			DeviceMatrixCL3D::Ptr ff_im_gpu = fb_gpu.apply_cl(dmpCL_gpu);
			DeviceMatrixCL::Ptr block_histogram_gpu = cell_histogram_dense_cl(
				ff_im_gpu, num_filters, 8, 0, 0, 
				gpu_image.size().height, gpu_image.size().width);
			//DeviceMatrixCL::Ptr result_gpu = clf_gpu.apply(block_histogram_gpu);
		}		
		
	}

	clFinish(cl_context_source.getContext(vivid::DEVICE_GPU)->getCommandQueue());
	clFinish(cl_context_source.getContext(vivid::DEVICE_CPU)->getCommandQueue());

	//printf("Blocks CPU y: %d, x: %d\n", block_histogram_cpu->height, block_histogram_cpu->width);
	//printf("Blocks GPU y: %d, x: %d\n", block_histogram_gpu->height, block_histogram_gpu->width);

	double toc = omp_get_wtime();

	std::cout << "Total runtime: " << toc - tic << std::endl;

	//std::cout << "Press ENTER to end." << std::endl;
	//std::cin.ignore( std::numeric_limits <std::streamsize> ::max(), '\n' );

	return 0;
}