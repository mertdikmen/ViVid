#include <fstream>
#include "opencv2\opencv.hpp"
#include "vivid.hpp"

#include "boost/program_options.hpp"

#include "classifier.hpp"
#include "filterbank.hpp"

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

	float* f_imData = (float*) f_exampleImage.data;
	DeviceMatrixCL::Ptr dmpCL_cpu = 
		makeDeviceMatrixCL(f_exampleImage.size().height, f_exampleImage.size().width, vivid::DEVICE_CPU);
	DeviceMatrixCL_copyToDevice(*dmpCL_cpu, f_imData);

	DeviceMatrixCL::Ptr dmpCL_gpu = 
		makeDeviceMatrixCL(f_exampleImage.size().height, exampleImage.size().width, vivid::DEVICE_GPU);
	DeviceMatrixCL_copyToDevice(*dmpCL_gpu, f_imData);

	const int num_filters = 100;
	const int filter_dim = 3;

	FilterBank fb_cpu(filter_dim, num_filters);
	fb_cpu.set_on_device(vivid::DEVICE_CPU);

	FilterBank fb_gpu(filter_dim, num_filters);
	fb_gpu.set_on_device(vivid::DEVICE_GPU);

	Classifier clf_cpu(128, 64, 8, 2, num_filters, vivid::DEVICE_CPU);
	Classifier clf_gpu(128, 64, 8, 2, num_filters, vivid::DEVICE_GPU);

	DeviceMatrixCL3D::Ptr ff_im_cpu = fb_cpu.apply_cl(dmpCL_cpu);
	DeviceMatrixCL3D::Ptr ff_im_gpu = fb_gpu.apply_cl(dmpCL_gpu);

	DeviceMatrixCL::Ptr block_histogram_cpu = cell_histogram_dense_cl(
		ff_im_cpu, num_filters, 8, 0, 0, 
		exampleImage.size().height, exampleImage.size().width);

	DeviceMatrixCL::Ptr block_histogram_gpu = cell_histogram_dense_cl(
		ff_im_gpu, num_filters, 8, 0, 0, 
		exampleImage.size().height, exampleImage.size().width);

	DeviceMatrixCL::Ptr result_cpu = clf_cpu.apply(block_histogram_cpu);
	DeviceMatrixCL::Ptr result_gpu = clf_gpu.apply(block_histogram_gpu);

	std::cout << "Press ENTER to end." << std::endl;
	std::cin.ignore( std::numeric_limits <std::streamsize> ::max(), '\n' );

	return 0;
}