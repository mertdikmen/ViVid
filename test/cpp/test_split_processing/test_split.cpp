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
	std::ifstream config_stream(config_filename.c_str());
	bpo::options_description desc("Options");
	desc.add_options()
		("CPUContext.platform", 
		 bpo::value<std::string>(&cpu_platform)->default_value("Intel"), "CPU platform")
		("GPUContext.platform", 
		 bpo::value<std::string>(&gpu_platform)->default_value("NVIDIA"), "GPU platform");

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
	
	cv::Mat exampleImage = cv::imread(exampleImagePath, 0);

	std::cout << "CPU Platform: " << cpu_platform << std::endl;
	std::cout << "GPU Platform: " << gpu_platform << std::endl;
	
	//const int num_filters = 100;
	//const int filter_dim = 3;

	//FilterBank fb(filter_dim, num_filters);
	//fb.set_on_device();

	//create vivid opencl contexts
	
	return 0;
}