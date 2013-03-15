#include "vivid.hpp"

class ClassifierOCV
{
public:
	ClassifierOCV(
		const int window_height, const int window_width, 
		const int cell_size, const int block_size, const int dict_size):
	_window_height(window_height), _window_width(window_width), _dict_size(dict_size),
		_cell_size(cell_size), _block_size(block_size)
	{
		_n_cells_x = _window_width / cell_size;
		_n_cells_y = _window_height / cell_size;

		_n_blocks_x = _n_cells_x - _block_size + 1;
		_n_blocks_y = _n_cells_y - _block_size + 1;

		_n_total_coeff = _block_size * _block_size * _n_blocks_x * _n_blocks_y * _dict_size;

		coefficients = new float[_n_total_coeff];

		for (int i = 0; i < _n_total_coeff; i++)
		{
			coefficients[i] = float( std::rand() ) / RAND_MAX;
		}

		classifierCV = cv::Mat(_n_total_coeff / _dict_size, _dict_size, CV_32FC1, coefficients);
	}

	cv::Mat apply(cv::Mat& input)
	{
		cv::Mat output(input.rows, classifierCV.rows, CV_32FC1);

		float* output_data = (float*) output.data;

		for (int i = 0; i < input.rows; i++)
		{
			for (int j = 0; j < classifierCV.rows; j++)
			{
				//std::cout << i << "\t" << j << std::endl;
				cv::Mat diff = classifierCV.row(j) - input.row(i);
				output_data[i * output.cols + j] = cv::norm(diff);				
			}
		}

		return output;
	}

public:
	const int _window_height;
	const int _window_width;
	const int _cell_size;
	const int _block_size;
	const int _dict_size;
	int _n_total_coeff;
	int _n_cells_x, _n_cells_y;
	int _n_blocks_x, _n_blocks_y;

	float* coefficients;

	cv::Mat classifierCV;
};

class Classifier
{
public:
	Classifier(
		const int window_height, const int window_width, 
		const int cell_size, const int block_size, const int dict_size, 
		vivid::DeviceType device_type):
	_window_height(window_height), _window_width(window_width), _dict_size(dict_size),
		_cell_size(cell_size), _block_size(block_size)
	{
		_n_cells_x = _window_width / cell_size;
		_n_cells_y = _window_height / cell_size;

		_n_blocks_x = _n_cells_x - _block_size + 1;
		_n_blocks_y = _n_cells_y - _block_size + 1;

		_n_total_coeff = _block_size * _block_size * _n_blocks_x * _n_blocks_y * _dict_size;

		coefficients = new float[_n_total_coeff];

		classifierCL = makeDeviceMatrixCL(_n_total_coeff / _dict_size, _dict_size, device_type);

		for (int i = 0; i < _n_total_coeff; i++)
		{
			coefficients[i] = float( std::rand() ) / RAND_MAX;
		}

		DeviceMatrixCL_copyToDevice(*classifierCL, coefficients);

		result = makeDeviceMatrixCL(1,1,device_type);
	};

	//TODO: read from file Classifier(const string file_name){};

	DeviceMatrixCL::Ptr apply(DeviceMatrixCL::Ptr blocks)
	{
		pwdist_cl(classifierCL, blocks, result);

		return result;
	};

	~Classifier()
	{
		if (coefficients != NULL)
		{
			delete[] coefficients;
			coefficients = NULL;
		}
	};

	DeviceMatrixCL::Ptr classifierCL;
private:
	const int _window_height;
	const int _window_width;
	const int _cell_size;
	const int _block_size;
	const int _dict_size;
	int _n_total_coeff;
	int _n_cells_x, _n_cells_y;
	int _n_blocks_x, _n_blocks_y;

	float* coefficients;

	DeviceMatrixCL::Ptr result;
};