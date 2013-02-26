#include "vivid.hpp"

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

		  _n_total_coeff = _block_size * _block_size * _n_blocks_x * _n_blocks_y, _dict_size;

		  coefficients = new float[_n_total_coeff];

		  classifierCL = makeDeviceMatrixCL(_n_total_coeff / _dict_size, _dict_size, device_type);

		  for (int i = 0; i < _n_total_coeff; i++)
		  {
			  coefficients[i] = float( std::rand() ) / RAND_MAX;
		  }

		  DeviceMatrixCL_copyToDevice(*classifierCL, coefficients);
	  };

	  //TODO: read from file Classifier(const string file_name){};

	  DeviceMatrixCL::Ptr apply(DeviceMatrixCL::Ptr blocks)
	  {
		  return pwdist_cl(classifierCL, blocks);
	  };

	  ~Classifier()
	  {
		  if (coefficients != NULL)
		  {
			  delete[] coefficients;
			  coefficients = NULL;
		  }
	  };

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

	DeviceMatrixCL::Ptr classifierCL;
};