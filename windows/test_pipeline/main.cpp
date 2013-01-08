#include "opencv2\opencv.hpp"
#include "vivid.hpp"

class FilterBank
{
public:
	FilterBank(const int ndim, const int nfilters):
	  n_filters(nfilters), n_dim(ndim)
	  {
		  data = new float[n_filters * n_dim * n_dim];
		  for (int i = 0; i < n_filters * n_dim * n_dim; i++)
		  {
			  data[i] = float( std::rand() ) / RAND_MAX;
		  }
	  };

	  //TODO: Read from file sFilterBank(const string file_name){};

	  DeviceMatrixCL3D::Ptr apply_cl(DeviceMatrixCL::Ptr dmpCL)
	  {
		  DeviceMatrixCL3D::Ptr retdm = filter_frame_cl_3(dmpCL, n_filters, 1, FF_OPTYPE_COSINE);
		  return retdm;
	  };

	  void set_on_device()
	  {
		  set_filter_bank_cl(data, n_filters * n_dim * n_dim);
	  };

	  ~FilterBank()
	  {
		  if (data!=NULL)
		  {
			  delete[] data;
			  data = NULL;
		  }
	  }

	  float* data;

private:
	int n_dim;
	int n_filters;
};

class Classifier
{
public:
	Classifier(const int window_height, const int window_width, const int cell_size, const int block_size, const int dict_size):
	  _window_height(window_height), _window_width(window_width), _dict_size(dict_size),
		  _cell_size(cell_size), _block_size(block_size)
	  {
		  _n_cells_x = _window_width / cell_size;
		  _n_cells_y = _window_height / cell_size;

		  _n_blocks_x = _n_cells_x - _block_size + 1;
		  _n_blocks_y = _n_cells_y - _block_size + 1;

		  _n_total_coeff = _block_size * _block_size * _n_blocks_x * _n_blocks_y, _dict_size;

		  coefficients = new float[_n_total_coeff];
		  
		  classifierCL = makeDeviceMatrixCL(_n_total_coeff / _dict_size, _dict_size);

		  for (int i = 0; i < _n_total_coeff; i++)
		  {
			  coefficients[i] = float( std::rand() ) / RAND_MAX;
		  }

		  DeviceMatrixCL_copyToDevice(*classifierCL, coefficients);
	  };

	  //TODO: read from file Classifier(const string file_name){};

	  DeviceMatrixCL::Ptr apply(DeviceMatrixCL::Ptr blocks)
	  {
		  pwdist_cl(classifierCL, blocks);
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

int main(int argc, char* argv[])
{
	static char* exampleImagePath = "..\\..\\..\\media\\kewell1.jpg";

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
	FilterBank fb(filter_dim, num_filters);
	fb.set_on_device();

	Classifier clf(128, 64, 8, 2, num_filters);

	//load the image on device
	DeviceMatrixCL::Ptr dmpCL = makeDeviceMatrixCL(height, width);
	DeviceMatrixCL_copyToDevice(*dmpCL, f_imData);

	DeviceMatrixCL3D::Ptr ff_im = fb.apply_cl(dmpCL);

	//DeviceMatrixCL3D::Ptr = cell_histogram_dense_cl(asCL, wtCL, num_filters, 8, 0, 0, height, width);

	//DeviceMatrixCL::Ptr result = clf.apply(
		
	return 0;
}