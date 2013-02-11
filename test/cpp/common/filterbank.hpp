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
		//  double tic0= omp_get_wtime();
		  DeviceMatrixCL3D::Ptr retdm = filter_frame_cl_3(dmpCL, n_filters, 1, FF_OPTYPE_COSINE);
	//	  double tic1= omp_get_wtime();
		//  std::cout << "---filter outside time: " << tic1 - tic0 << std::endl;
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
