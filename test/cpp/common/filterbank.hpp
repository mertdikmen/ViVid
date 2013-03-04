#include "vivid.hpp"

class FilterBank
{
public:
	FilterBank(const int ndim, const int nfilters, vivid::DeviceType device_type):
		n_filters(nfilters), n_dim(ndim)
	{
		data = new float[n_filters * n_dim * n_dim];
		for (int i = 0; i < n_filters * n_dim * n_dim; i++)
		{
			data[i] = float( std::rand() ) / RAND_MAX;
		}

		set_filter_bank_cl(data, n_filters * n_dim * n_dim, device_type);

		ff_image = makeDeviceMatrixCL3D(2, 100, 100, device_type);
	};

	//TODO: Read from file sFilterBank(const string file_name){};
	DeviceMatrixCL3D::Ptr apply_cl(DeviceMatrixCL::Ptr dmpCL)
	{
		filter_frame_cl_3(dmpCL, ff_image, n_filters, 1, FF_OPTYPE_COSINE);
		//return filter_frame_cl_3(dmpCL, n_filters, 1, FF_OPTYPE_COSINE);
		return ff_image;
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
	DeviceMatrixCL3D::Ptr ff_image;
	int n_dim;
	int n_filters;
};
