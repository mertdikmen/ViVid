#include <iostream>
#include "vivid.hpp"

class FilterBankOCV
{
public:
	FilterBankOCV(const int ndim, const int nfilters):
		n_filters(nfilters), n_dim(ndim)
	{
		std::cout << n_filters << std::endl;
		std::cout << n_dim << std::endl;
		data = new float[n_filters * n_dim * n_dim];
		for (int i = 0; i < n_filters * n_dim * n_dim; i++)
		{
			data[i] = float( std::rand() ) / RAND_MAX;
		}

		for (int i = 0; i < n_filters; i++)
		{
			filters.push_back(cv::Mat(n_dim, n_dim, CV_32FC1, data + i * (n_dim * n_dim)));
		}
	}

	~FilterBankOCV()
	{
		delete data;
	}

	cv::Mat apply(cv::Mat& input)
	{
		cv::Mat max_val = cv::Mat::zeros(input.size(), input.type());
		cv::Mat max_ind = cv::Mat::zeros(input.size(), input.type());
		cv::Mat temp_result(input.size(), input.type());
		for (int i = 0; i < n_filters; i++)
		{
			cv::filter2D(input, temp_result, CV_32F, filters[i]);
			//cv::matchTemplate(input, filters[i], temp_result, CV_TM_CCORR);
			temp_result = cv::abs(temp_result);
			for (int ii = 0; ii < temp_result.size().area(); ii++)
			{
				if (temp_result.data[ii] > max_val.data[ii])
				{
					max_val.data[ii] = temp_result.data[ii];
					max_ind.data[ii] = ii;
				}
			}
		}

		return max_val;
	}

	float* data;

private:
	int n_dim;
	int n_filters;

	std::vector<cv::Mat> filters;
};

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
