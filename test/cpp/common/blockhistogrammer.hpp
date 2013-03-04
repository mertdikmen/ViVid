#ifndef BLOCKHISTOGRAMMER_HPP
#define BLOCKHISTOGRAMMER_HPP

#include "vivid.hpp"

class BlockHistogrammer
{
public:
	BlockHistogrammer(const int _n_filters, const int _cell_size, vivid::DeviceType _device_type):
		n_filters(_n_filters),  cell_size(_cell_size), device_type(_device_type)
	{
		block_histogram = makeDeviceMatrixCL(1, 1, device_type);
	}

	DeviceMatrixCL::Ptr apply(DeviceMatrixCL3D::Ptr& ff_im, 
		const int start_y = 0, const int start_x = 0,
		int stop_y = -1, int stop_x = -1)
	{
		if (stop_y == -1){ stop_y = ff_im->dim_y; }
		if (stop_x == -1){ stop_x = ff_im->dim_x; }

		cell_histogram_dense_cl(ff_im, block_histogram, n_filters, cell_size, start_y, start_x, stop_y, stop_x);

		return block_histogram;
	}

private:
	int n_filters;
	int cell_size;

	vivid::DeviceType device_type;

	DeviceMatrixCL::Ptr block_histogram;
};

#endif