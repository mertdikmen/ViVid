#ifndef _FLEXIBLE_FILTER_WRAPPER_HPP_
#define _FLEXIBLE_FILTER_WRAPPER_HPP_ 1

#include "FlexibleFilter.hpp"
#include "NumPyWrapper.hpp"
#include <boost/python.hpp>

using namespace boost::python;

/**POWER TEST**/
DeviceMatrixCL3D::Ptr filter_frame_cl_3_batch(object& npy_array,
										const int dim_t, const int nchannels,
										const int optype, vivid::DeviceType device_type);

void export_FlexibleFilter();

int update_filter_bank(object& filterbank_array);

#endif /*_FLEXIBLE_FILTER_WRAPPER_HPP_*/
