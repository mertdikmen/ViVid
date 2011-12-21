#ifndef _BLOCK_HISTOGRAM_HPP_
#define _BLOCK_HISTOGRAM_HPP_ 1

#include "DeviceMatrix.hpp"

#include <boost/python.hpp>

DeviceMatrix3D::Ptr cell_histogram_dense_cuda(
    const DeviceMatrix::Ptr& assignment_mat,
    const DeviceMatrix::Ptr& weight_mat,
    const int max_bin, const int cell_size, 
    boost::python::object& start_inds, boost::python::object& stop_inds);


#endif //_BLOCK_HISTOGRAM_HPP_
