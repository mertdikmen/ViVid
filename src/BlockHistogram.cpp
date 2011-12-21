#include "BlockHistogram.hpp"
#include "BlockHistogramLocal.hpp"
#include "NumPyWrapper.hpp"

using namespace boost::python;

DeviceMatrix3D::Ptr cell_histogram_dense_cuda(
    const DeviceMatrix::Ptr& assignment_mat,
    const DeviceMatrix::Ptr& weight_mat,
    const int max_bin, const int cell_size, 
    object& start_inds, object& stop_inds)
{
    
    NumPyArray start_arr(start_inds);
    NumPyArray stop_arr(stop_inds);

    const int start_y = (int) start_arr.data()[0];
    const int start_x = (int) start_arr.data()[1];

    const int stop_y = (int) stop_arr.data()[0];
    const int stop_x = (int) stop_arr.data()[1];


    int n_parts_y = (stop_y - start_y) / cell_size;

    int n_parts_x = (stop_arr.data()[1] - start_arr.data()[1] ) / cell_size;

    #ifdef METHOD_2
    n_parts_y += (n_parts_y % 2);
    n_parts_x += (n_parts_x % 2);
    #endif

    DeviceMatrix3D::Ptr histogram = makeDeviceMatrix3D(
        n_parts_y, n_parts_x, max_bin);

    cell_histogram_dense_device(histogram.get(),
                                assignment_mat.get(),
                                weight_mat.get(),
                                max_bin,
                                cell_size,
                                start_y, start_x);
   
    return histogram;
}

