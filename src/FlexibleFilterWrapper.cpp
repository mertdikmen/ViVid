#include "FlexibleFilterWrapper.hpp"

int update_filter_bank(object& filterbank_array){
    NumPyMatrix3D arr(filterbank_array);
    //Here turn it into a float array and pass it to the FlexibleFilter.cpp
    int data_size = arr.dim_t() * arr.dim_x() * arr.dim_y();
    set_filter_bank(arr.data(), data_size);

    return 0;
}

void export_FlexibleFilter()
{
    def<DeviceMatrix3D::Ptr (const DeviceMatrix::Ptr&, 
                             const int dim_t, const int dim_y, const int dim_x, const int nchannels,
                             const int optype ) >
        ("_filter_frame_cuda", filter_frame_cuda);

    def<DeviceMatrix3D::Ptr (const DeviceMatrix::Ptr&, 
                             const int dim_t, const int dim_y, const int dim_x, const int nchannels,
                             const int optype ) >
        ("_filter_frame_cuda_noargmin", filter_frame_cuda_noargmin);

    def<DeviceMatrix3D::Ptr (const DeviceMatrix3D::Ptr&,
                             const int cell_size,
                             const int offset_y, const int offset_x,
                             const int n_bins) >
        ("_get_cell_histograms_cuda", get_cell_histograms_cuda);

    def ("_update_filter_bank", update_filter_bank);
}
