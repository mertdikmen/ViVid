#include "FlexibleFilter.hpp"
#include "FlexibleFilterLocal.hpp"

#include <cuda_runtime.h>

int set_filter_bank(float* filter_bank, int size){
    return update_filter_bank_internal(filter_bank,size); 
}

DeviceMatrix3D::Ptr filter_frame_cuda(const DeviceMatrix::Ptr& frame,
                                    const int dim_t, const int dim_y, const int dim_x, const int nchannels,
                                    const int optype){

    DeviceMatrix3D::Ptr out = makeDeviceMatrix3D(2, frame->height, frame->width / nchannels);

    //dist_filter(frame.get(), dim_t, dim_y, dim_x, filter_bank.get(), out.get(), optype);

    dist_filter2(frame.get(), dim_t, dim_y, dim_x, nchannels, out.get(), optype);

    return out;
}

DeviceMatrix3D::Ptr filter_frame_cuda_noargmin(const DeviceMatrix::Ptr& frame,
                                    const int dim_t, const int dim_y, const int dim_x, const int nchannels,
                                    const int optype){

    DeviceMatrix3D::Ptr out = makeDeviceMatrix3D(frame->height, frame->width / nchannels, dim_t);
    
    dist_filter_noargmin(frame.get(), dim_t, dim_y, dim_x, nchannels, out.get(), optype);

    return out;
}

DeviceMatrix3D::Ptr get_cell_histograms_cuda(const DeviceMatrix3D::Ptr& inds_and_weights,
                                             const int cell_size,
                                             const int offset_y, const int offset_x,
                                             const int n_bins){

#ifndef CUDA_NO_SM_11_ATOMIC_INTRINSICS
//	printf("WARNING! Not using atomics!\n");
#endif

    int frame_height = inds_and_weights->dim_y;
    int frame_width = inds_and_weights->dim_x;

    int n_cells_y = ( frame_height - offset_y ) / cell_size;
    int n_cells_x = ( frame_width - offset_x ) / cell_size;

    DeviceMatrix3D::Ptr out = makeDeviceMatrix3D(n_cells_y, n_cells_x, n_bins);
    out->zero();

    hist_all_cells(inds_and_weights.get(), out.get(), cell_size, offset_y, offset_x, n_bins);

    return out;
}
