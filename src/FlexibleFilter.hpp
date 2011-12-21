#ifndef _FLEXIBLE_FILTER_HPP_
#define _FLEXIBLE_FILTER_HPP_ 1

#include "DeviceMatrix.hpp"
#include <stdio.h>

#define FF_OPTYPE_EUCLIDEAN 0
#define FF_OPTYPE_COSINE 1

int set_filter_bank(float* filter_bank, int size);

/*
DeviceMatrix3D::Ptr filter_frame_cuda(const DeviceMatrix::Ptr& frame,
                                    const int dim_t, const int dim_y, const int dim_x, const int nchannels,
                                    const int optype);
*/

DeviceMatrix3D::Ptr filter_frame_cuda_noargmin(const DeviceMatrix::Ptr& frame,
                                    const int dim_t, const int dim_y, const int dim_x, const int nchannels,
                                    const int optype);

DeviceMatrix3D::Ptr get_cell_histograms_cuda(const DeviceMatrix3D::Ptr& inds_and_weights,
                                             const int cell_size,
                                             const int offset_y, const int offset_x,
                                             const int n_bins);

DeviceMatrix3D::Ptr filter_frame_cuda_3(const DeviceMatrix::Ptr& frame,
                                    const int dim_t, const int nchannels,
                                    const int optype);

DeviceMatrix3D::Ptr filter_frame_cuda_5(const DeviceMatrix::Ptr& frame,
                                    const int dim_t, const int nchannels,
                                    const int optype);

DeviceMatrix3D::Ptr filter_frame_cuda_7(const DeviceMatrix::Ptr& frame,
                                    const int dim_t, const int nchannels,
                                    const int optype);

static const unsigned int BLOCK_SIZE = 16;
static const unsigned int BLOCK_16 = 16;
static const unsigned int BLOCK_8 = 8;

#endif /* _FLEXIBLE_FILTER_HPP_ */
