#ifndef _NEARESTNEIGHBORLOCAL_HPP_
#define _NEARESTNEIGHBORLOCAL_HPP_ 1

#include "DeviceMatrix.hpp"

void pwdist_generic( const DeviceMatrix* features_train,
                     const DeviceMatrix* features_test,
                     DeviceMatrix* output,
                     int type);

void argmin_cuda_local(const DeviceMatrix* matrix, DeviceMatrix* output);
void argmax_cuda_local(const DeviceMatrix* matrix, DeviceMatrix* output);

void min_cuda_local(const DeviceMatrix* matrix, DeviceMatrix* output);
void max_cuda_local(const DeviceMatrix* matrix, DeviceMatrix* output);

#endif /* _NEARESTNEIGHBORLOCAL_HPP_ */
