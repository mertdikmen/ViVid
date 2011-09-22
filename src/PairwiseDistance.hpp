#ifndef _PAIRWISE_DISTANCE_HPP_
#define _PAIRWISE_DISTANC_HPP_ 1

#define EUCLIDEAN 0
#define DOTPRODUCT 1
#define CHISQUARED 2
#define CITYBLOCK 3
#define ABSDOTPRODUCT 4

#include "DeviceMatrix.hpp"

DeviceMatrix::Ptr pwdist_cuda( const DeviceMatrix::Ptr& features_train,
                               const DeviceMatrix::Ptr& features_test);

DeviceMatrix::Ptr pwdot_cuda( const DeviceMatrix::Ptr& features_train, 
                              const DeviceMatrix::Ptr& features_test);

DeviceMatrix::Ptr pwabsdot_cuda( const DeviceMatrix::Ptr& features_train, 
                                 const DeviceMatrix::Ptr& features_test);

DeviceMatrix::Ptr pwchisq_cuda( const DeviceMatrix::Ptr& features_train,
                                const DeviceMatrix::Ptr& features_test);

DeviceMatrix::Ptr pwcityblock_cuda( const DeviceMatrix::Ptr& features_train,
                                     const DeviceMatrix::Ptr& features_test);

DeviceMatrix::Ptr argmin_cuda(const DeviceMatrix::Ptr& matrix);
DeviceMatrix::Ptr argmax_cuda(const DeviceMatrix::Ptr& matrix);

DeviceMatrix::Ptr min_cuda(const DeviceMatrix::Ptr& matrix);
DeviceMatrix::Ptr max_cuda(const DeviceMatrix::Ptr& matrix);


#endif /* _PAIRWISE_DISTANCE_HPP_ */
