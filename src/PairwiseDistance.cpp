#include "PairwiseDistance.hpp"
#include "PairwiseDistanceLocal.hpp"

DeviceMatrix::Ptr pwdist_cuda( const DeviceMatrix::Ptr& features_train,
                               const DeviceMatrix::Ptr& features_test){

    DeviceMatrix::Ptr out = makeDeviceMatrix(features_train->height,
                                             features_test->height);
    pwdist_generic(features_train.get(), features_test.get(), out.get(), EUCLIDEAN);
    return out;
}

DeviceMatrix::Ptr pwdot_cuda( const DeviceMatrix::Ptr& features_train,
                               const DeviceMatrix::Ptr& features_test){

    DeviceMatrix::Ptr out = makeDeviceMatrix(features_train->height,
                                             features_test->height);
    pwdist_generic(features_train.get(), features_test.get(), out.get(), DOTPRODUCT);
    return out;
}

DeviceMatrix::Ptr pwabsdot_cuda( const DeviceMatrix::Ptr& features_train,
                               const DeviceMatrix::Ptr& features_test){

    DeviceMatrix::Ptr out = makeDeviceMatrix(features_train->height,
                                             features_test->height);
    pwdist_generic(features_train.get(), features_test.get(), out.get(), ABSDOTPRODUCT);
    return out;
}

DeviceMatrix::Ptr pwchisq_cuda( const DeviceMatrix::Ptr& features_train,
                              const DeviceMatrix::Ptr& features_test){

    DeviceMatrix::Ptr out = makeDeviceMatrix(features_train->height,
                                             features_test->height);
    pwdist_generic(features_train.get(), features_test.get(), out.get(), CHISQUARED);
    return out;
}

DeviceMatrix::Ptr pwcityblock_cuda( const DeviceMatrix::Ptr& features_train,
                                    const DeviceMatrix::Ptr& features_test){

    DeviceMatrix::Ptr out = makeDeviceMatrix(features_train->height,
                                             features_test->height);
    pwdist_generic(features_train.get(), features_test.get(), out.get(), CITYBLOCK);
    return out;
}

DeviceMatrix::Ptr argmin_cuda(const DeviceMatrix::Ptr& matrix)
{
    DeviceMatrix::Ptr out = makeDeviceMatrix(matrix->height, 1);
    argmin_cuda_local(matrix.get(), out.get());
    return out;
}

DeviceMatrix::Ptr argmax_cuda(const DeviceMatrix::Ptr& matrix)
{
    DeviceMatrix::Ptr out = makeDeviceMatrix(matrix->height, 1);
    argmax_cuda_local(matrix.get(), out.get());
    return out;
}

DeviceMatrix::Ptr min_cuda(const DeviceMatrix::Ptr& matrix)
{
    DeviceMatrix::Ptr out = makeDeviceMatrix(matrix->height, 1);
    min_cuda_local(matrix.get(), out.get());
    return out;
}

DeviceMatrix::Ptr max_cuda(const DeviceMatrix::Ptr& matrix)
{
    DeviceMatrix::Ptr out = makeDeviceMatrix(matrix->height, 1);
    max_cuda_local(matrix.get(), out.get());
    return out;
}
