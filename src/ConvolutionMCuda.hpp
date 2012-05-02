#ifndef _CONVOLUTIONMCUDA_HPP_
#define _CONVOLUTIONMCUDA_HPP_ 1

#include "DeviceMatrix.hpp"

bool try_convolution0_mcuda(const MCudaMatrix3D::Ptr& video,
                            const MCudaMatrix3D::Ptr& kernel,
                            MCudaMatrix3D::Ptr& output);

bool try_convolution2_mcuda(const MCudaMatrix3D::Ptr& video,
                            const MCudaMatrix3D::Ptr& kernel,
                            MCudaMatrix3D::Ptr& output);

bool try_convolution4_mcuda(const MCudaMatrix3D::Ptr& video,
                            const MCudaMatrix3D::Ptr& kernel,
                            MCudaMatrix3D::Ptr& output);

//! Call a particular MCUDA implementation
MCudaMatrix3D::Ptr convolve3d_mcuda(const MCudaMatrix3D::Ptr& video,
                                    const MCudaMatrix3D::Ptr& kernel,
                                    int algorithm);


#endif /* _CONVOLUTIONMCUDA_HPP_ */
