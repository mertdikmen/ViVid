#ifndef _CONVOLUTION_HPP_
#define _CONVOLUTION_HPP_ 1

#include "DeviceMatrix.hpp"


//! Perform a 3D convolution.
void convolve3d(const DeviceMatrix3D::Ptr& video,
                const DeviceMatrix3D::Ptr& kernel,
                DeviceMatrix3D::Ptr& output);

//! Perform a 3D convolution using a specific algorithm
void convolve3d(const DeviceMatrix3D::Ptr& video,
                const DeviceMatrix3D::Ptr& kernel,
                DeviceMatrix3D::Ptr& output,
                unsigned int algorithm);

//! Convenience function that allocates the output
DeviceMatrix3D::Ptr convolve3d(const DeviceMatrix3D::Ptr& video,
                               const DeviceMatrix3D::Ptr& kernel);

//! Call that enforces a specific path
DeviceMatrix3D::Ptr convolve3d_specific(const DeviceMatrix3D::Ptr& video,
                                        const DeviceMatrix3D::Ptr& kernel,
                                        int algorithm);

//! Perform a 1D (complex) convolution in the t direction
void convolve_complex_t(const DeviceMatrix3D::Ptr& video,
                        const DeviceMatrix3D::Ptr& kernel,
                        float scale,
                        DeviceMatrix3D::Ptr& output);

//! Call that enforces a specific path
void convolve_complex_t_specific(const DeviceMatrix3D::Ptr& video,
                                 const DeviceMatrix3D::Ptr& kernel,
                                 float scale,
                                 DeviceMatrix3D::Ptr& output,
                                 unsigned int algorithm);

//! Debugging helper
unsigned int debug_convolution_algorithm_used();

#endif /* _CONVOLUTION_HPP_ */
