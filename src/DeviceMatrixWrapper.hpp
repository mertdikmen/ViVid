// @(#)DeviceMatrixWrapper.hpp Python interface for DeviceMatrix
//
//////////////////////////////////////////////////////////////////////

#ifndef _DEVICEMATRIXWRAPPER_HPP_
#define _DEVICEMATRIXWRAPPER_HPP_ 1

#include "DeviceMatrix.hpp"
#include "NumPyWrapper.hpp"

#include <boost/python.hpp>



DeviceMatrix::Ptr makeDeviceMatrix(const boost::python::object& array);
boost::python::object DeviceMatrix_copyFromDevice(const DeviceMatrix& self);
void DeviceMatrix_copyToDevice(DeviceMatrix& self,const NumPyMatrix& matrix);

DeviceMatrixCL::Ptr makeDeviceMatrixCL(const boost::python::object& array);
boost::python::object DeviceMatrixCL_copyFromDevice(const DeviceMatrixCL& self);
void DeviceMatrixCL_copyToDevice(DeviceMatrixCL& self, const NumPyMatrix& matrix);
void DeviceMatrixCL_copyToDevice(DeviceMatrixCL& self, const float* data);

DeviceMatrix3D::Ptr makeDeviceMatrix3D(const boost::python::object& array);
boost::python::object DeviceMatrix3D_copyFromDevice(const DeviceMatrix3D& self);
void DeviceMatrix3D_copyToDevice(DeviceMatrix3D& self,const boost::python::object& array);


DeviceMatrixCL3D::Ptr makeDeviceMatrixCL3D(const boost::python::object& array);
boost::python::object DeviceMatrixCL3D_copyFromDevice(const DeviceMatrixCL3D& self);
void DeviceMatrixCL3D_copyToDevice(DeviceMatrixCL3D& self,const boost::python::object& array);

void export_DeviceMatrix();

#endif /* _DEVICEMATRIXWRAPPER_HPP_ */
