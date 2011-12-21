#include "DeviceMatrixWrapper.hpp"

#define PY_ARRAY_UNIQUE_SYMBOL tb
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>

#include <iostream>

#include <cuda_runtime.h>
using namespace boost::python;

//#include <cutil.h>

// Cribbed (and modified) from cutil.h (can't seem to include the
// whole thing)
#  define CUDA_SAFE_CALL_NO_SYNC( call) do {                                 \
    cudaError_t err = call;                                                  \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        exit(EXIT_FAILURE);                                                  \
    } } while (0)


DeviceMatrix::Ptr makeDeviceMatrix(boost::python::object& array)
{
  // If we already have a DeviceMatrix, just return it.  This sholuld
  // help unify code paths.
  extract<DeviceMatrix::Ptr> get_matrix(array);
  if (get_matrix.check()) {
    return get_matrix();
  }

  NumPyMatrix arr(array);

  DeviceMatrix::Ptr retval = makeDeviceMatrix(arr.height(), arr.width());

  DeviceMatrix_copyToDevice(*retval, arr);
  return retval;
}

boost::python::object DeviceMatrix_copyFromDevice(const DeviceMatrix& self)
{
  NumPyMatrix retval(self.height, self.width);
  //printf("reading %p (%i x %i)\n", self.data, self.width, self.height);
  
  if ((self.width > 0) && (self.height > 0)) {
    const size_t widthInBytes = self.width * sizeof(float);
    CUDA_SAFE_CALL_NO_SYNC
      (cudaMemcpy2D(retval.data(), widthInBytes,
                    self.data, self.pitch * sizeof(float),
                    widthInBytes, self.height,
                    cudaMemcpyDeviceToHost));
  }

  return retval.array;
}

void DeviceMatrix_copyToDevice(DeviceMatrix& self,
                               const NumPyMatrix& matrix)
{
  assert(self.width  == matrix.width());
  assert(self.height == matrix.height());

  if ((self.width > 0) && (self.height > 0)) {
    const size_t widthInBytes = self.width * sizeof(float);
    CUDA_SAFE_CALL_NO_SYNC
      (cudaMemcpy2D(self.data, self.pitch * sizeof(float),
                    matrix.data(), widthInBytes,
                    widthInBytes, self.height,
                    cudaMemcpyHostToDevice));
  }
}

DeviceMatrix3D::Ptr makeDeviceMatrix3D(const boost::python::object& array)
{
  PyObject* contig
    = PyArray_FromAny(array.ptr(), PyArray_DescrFromType(PyArray_FLOAT),
                      3, 3, NPY_CARRAY, NULL);
  handle<> temp(contig);
  object arr(temp);

  DeviceMatrix3D::Ptr retval = makeDeviceMatrix3D(PyArray_DIM(arr.ptr(), 0),
                                                  PyArray_DIM(arr.ptr(), 1),
                                                  PyArray_DIM(arr.ptr(), 2));
  DeviceMatrix3D_copyToDevice(*retval, arr);
  return retval;
}

#if 0
// It would seem that cudaMemcpy3D would be the ideal function for the
// job, but it fails mysteriously if x*y > 2**18
boost::python::object DeviceMatrix3D_copyFromDevice(const DeviceMatrix3D& self)
{
    npy_intp dims[3] = {self.dim_t, self.dim_y, self.dim_x};

    PyObject* arr = PyArray_New(&PyArray_Type, 3, dims, PyArray_FLOAT,
                                NULL, NULL, 0, NPY_C_CONTIGUOUS, NULL);
    handle<> temp(arr);
    object retval(temp);

    if ((self.dim_x > 0) && (self.dim_y > 0) && (self.dim_t > 0)) {
        // Largely cribbed from
        // http://forums.nvidia.com/lofiversion/index.php?t77910.html

        cudaMemcpy3DParms copyParams = {0};
        /**
         * @todo Redefine DeviceMatrix3D to be closer to the form that
         * the library wants.
         */
        copyParams.srcPtr // Device
          = make_cudaPitchedPtr((void*)self.data,
                                self.pitch_y * sizeof(float),
                                self.dim_x,
                                self.pitch_t / self.pitch_y);

        copyParams.dstPtr // Host
          = make_cudaPitchedPtr(PyArray_DATA(retval.ptr()),
                                self.dim_x * sizeof(float),
                                self.dim_x,
                                self.dim_y);

        copyParams.kind = cudaMemcpyDeviceToHost;

        copyParams.extent
          = make_cudaExtent(self.dim_x*sizeof(float), self.dim_y, self.dim_t);

        CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy3D(&copyParams));
    }

  return retval;
}
#else
// Hack around problem with cudaMemcpy3D by using cudaMemcpy2D
boost::python::object DeviceMatrix3D_copyFromDevice(const DeviceMatrix3D& self)
{
    npy_intp dims[3] = {self.dim_t, self.dim_y, self.dim_x};

    PyObject* arr = PyArray_New(&PyArray_Type, 3, dims, PyArray_FLOAT,
                                NULL, NULL, 0, 0, NULL);
    handle<> temp(arr);
    object retval(temp);

    if ((self.dim_x == 0) || (self.dim_y == 0) || (self.dim_t == 0)) {
        // Bail early if there is nothing to copy
        return retval;
    }

    if (self.pitch_t == self.dim_y * self.pitch_y) {
        // Shortcut if we're packed in the t direction
        const size_t widthInBytes = self.dim_x * sizeof(float);
        CUDA_SAFE_CALL_NO_SYNC
            (cudaMemcpy2D(PyArray_DATA(retval.ptr()), widthInBytes,
                          self.data, self.pitch_y * sizeof(float),
                          widthInBytes, self.dim_y * self.dim_t,
                          cudaMemcpyDeviceToHost));

        return retval;
    }

    // Do a series of copies to fill in the 3D array
    for (size_t t=0; t < self.dim_t; t++) {
        const size_t widthInBytes = self.dim_x * sizeof(float);
        float* host_start = (float*)PyArray_DATA(retval.ptr())
            + t * self.dim_y * self.dim_x;
        float* device_start = self.data + t * self.pitch_t;
        CUDA_SAFE_CALL_NO_SYNC
            (cudaMemcpy2D(host_start, widthInBytes,
                          device_start, self.pitch_y * sizeof(float),
                          widthInBytes, self.dim_y,
                          cudaMemcpyDeviceToHost));
    }
    return retval;
}
#endif
void DeviceMatrix3D_copyToDevice(DeviceMatrix3D& self,
                                 const object& array)
{
    PyObject* contig
        = PyArray_FromAny(array.ptr(), PyArray_DescrFromType(PyArray_FLOAT),
                          3, 3, NPY_CARRAY, NULL);
    handle<> temp(contig);
    object arr(temp);

    // Make sure that we are packed in the t direction
    assert(self.pitch_t == self.dim_y * self.pitch_y);

    if ((self.dim_x > 0) && (self.dim_y > 0) && (self.dim_t > 0)) {
        const size_t widthInBytes = self.dim_x * sizeof(float);
        CUDA_SAFE_CALL_NO_SYNC
        (cudaMemcpy2D(self.data, self.pitch_y * sizeof(float),
                      PyArray_DATA(arr.ptr()), widthInBytes,
                      widthInBytes, self.dim_y * self.dim_t,
                      cudaMemcpyHostToDevice));
    }
}


MCudaMatrix3D::Ptr makeMCudaMatrix3D(const object& array)
{
  PyObject* contig
    = PyArray_FromAny(array.ptr(), PyArray_DescrFromType(PyArray_FLOAT),
                      3, 3, NPY_CARRAY, NULL);
  handle<> temp(contig);
  object arr(temp);

  MCudaMatrix3D::Ptr retval = makeMCudaMatrix3D(PyArray_DIM(arr.ptr(), 0),
                                                PyArray_DIM(arr.ptr(), 1),
                                                PyArray_DIM(arr.ptr(), 2));
  memcpy(retval->data, PyArray_DATA(arr.ptr()),
         retval->dim_t * retval->dim_y * retval->dim_x * sizeof(float));
  return retval;
}

boost::python::object MCudaMatrix3D_copyFromDevice(const MCudaMatrix3D& self)
{
    npy_intp dims[3] = {self.dim_t, self.dim_y, self.dim_x};

    PyObject* arr = PyArray_New(&PyArray_Type, 3, dims, PyArray_FLOAT,
                                NULL, NULL, 0, 0, NULL);
    handle<> temp(arr);
    object retval(temp);

    /**
     * @todo Avoid the copy
     */
    memcpy(PyArray_DATA(retval.ptr()), self.data,
           self.dim_t * self.dim_y * self.dim_x * sizeof(float));

    return retval;
}




#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
void export_DeviceMatrix()
{
    class_<DeviceMatrix, DeviceMatrix::Ptr >
        ("DeviceMatrix", no_init)
        .def("__init__",
             make_constructor<DeviceMatrix::Ptr (object&)>
             (makeDeviceMatrix))
        .def("mat", DeviceMatrix_copyFromDevice);

    class_<DeviceMatrix3D, DeviceMatrix3D::Ptr >
        ("DeviceMatrix3D", no_init)
        .def("__init__",
             make_constructor<DeviceMatrix3D::Ptr (const object&)>
             (makeDeviceMatrix3D))
        .def("mat", DeviceMatrix3D_copyFromDevice)
        .def("set", DeviceMatrix3D_copyToDevice)
        .def("crop", cropDeviceMatrix3D)
      ;
    def("_makeDeviceMatrix3DPacked", makeDeviceMatrix3DPacked);

    // Don't tell python about the subclass relationship -- we should
    // try to keep this as distinct from DeviceMatrix3D as possible
    class_<MCudaMatrix3D, MCudaMatrix3D::Ptr >
        ("MCudaMatrix3D", no_init)
        .def("__init__",
             make_constructor<MCudaMatrix3D::Ptr (const object&)>
             (makeMCudaMatrix3D))
        .def("mat", MCudaMatrix3D_copyFromDevice)
      ;

    class_<DeviceMatrix::PtrList >("DeviceMatrix3DList", no_init)
        .def(vector_indexing_suite<DeviceMatrix::PtrList, true>());
}
