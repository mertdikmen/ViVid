// @(#)NumPyWrapper.cpp implementation of the NumPyWrapper class.
//
//////////////////////////////////////////////////////////////////////

#include "NumPyWrapper.hpp"

#define PY_ARRAY_UNIQUE_SYMBOL tb
#define NO_IMPORT
#include <numpy/arrayobject.h>

using namespace boost::python;

//////////////////////////////////////////////////////////////////////
// Static initializer
//////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////
NumPyArray::NumPyArray(const boost::python::object& source)
{
  // The reference counting is a bit tricky, but I think that
  // PyArray_FromAny borrows the reference of the first argument.  It
  // seems to be used in the numpy source code with arguments to
  // functions, which according to the python c api documentation are
  // borrowed references.
  PyObject* contig
    = PyArray_FromAny(source.ptr(), PyArray_DescrFromType(PyArray_FLOAT),
                      1, 1, NPY_ARRAY_CARRAY, NULL);
  
  boost::python::expect_non_null(contig);
  // http://mail.python.org/pipermail/cplusplus-sig/2008-October/013895.html
  handle<> temp(contig);
  array = object(temp);
}

NumPyArray::NumPyArray(unsigned int size)
{
  npy_intp dim = {size};

  PyObject* arr = PyArray_ZEROS(1, &dim, PyArray_FLOAT,0);
  handle<> temp(arr);
  array = object(temp);
}

NumPyMatrix::NumPyMatrix(const boost::python::object& source)
{
  // The reference counting is a bit tricky, but I think that
  // PyArray_FromAny borrows the reference of the first argument.  It
  // seems to be used in the numpy source code with arguments to
  // functions, which according to the python c api documentation are
  // borrowed references.
  //printf("%f\n",source.ptr()[0]);
  PyObject* contig
    = PyArray_FromAny(source.ptr(), PyArray_DescrFromType(PyArray_FLOAT),
                      2, 2, NPY_ARRAY_CARRAY, NULL);
  // http://mail.python.org/pipermail/cplusplus-sig/2008-October/013895.html
  boost::python::expect_non_null(contig);
  handle<> temp(contig);
  array = object(temp);
}

NumPyMatrix::NumPyMatrix(unsigned int height, unsigned int width) {

  npy_intp dims[2] = {height, width};

  PyObject* arr = PyArray_SimpleNew(2, dims, PyArray_FLOAT);
  handle<> temp(arr);
  array = object(temp);
}


NumPyMatrix3D::NumPyMatrix3D(const boost::python::object& source)
{
  // The reference counting is a bit tricky, but I think that
  // PyArray_FromAny borrows the reference of the first argument.  It
  // seems to be used in the numpy source code with arguments to
  // functions, which according to the python c api documentation are
  // borrowed references.
  PyObject* contig
    = PyArray_FromAny(source.ptr(), PyArray_DescrFromType(PyArray_FLOAT),
                      3, 3, NPY_ARRAY_CARRAY, NULL);
  // http://mail.python.org/pipermail/cplusplus-sig/2008-October/013895.html
  handle<> temp(contig);
  array = object(temp);
}

NumPyMatrix3D::NumPyMatrix3D(unsigned int dim_t, unsigned int dim_y,
                           unsigned int dim_x) {
  npy_intp dims[3] = {dim_t, dim_y, dim_x};

  PyObject* arr = PyArray_SimpleNew(3, dims, PyArray_FLOAT);
  handle<> temp(arr);
  array = object(temp);
}


NumPyImage::NumPyImage(const boost::python::object& source)
{
  // The reference counting is a bit tricky, but I think that
  // PyArray_FromAny borrows the reference of the first argument.  It
  // seems to be used in the numpy source code with arguments to
  // functions, which according to the python c api documentation are
  // borrowed references.
  PyObject* contig
    = PyArray_FromAny(source.ptr(), PyArray_DescrFromType(PyArray_UINT8),
                      2, 3, NPY_ARRAY_CARRAY, NULL);
  // http://mail.python.org/pipermail/cplusplus-sig/2008-October/013895.html
  handle<> temp(contig);
  array = object(temp);
}

NumPyImage::NumPyImage(unsigned int height,
                       unsigned int width,
                       unsigned int depth) {
  npy_intp dims[3] = {height, width, depth};

  PyObject* arr = PyArray_SimpleNew(depth==1 ? 2: 3,
                                    dims, PyArray_UINT8);
  handle<> temp(arr);
  array = object(temp);
}

//////////////////////////////////////////////////////////////////////
// Implementation
//////////////////////////////////////////////////////////////////////

void NumPyWrapper::init() {
//  import_array();
}

//NUMPY ARRAY
unsigned int NumPyArray::size() const {
  return PyArray_DIM(array.ptr(), 0);
}

float* NumPyArray::data() {
  return (float*)PyArray_DATA(array.ptr());
}

const float* NumPyArray::data() const {
  return (float*)PyArray_DATA(array.ptr());
}



//NUMPY MATRIX
unsigned int NumPyMatrix::width() const {
  return PyArray_DIM(array.ptr(), 1);
}

unsigned int NumPyMatrix::height() const {
  return PyArray_DIM(array.ptr(), 0);
}

float* NumPyMatrix::data() {
  return (float*)PyArray_DATA(array.ptr());
}

const float* NumPyMatrix::data() const {
  return (float*)PyArray_DATA(array.ptr());
}


unsigned int NumPyMatrix3D::dim_t() const {
  return PyArray_DIM(array.ptr(), 0);
}

unsigned int NumPyMatrix3D::dim_y() const {
  return PyArray_DIM(array.ptr(), 1);
}

unsigned int NumPyMatrix3D::dim_x() const {
  return PyArray_DIM(array.ptr(), 2);
}

float* NumPyMatrix3D::data() {
  return (float*)PyArray_DATA(array.ptr());
}

const float* NumPyMatrix3D::data() const {
  return (float*)PyArray_DATA(array.ptr());
}



unsigned int NumPyImage::width() const {
  return PyArray_DIM(array.ptr(), 1);
}

unsigned int NumPyImage::height() const {
  return PyArray_DIM(array.ptr(), 0);
}

unsigned int NumPyImage::depth() const {
  return (PyArray_NDIM(array.ptr()) == 2) ? 1 : PyArray_DIM(array.ptr(), 2);
}

unsigned char* NumPyImage::data() {
  return (unsigned char*)PyArray_DATA(array.ptr());
}

const unsigned char* NumPyImage::data() const {
  return (unsigned char*)PyArray_DATA(array.ptr());
}
