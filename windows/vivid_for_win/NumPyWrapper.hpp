// @(#)NumPyWrapper.hpp interface for the NumPyWrapper class.
//
//////////////////////////////////////////////////////////////////////

#ifndef _NUMPYWRAPPER_HPP_
#define _NUMPYWRAPPER_HPP_ 1

#include <boost/python.hpp>
#include <vector>

//! Minimalist 2D float matrix class
/**
 * We actually store the data in a boost::python::object and provide
 * some accessors.
 */
struct NumPyArray{
  NumPyArray(const boost::python::object& source);
  NumPyArray(unsigned int size);

  boost::python::object array;
  unsigned int size() const;
  float* data();
  const float* data() const;
};

struct NumPyMatrix {
  NumPyMatrix(unsigned int height, unsigned int width);

  /**
   * @todo Consider making this explicit.  However, automatic
   * conversion from random python object is actually pretty
   * convenient.
   */
  NumPyMatrix(const boost::python::object& source);

  //! They underlying PyArrayObject
  /**
   * Note that we ensure that this is a 2D contigious float array.
   */
  boost::python::object array;
  unsigned int width() const;
  unsigned int height() const;
  float* data();
  const float* data() const;
};

//! Minimalist 3D float matrix class
/**
 * We actually store the data in a boost::python::object and provide
 * some accessors.
 */
struct NumPyMatrix3D {
  NumPyMatrix3D(unsigned int dim_t = 0,
                unsigned int dim_y = 0,
                unsigned int dim_x = 0);

  /**
   * @todo Consider making this explicit.  However, automatic
   * conversion from random python object is actually pretty
   * convenient.
   */
  NumPyMatrix3D(const boost::python::object& source);

  //! They underlying PyArrayObject
  /**
   * Note that we ensure that this is a 3D contigious float array.
   */
  boost::python::object array;
  unsigned int dim_x() const;
  unsigned int dim_y() const;
  unsigned int dim_t() const;
  float* data();
  const float* data() const;
};


//! Minimalist 2/3-channel uint8 converter/wrapper
/**
 * We actually store the data in a boost::python::object and provide
 * some accessors.
 */
struct NumPyImage {
    NumPyImage(unsigned int height = 0, unsigned int width = 0,
               unsigned int depth = 0);

  /**
   * @todo Consider making this explicit.  However, automatic
   * conversion from random python object is actually pretty
   * convenient.
   */
  NumPyImage(const boost::python::object& source);

  //! They underlying PyArrayObject
  /**
   * Note that we ensure that this is a 2D or 3D contigious uint8
   * array.
   */
  boost::python::object array;
  unsigned int width() const;
  unsigned int height() const;
  unsigned int depth() const;
  unsigned char* data();
  const unsigned char* data() const;
};



//! Little class to avoid pulling in NumPy headers
/**
 * They don't seem to compile under pedantic.
 */

class NumPyWrapper
{
public:
  // Calls import_array()
  static void init();
};

#endif /* _NUMPYWRAPPER_HPP_ */
