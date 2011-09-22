#include "NumPyWrapper.hpp"
#include "DeviceMatrixWrapper.hpp"
#include "PairwiseDistanceWrapper.hpp"
#include "fastexp.h"

#include <boost/python.hpp>
#define PY_ARRAY_UNIQUE_SYMBOL tb
#include <numpy/arrayobject.h>

////////////////////////
// Faster operations
////////////////////////
boost::python::object fast_exp(boost::python::object& input_mat, const int approx_level){
    PyObject* input_block = PyArray_FromAny(input_mat.ptr(), PyArray_DescrFromType(PyArray_FLOAT),
                                             1, 3, NPY_CARRAY, NULL);
    boost::python::expect_non_null(input_block);

    int num_dim = ((PyArrayObject*) input_block)->nd;
    //npy_intp* d_strides = ((PyArrayObject*) input_block)->strides;

    npy_intp* n_dims = PyArray_DIMS(input_block);
    PyObject* output_block = PyArray_SimpleNew(num_dim, n_dims, PyArray_FLOAT);

    long obj_size = 1;
    for (int i=0; i<num_dim; i++){ obj_size = obj_size * n_dims[i]; } 

    float* in_data  = (float*) PyArray_DATA(input_block);
    float* out_data = (float*) PyArray_DATA(output_block);

    switch (approx_level) {
        case 3:
            for (int i=0; i<obj_size; i++){out_data[i] = fastexp3(in_data[i]);}
            break;
        case 4:
            for (int i=0; i<obj_size; i++){out_data[i] = fastexp4(in_data[i]);}
            break;
        case 5:
            for (int i=0; i<obj_size; i++){out_data[i] = fastexp5(in_data[i]);}
            break;
        case 6:
            for (int i=0; i<obj_size; i++){out_data[i] = fastexp6(in_data[i]);}
            break;
        case 7:
            for (int i=0; i<obj_size; i++){out_data[i] = fastexp7(in_data[i]);}
            break;
        case 8:
            for (int i=0; i<obj_size; i++){out_data[i] = fastexp8(in_data[i]);}
            break;
        case 9:
            for (int i=0; i<obj_size; i++){out_data[i] = fastexp9(in_data[i]);}
            break;
        default:
            std::cout << "Invalid approximation level.  Select [3-9]" << std::endl;

    }

    boost::python::handle<> out_mat(output_block);
    Py_DECREF(input_block);
    return boost::python::object(out_mat);
}

BOOST_PYTHON_MODULE(_vivid)
{
    NumPyWrapper::init();

    export_DeviceMatrix();
    export_PairwiseDistance();

    def("fast_exp", fast_exp);
    //class_<VideoReader>("FileVideo", init<const std::string& >() )
    //    .def("get_frame", &VideoReader::get_frame)
    //    .def("n_frames", &VideoReader::get_num_total_frames)
    //;
}

