#include "FlexibleFilterWrapper.hpp"
#include "FlexibleFilter.hpp"
#include "DeviceMatrixWrapper.hpp"

#include <boost/python.hpp>
#include "NumPyWrapper.hpp"

#ifdef _WIN32
#include "omp.h"
#else
#include "omp_unix.h"
#endif

#define PY_ARRAY_UNIQUE_SYMBOL tb
#define NO_IMPORT
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

using namespace boost::python;
//#include "cpucounters.h"

int update_filter_bank_cuda(object& filterbank_array){
    NumPyMatrix3D arr(filterbank_array);
    //Here turn it into a float array and pass it to the FlexibleFilter.cpp
    int data_size = arr.dim_t() * arr.dim_x() * arr.dim_y();
    set_filter_bank_cuda(arr.data(), data_size);

    return 0;
}

int update_filter_bank_cl(object& filterbank_array, vivid::DeviceType device_type){
    NumPyMatrix3D arr(filterbank_array);
    //Here turn it into a float array and pass it to the FlexibleFilter.cpp
    int data_size = arr.dim_t() * arr.dim_x() * arr.dim_y();
    set_filter_bank_cl(arr.data(), data_size, device_type);

    return 0;
}

object cosine_filter_c(object& frame, object& filter_bank)
{
    NumPyMatrix frame_mat(frame);

    const int height = frame_mat.height();
    const int width = frame_mat.width();

    PyArrayObject* filter_bank_parr = reinterpret_cast<PyArrayObject*>(PyArray_FromAny(
        filter_bank.ptr(), 
        PyArray_DescrFromType(NPY_FLOAT),
        1, 4, NPY_ARRAY_CARRAY, NULL));

    expect_non_null(filter_bank_parr);

    int num_dim = PyArray_NDIM((PyArrayObject*) filter_bank_parr);

    npy_intp* dimensions = PyArray_DIMS(filter_bank_parr);

    const int n_filters = dimensions[0];
    const int filter_h = dimensions[1];
    const int filter_w = dimensions[2];
    
    const int filter_stride = filter_h * filter_w;

    float* fb_array = (float*) PyArray_DATA(filter_bank_parr);
    float* fr_data = (float*) frame_mat.data();

    //allocate output
    npy_intp dims[3] = {2, height, width};
    PyObject* arr = PyArray_SimpleNew(3, dims, NPY_FLOAT);
    float* out_data = (float*)PyArray_DATA((PyArrayObject*)arr);
    memset(out_data, 0, sizeof(float) * height * width * 2);

	cosine_filter(fr_data, fb_array, height, width, filter_h, filter_w, n_filters, out_data);

    //std::cout << "FF C time: " << toc - tic << std::endl;
    //printf("FF C time %.8f\n", toc - tic);

    Py_DECREF(filter_bank_parr);
    handle<> temp_out(arr);
    return boost::python::object(temp_out);
}

/* Batch processing fuctions */
DeviceMatrixCL3D::Ptr filter_frame_cl_3_batch(const boost::python::object& npy_array,
        const int dim_t, const int nchannels, const int optype, vivid::DeviceType device_type)
{
    PyArrayObject* contig
        = reinterpret_cast<PyArrayObject*>(PyArray_FromAny(
                npy_array.ptr(), PyArray_DescrFromType(NPY_FLOAT),
                3, 3, NPY_ARRAY_CARRAY, NULL));
    handle<> temp((PyObject*)contig);
    object arr(temp);

    const int d0 = PyArray_DIM(contig, 0);
    const int d1 = PyArray_DIM(contig, 1);
    const int d2 = PyArray_DIM(contig, 2);

    //std::cout << "d0: " << d0 << ", d1: " << d1 << ", d2: " << d2 << std::endl;
    //TheContext * contex = new TheContext(1);
    DeviceMatrixCL::Ptr frame = makeDeviceMatrixCL(d1, d2, device_type);
    //DeviceMatrixCL_copyToDevice(frame

    float* data = (float*)PyArray_DATA(contig);

    //stride length for jumping frames
    int frame_stride = PyArray_STRIDE(contig, 0);

    //Create the output array
    DeviceMatrixCL3D::Ptr out = makeDeviceMatrixCL3D(2,d1,d2, device_type);

//    /**
//      Measuring energy
//     **/
//    PCM * m = PCM::getInstance();
//    m->resetPMU();
//    // program counters, and on a failure just exit 
//    if (m->program() != PCM::Success) exit( 0);
//
//    SystemCounterState before_sstate = getSystemCounterState();
//    double *package_before=(double*)  malloc(sizeof(double));
//    double* pp0_before =(double*)  malloc(sizeof(double));
//    double* pp1_before =  (double*)  malloc(sizeof(double));
//
//    for (uint32 i = 0; i < m->getNumSockets(); ++i)
//        getSocketCounterState(i,package_before,pp0_before,pp1_before);
//
//    double pbefore= *package_before;
//    double pp0before= *pp0_before;
//    double pp1before = *pp1_before;

    for (int i = 0; i<d0; i++)
    {
        std::cout << "Processing frame#: " << i << std::endl;
        DeviceMatrixCL_copyToDevice(*frame, data);
        dist_filter2_d3_cl(frame.get(), dim_t, nchannels, out.get(), optype);
        data += frame_stride / sizeof(float);   
    }

//    for (uint32 i = 0; i < m->getNumSockets(); ++i)
//        getSocketCounterState(i,package_before,pp0_before,pp1_before);
//
//    double pafter= *package_before;
//    double pp0after= *pp0_before;
//    double pp1after = *pp1_before;
//
//    printf("\n");
//    printf("PK\tPP0\tPP1\n");
//    printf("%.2f\t%.2f\t%.2f\n",pafter-pbefore,pp0after-pp0before,pp1after-pp1before);
//    printf("\n");
//
//    m->resetPMU();

    return out;
}

void export_FlexibleFilter()
{
    /* For power testing purposes only */
    def<DeviceMatrixCL3D::Ptr (const object&,
                               const int dim_t, const int nchannels,
                               const int optype,
                               vivid::DeviceType device_type) >
        ("_filter_frame_cl_3_batch", filter_frame_cl_3_batch);

    /* End test code */


    def<DeviceMatrix3D::Ptr (const DeviceMatrix::Ptr&, 
                             const int dim_t, const int nchannels,
                             const int optype ) >
        ("_filter_frame_cuda_3", filter_frame_cuda_3);
	
	def<DeviceMatrixCL3D::Ptr (const DeviceMatrixCL::Ptr&, 
                             const int dim_t, const int nchannels,
                             const int optype ) >
	    ("_filter_frame_cl_3", filter_frame_cl_3);
	
    def<DeviceMatrix3D::Ptr (const DeviceMatrix::Ptr&, 
                             const int dim_t, const int nchannels,
                             const int optype ) >
        ("_filter_frame_cuda_5", filter_frame_cuda_5);
	
	def<DeviceMatrixCL3D::Ptr (const DeviceMatrixCL::Ptr&, 
                             const int dim_t, const int nchannels,
                             const int optype ) >
	    ("_filter_frame_cl_5", filter_frame_cl_5);

    def<DeviceMatrix3D::Ptr (const DeviceMatrix::Ptr&, 
                             const int dim_t, const int nchannels,
                             const int optype ) >
        ("_filter_frame_cuda_7", filter_frame_cuda_7);

	def<DeviceMatrixCL3D::Ptr (const DeviceMatrixCL::Ptr&, 
                             const int dim_t, const int nchannels,
                             const int optype ) >
	    ("_filter_frame_cl_7", filter_frame_cl_7);

    def<DeviceMatrix3D::Ptr (const DeviceMatrix::Ptr&, 
                             const int dim_t, const int dim_y, const int dim_x, const int nchannels,
                             const int optype ) >
        ("_filter_frame_cuda_noargmin", filter_frame_cuda_noargmin);

	def<DeviceMatrixCL3D::Ptr (const DeviceMatrixCL::Ptr&, 
                             const int dim_t, const int dim_y, const int dim_x, const int nchannels,
                             const int optype ) >
	    ("_filter_frame_cl_noargmin", filter_frame_cl_noargmin);
	
    def ("_update_filter_bank_cuda", update_filter_bank_cuda);
    def <int (boost::python::object& obj, vivid::DeviceType device_type)> ("_update_filter_bank_cl", update_filter_bank_cl);

    def("cosine_filter_c", cosine_filter_c);
}
