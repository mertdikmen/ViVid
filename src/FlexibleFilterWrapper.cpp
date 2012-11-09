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

int update_filter_bank_cl(object& filterbank_array){
    NumPyMatrix3D arr(filterbank_array);
    //Here turn it into a float array and pass it to the FlexibleFilter.cpp
    int data_size = arr.dim_t() * arr.dim_x() * arr.dim_y();
    set_filter_bank_cl(arr.data(), data_size);

    return 0;
}

object cosine_filter_c(object& frame, object& filter_bank)
{
    NumPyMatrix frame_mat(frame);

    const int height = frame_mat.height();
    const int width = frame_mat.width();

    PyObject* filter_bank_parr = PyArray_FromAny(
        filter_bank.ptr(), 
        PyArray_DescrFromType(PyArray_FLOAT),
        1, 4, NPY_CARRAY, NULL);

    expect_non_null(filter_bank_parr);

    int num_dim = ((PyArrayObject*) filter_bank_parr)->nd;

    const int n_filters = ((PyArrayObject*) filter_bank_parr)->dimensions[0];
    const int filter_h = ((PyArrayObject*) filter_bank_parr)->dimensions[1];
    const int filter_w = ((PyArrayObject*) filter_bank_parr)->dimensions[2];
    
    const int filter_stride = filter_h * filter_w;

    float* fb_array = (float*) PyArray_DATA(filter_bank_parr);
    float* fr_data = (float*) frame_mat.data();

    //allocate output
    npy_intp dims[3] = {2, height, width};
    PyObject* arr = PyArray_SimpleNew(3, dims, PyArray_FLOAT);
    float* out_data = (float*)PyArray_DATA(arr);
    memset(out_data, 0, sizeof(float) * height * width * 2);

    //do convolution
    const int apron_y = filter_h / 2;
    const int apron_x = filter_w / 2;

    const int filter_size = filter_h * filter_w;

    const int filter_bank_size = filter_size * n_filters;

	int *pixel_offsets=(int*) malloc(sizeof(int)*filter_size);

    int oi = 0;
    for (int ii=-apron_y; ii<=apron_y; ii++){
        for (int jj=-apron_y; jj<=apron_y; jj++){
            pixel_offsets[oi] = ii * width + jj;
            oi++;
        }
    }

    double tic = omp_get_wtime();

    int n_threads = omp_get_num_procs();
    int valid_height = height - 2 * apron_y;
    int height_step = valid_height / n_threads + 1;

    #pragma omp parallel for
    for (int tid=0; tid<n_threads; tid++){
        int start_y = apron_y + tid * height_step;
        int end_y = min(start_y + height_step, height - apron_y);
    
    for (int i=start_y; i<end_y; i++){
        float* fr_ptr = fr_data + i * width + apron_x;
        float* ass_out = out_data + i * width + apron_x;
        float* wgt_out = ass_out + height * width;

        float *image_cache=(float*) malloc(sizeof(float)*filter_size);
        for (int j=apron_x; j<(width - apron_x); j++){

            for (int ii=0; ii< filter_size; ii++){
                image_cache[ii] = fr_ptr[pixel_offsets[ii]];
            } 

            float max_sim = -1e6;
            float best_ind = -1.0f;

            int fi=0;
            int filter_ind = 0;
            float temp_sum;
            while (fi<filter_bank_size)
            {
                temp_sum = 0.0f;

                for (int ii=0; ii < filter_size; ii++){
                    temp_sum += fb_array[fi++] * image_cache[ii];
                }

                temp_sum = fabs(temp_sum);

                if (temp_sum > max_sim){
                    max_sim = temp_sum;
                    best_ind = filter_ind;
                }

                filter_ind++;
            }
            *ass_out = best_ind;
            *wgt_out = max_sim;

            fr_ptr++;
            ass_out++;
            wgt_out++;
        }
    }
    }

    double toc = omp_get_wtime();

    //std::cout << "FF C time: " << toc - tic << std::endl;
    printf("FF C time %.8f\n", toc - tic);

    Py_DECREF(filter_bank_parr);
    handle<> temp_out(arr);
    return boost::python::object(temp_out);
}

/* Batch processing fuctions */
DeviceMatrixCL3D::Ptr filter_frame_cl_3_batch(const boost::python::object& npy_array,
        const int dim_t, const int nchannels, const int optype)
{
    PyObject* contig
        = PyArray_FromAny(
                npy_array.ptr(), PyArray_DescrFromType(PyArray_FLOAT),
                3, 3, NPY_CARRAY, NULL);
    handle<> temp(contig);
    object arr(temp);

    const int d0 = PyArray_DIM(contig, 0);
    const int d1 = PyArray_DIM(contig, 1);
    const int d2 = PyArray_DIM(contig, 2);

    //std::cout << "d0: " << d0 << ", d1: " << d1 << ", d2: " << d2 << std::endl;
    //TheContext * contex = new TheContext(1);
    DeviceMatrixCL::Ptr frame = makeDeviceMatrixCL(d1, d2);
    //DeviceMatrixCL_copyToDevice(frame

    float* data = (float*)PyArray_DATA(contig);

    //stride length for jumping frames
    int frame_stride = PyArray_STRIDE(contig, 0);

    //Create the output array
    DeviceMatrixCL3D::Ptr out = makeDeviceMatrixCL3D(2,d1,d2);

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
                               const int optype) >
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
    def ("_update_filter_bank_cl", update_filter_bank_cl);

    def("cosine_filter_c", cosine_filter_c);
}
