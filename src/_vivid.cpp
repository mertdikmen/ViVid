#include "NumPyWrapper.hpp"
#include "DeviceMatrixWrapper.hpp"
#include "PairwiseDistanceWrapper.hpp"
#include "FlexibleFilterWrapper.hpp"
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

object cell_histogram_dense(object& input_mat, object& weight_mat, 
                             const int max_bin, const int cell_size, 
                             object& offset)
{
    NumPyArray offset_arr(offset);
    NumPyMatrix id_m(input_mat);
    NumPyMatrix wgts(weight_mat);

    int n_parts_y = (id_m.height() - 2 * offset_arr.data()[0] ) / cell_size;
    int n_parts_x = (id_m.width()  - 2 * offset_arr.data()[1] ) / cell_size;

    npy_intp dims[3] = {n_parts_y, n_parts_x, max_bin};
    
    PyObject* arr = PyArray_SimpleNew(3, dims, PyArray_FLOAT);
    float* out_data = (float*)PyArray_DATA(arr);
    
    memset(out_data, 0, sizeof(float) * n_parts_y * n_parts_x * max_bin);

    int start_i = offset_arr.data()[0];
    int start_j = offset_arr.data()[1];

    const int im_width = id_m.width();
    const int im_height = id_m.height();

    const float* id_data = (float*) id_m.data();
    const float* wt_data = (float*) wgts.data();

    #pragma omp for 
    for (int write_i=0; write_i<n_parts_y; write_i++){
        for (int write_j=0; write_j<n_parts_x; write_j++){
            int out_ind = (write_i*n_parts_x + write_j) * max_bin;

            int read_i = (start_i + (write_i * cell_size)) * im_width;
            for (int i=0; i<cell_size; i++){
                int read_j = start_j + write_j * cell_size ;

                for (int j=0; j<cell_size; j++){
                    int bin_ind = (int)id_data[read_i+read_j];
        
                    //will ignore the value if the input_mat value is outside the range
                    if ( (bin_ind >= 0) && (bin_ind < max_bin) ){
                        float weight = wt_data[read_i+read_j];
                        out_data[out_ind + bin_ind] += weight;
                    }

                    read_j ++;    
                }
                read_i += im_width;
            }
        }
    }
    
    handle<> temp_out(arr);
    return boost::python::object(temp_out);
}

object group_blocks(object& block_mat, object& grouping_inds, bool normalize_flag){

    PyObject* contig_block = PyArray_FromAny(block_mat.ptr(), PyArray_DescrFromType(PyArray_FLOAT),
                                       3, 3, NPY_CARRAY, NULL);

    PyObject* contig_group = PyArray_FromAny(grouping_inds.ptr(), PyArray_DescrFromType(PyArray_INT),
                                       2, 2, NPY_CARRAY, NULL);

    float* blocks_data = (float*)PyArray_DATA(contig_block);
    int* grouping_data = (int*)PyArray_DATA(contig_group);

    int n_samples = (int) PyArray_DIM(contig_block, 0);
    int n_cells = (int) PyArray_DIM(contig_block, 1);
    int n_dims = (int) PyArray_DIM(contig_block,2);

    int n_blocks = (int) PyArray_DIM(contig_group,0);
    int block_dim = (int) PyArray_DIM(contig_group,1);


    npy_intp dims[3] = {n_samples, n_blocks, n_dims};
    
    PyObject* arr = PyArray_SimpleNew(3, dims, PyArray_FLOAT);
    float* out_data = (float*)PyArray_DATA(arr);
    
    memset(out_data, 0, sizeof(float) * n_samples*n_blocks*n_dims);

    for (int si = 0; si<n_samples; si++){
        int sample_wi = si * (n_blocks * n_dims);
        int sample_ri = si * (n_cells * n_dims);

        for (int bi = 0; bi < n_blocks; bi++){
            int group_i = block_dim * bi;


            for (int ci =0; ci<block_dim; ci++){
                int cur_cell = grouping_data[group_i+ci];
                

                for (int k=0; k<n_dims; k++){
                    int cur_cell_i = cur_cell * n_dims;
    
                    out_data[sample_wi + bi * n_dims + k] += blocks_data[sample_ri + cur_cell_i+k];

                }
            }
        }
    }
    
    handle<> temp_out(arr);
    return boost::python::object(temp_out);

}

object group_cell_histograms(object& cell_histograms, 
                             const int block_size_y, const int block_size_x, /* in terms of number of cells */
                             const int block_step /* in terms of number of cells */)
{

    PyObject* pyo_cell_histograms = PyArray_FromAny(cell_histograms.ptr(), 
                                   PyArray_DescrFromType(PyArray_FLOAT), 
                                   3, 3, 
                                   NPY_CARRAY, NULL);

    boost::python::expect_non_null(pyo_cell_histograms);

    npy_intp* n_dims = PyArray_DIMS(pyo_cell_histograms);

    
    const int cells_y = n_dims[0]; 
    const int cells_x = n_dims[1];
    const int hist_size = n_dims[2];

    const int blocks_y = (cells_y - block_size_y) / block_step + 1;
    const int blocks_x = (cells_x - block_size_x) / block_step + 1;

    npy_intp output_dims[3];
    output_dims[0] = blocks_y;
    output_dims[1] = blocks_x;
    output_dims[2] = hist_size * block_size_y * block_size_x;

    PyObject* pyo_output_blocks = PyArray_SimpleNew(3, output_dims, PyArray_FLOAT);

    float* cell_histograms_data = (float*) PyArray_DATA(pyo_cell_histograms);
    float* output_blocks_data = (float*) PyArray_DATA(pyo_output_blocks);

    #pragma omp for
    for (int by=0; by<blocks_y; by++){
        for (int bx=0; bx<blocks_x; bx++){

            const int cy = block_step * by;
            const int cx = block_step * bx;

            for (int iy=0; iy<block_size_y; iy++){
                for (int ix=0; ix<block_size_x; ix++){

                    float* src_ptr = cell_histograms_data + ( (cy+iy) * cells_x + (cx+ix) ) * hist_size;
                    float* dst_ptr = output_blocks_data + (by * blocks_x + bx) * hist_size * block_size_y * block_size_x + 
                                    (iy * block_size_x + ix) * hist_size;

                    memcpy(dst_ptr, src_ptr, sizeof(float) * hist_size);

                }
            }
        }    
    }
    
    handle<> out_mat(pyo_output_blocks);
    
    Py_DECREF(pyo_cell_histograms);
    return boost::python::object(out_mat);
}


BOOST_PYTHON_MODULE(_vivid)
{
    NumPyWrapper::init();

    export_DeviceMatrix();
    export_PairwiseDistance();
    export_FlexibleFilter();

    import_array();

    def("fast_exp", fast_exp);
    def("cell_histogram_dense", cell_histogram_dense);
    def("group_blocks", group_blocks); 
    def("group_cell_histograms", group_cell_histograms);
   
    //class_<VideoReader>("FileVideo", init<const std::string& >() )
    //    .def("get_frame", &VideoReader::get_frame)
    //    .def("n_frames", &VideoReader::get_num_total_frames)
    //;
}

