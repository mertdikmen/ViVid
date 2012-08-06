#include "NumPyWrapper.hpp"
#include "DeviceMatrixWrapper.hpp"
#include "PairwiseDistanceWrapper.hpp"
#include "FlexibleFilterWrapper.hpp"
#include "BlockHistogramWrapper.hpp"
#include "ConvolutionWrapper.hpp"
#include "OpenCLWrapper.hpp"
#include "fastexp.h"

#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#define PY_ARRAY_UNIQUE_SYMBOL tb
#include <numpy/arrayobject.h>


////////////////////////
// Faster operations
////////////////////////
boost::python::object fast_exp(boost::python::object& input_mat, 
                               const int approx_level)
{
    PyObject* input_block = PyArray_FromAny(
                                input_mat.ptr(), 
                                PyArray_DescrFromType(
                                    PyArray_FLOAT),
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

//class LBPComputer
//{
//    public:
//    LBPComputer(const int u, const int num_bits)
//    {
//        const int num_values = pow(2, num_bits);
//        m_lbp_map.resize(num_values);
//    }
//
//
//
//    private:
//        std::vector<int> m_lbp_map;
//}

std::vector<int> create_lbp_dictionary(
    const int u, const int num_bits)
{
    const int num_values = pow(2, num_bits);

    std::vector<int> lbp_map(num_values, 0);

    int number;
    int center_count = 1;
    for (int i=0; i<num_values; i++){
        int one_count = 0;
        number = i;

        //rotate_left 
        if (i >= (num_values / 2) ){ 
            number = ((number - num_values / 2) << 1) + 1;
        }
        else {
            number = number << 1;
        }

        //bitwise xor
        int xor_test = number ^ i;

        for (int j=0; j<num_bits;j++){
            if ((xor_test % 2) == 1){
                one_count++;
            }
            xor_test = xor_test >> 1;
        }
        if (one_count <= u){
            lbp_map[i] = center_count++;
        }
    }

    return lbp_map;
}

object compute_lbp_n8_r1(const object& input_mat, 
                         const std::vector<int> lbp_map)
{
    NumPyMatrix input(input_mat);

    const int height = input.height();
    const int width = input.width();

    const int imsize = height * width;

    npy_intp dims[2] = {height, width};

    PyObject* output = PyArray_SimpleNew(2, dims, PyArray_INT);
    int* out_data = (int*)PyArray_DATA(output);
    
    const float* input_data = (float*) input.data();

    memset(out_data, 0, sizeof(int) * imsize);

    const int offsets[8] = {-width - 1, -width, -width+1,
                            -1, 1,
                            width - 1, width, width+1};

    const int vals[8] = {1, 2, 4, 8, 16, 32, 64, 128};

    //set the values outside the margins to -1
    for (int i = 0; i < width; i++){
        out_data[i] = -1;
        out_data[imsize - i - 1] = -1;
    }

    for (int i = 0; i<imsize; i+= width){
        out_data[i] = -1;
        out_data[i + width -1] = -1;
    }


    for (int i = 1; i < height - 1; i++){
        const int row_offset = i * width;
        for (int j = 1; j < width - 1; j++){
            int pix_i = row_offset + j;
            int lbp_val = 0;
            for (int k = 0; k < 8; k++){
                lbp_val += 
                    (input_data[pix_i] < 
                     input_data[pix_i + offsets[k]]) ? vals[k] : 0;
            } 
            out_data[pix_i] = lbp_map[lbp_val];
        }
    }

    handle<> temp_out(output);
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

object add_cell_histograms(object& cell_histograms,
                           const int block_size_y, const int block_size_x,
                           const int block_step)
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

    assert(cells_y >= block_size_y);
    assert(cells_x >= block_size_x);

    const int blocks_y = (cells_y - block_size_y) / block_step + 1;
    const int blocks_x = (cells_x - block_size_x) / block_step + 1;

    npy_intp output_dims[3];
    output_dims[0] = blocks_y;
    output_dims[1] = blocks_x;
    output_dims[2] = hist_size;

    PyObject* pyo_output_blocks = PyArray_SimpleNew(3, output_dims, PyArray_FLOAT);

    float* cell_histograms_data = (float*) PyArray_DATA(pyo_cell_histograms);
    float* output_blocks_data = (float*) PyArray_DATA(pyo_output_blocks);

    memset(output_blocks_data, 0, sizeof(float) * blocks_y * blocks_x * hist_size);

    #pragma omp for
    for (int by=0; by<blocks_y; by++){
        const int cy = block_step * by;
        for (int bx=0; bx<blocks_x; bx++){
            const int cx = block_step * bx;

            float* dst_ptr = output_blocks_data +
                             (by * blocks_x + bx) * hist_size;

            for (int iy=0; iy<block_size_y; iy++){
                for (int ix=0; ix<block_size_x; ix++){

                    float* src_ptr = cell_histograms_data + 
                            ( (cy+iy) * cells_x + (cx+ix) ) * hist_size;

                    for (int hi=0; hi<hist_size; hi++){
                        dst_ptr[hi] += src_ptr[hi];
                    }
                }
            }
        }    
    }

    handle<> out_mat(pyo_output_blocks);

    Py_DECREF(pyo_cell_histograms);
    return boost::python::object(out_mat);
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
        const int cy = block_step * by;
        for (int bx=0; bx<blocks_x; bx++){
            const int cx = block_step * bx;

            for (int iy=0; iy<block_size_y; iy++){
                for (int ix=0; ix<block_size_x; ix++){

                    float* src_ptr = cell_histograms_data + 
                            ( (cy+iy) * cells_x + (cx+ix) ) * hist_size;
                    float* dst_ptr = output_blocks_data +
                            (by * blocks_x + bx) * hist_size * block_size_y *
                            block_size_x +
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
    //NumPyWrapper::init();

    import_array();

    export_DeviceMatrix();
    export_PairwiseDistance();
    export_FlexibleFilter();
    export_BlockHistogram();
    export_Convolution();
    export_OpenCL();

    class_< std::vector<int> >("std::vectorOfInt")
                 .def(vector_indexing_suite< std::vector<int>, true>());
    def("fast_exp", fast_exp);
    def("group_blocks", group_blocks); 
    def("group_cell_histograms", group_cell_histograms);
    def("add_cell_histograms", add_cell_histograms);
    def("create_lbp_dictionary", create_lbp_dictionary);
    def("compute_lbp_n8_r1", compute_lbp_n8_r1); 

    //class_<VideoReader>("FileVideo", init<const std::string& >() )
    //    .def("get_frame", &VideoReader::get_frame)
    //    .def("n_frames", &VideoReader::get_num_total_frames)
    //;
}

