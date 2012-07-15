#include "PairwiseDistanceWrapper.hpp"
#include "PairwiseDistance.hpp"
#include <boost/python.hpp>
#include "NumPyWrapper.hpp"
#include "omp.h"

using namespace boost::python;

object pwdist_c(object a, object b){
    NumPyMatrix matA(a);
    NumPyMatrix matB(b);

    const int awidth = matA.width();

    NumPyMatrix output(matA.height(),matB.height());

    const int owidth = output.width();

    // Because the compiler can't quite figure out that these never
    // change.
    const unsigned int aheight = matA.height();
    const unsigned int bheight = matB.height();

    float const* a_data = matA.data();
    float const* b_data = matB.data();
    float* const out_data = output.data();

//    omp_set_num_threads(4);
//    printf("Total threads = %d\n", omp_get_num_threads());

    double tic = omp_get_wtime();
    for (unsigned int i=0; i<aheight; i++){
        #pragma omp parallel for num_threads(4)
        for (unsigned int j=0; j<bheight; j++){
            float sum = 0.0;

            for (int k = 0; k < awidth; k++){
                float dif = (a_data[i*awidth+k] - b_data[j*awidth+k]);
                sum+=dif*dif;
            }
            out_data[i*owidth+j]=sum;
        }
    }
    double toc = omp_get_wtime();

    //printf("CPU time: %.6f\n", (toc - tic) * 1e6 );

    return output.array;
};

object pwcityblock_c(object a, object b){
    NumPyMatrix matA(a);
    NumPyMatrix matB(b);

    const int awidth = matA.width();

    NumPyMatrix output(matA.height(),matB.height());

    const int owidth = output.width();

    // Because the compiler can't quite figure out that these never
    // change.
    const unsigned int aheight = matA.height();
    const unsigned int bheight = matB.height();

    float const* a_data = matA.data();
    float const* b_data = matB.data();
    float* const out_data = output.data();

    for (unsigned int i=0; i<aheight; i++){
        for (unsigned int j=0; j<bheight; j++){
            float sum = 0.0;

            for (int k = 0; k < awidth; k++){
                sum+=fabs(a_data[i*awidth+k] - b_data[j*awidth+k]);
            }
            out_data[i*owidth+j]=sum;
        }
    }
    return output.array;
};

object pwchisq_c(object a, object b){
    NumPyMatrix matA(a);
    NumPyMatrix matB(b);

    const int awidth = matA.width();

    NumPyMatrix output(matA.height(),matB.height());

    const int owidth = output.width();

    // Because the compiler can't quite figure out that these never
    // change.
    const unsigned int aheight = matA.height();
    const unsigned int bheight = matB.height();

    float const* a_data = matA.data();
    float const* b_data = matB.data();
    float* const out_data = output.data();

    for (unsigned int i=0; i<aheight; i++){
        for (unsigned int j=0; j<bheight; j++){
            float chisq_val = 0.0;

            for (int k = 0; k < awidth; k++){
                float dif = (a_data[i*awidth+k] - b_data[j*awidth+k]);
                float sum = (a_data[i*awidth+k] + b_data[j*awidth+k]);
                chisq_val+=dif*dif/sum;
            }
            out_data[i*owidth+j]=chisq_val;
        }
    }
    return output.array;
};

object pwdot_c(object a, object b){
    NumPyMatrix matA(a);
    NumPyMatrix matB(b);

    const int awidth = matA.width();

    NumPyMatrix output(matA.height(),matB.height());

    const int owidth = output.width();

    // Because the compiler can't quite figure out that these never
    // change.
    const unsigned int aheight = matA.height();
    const unsigned int bheight = matB.height();

    float const* a_data = matA.data();
    float const* b_data = matB.data();
    float* const out_data = output.data();

    for (unsigned int i=0; i<aheight; i++){
        for (unsigned int j=0; j<bheight; j++){
            float dotpr_val = 0.0;

            for (int k = 0; k < awidth; k++){
                dotpr_val+= (a_data[i*awidth+k] * b_data[j*awidth+k]);
            }
            out_data[i*owidth+j]=dotpr_val;
        }
    }
    return output.array;
};

void export_PairwiseDistance()
{
    def<DeviceMatrix::Ptr (const DeviceMatrix::Ptr&, const DeviceMatrix::Ptr& ) >
        ("pwchisq_cuda",pwchisq_cuda);

    def<DeviceMatrix::Ptr (const DeviceMatrix::Ptr&, const DeviceMatrix::Ptr& ) >
        ("pwdist_cuda",pwdist_cuda);
	
    def<DeviceMatrix::Ptr (const DeviceMatrix::Ptr&, const DeviceMatrix::Ptr& ) >
        ("pwcityblock_cuda",pwcityblock_cuda);

    def<DeviceMatrix::Ptr (const DeviceMatrix::Ptr&, const DeviceMatrix::Ptr& ) >
        ("pwdot_cuda",pwdot_cuda);

    def<DeviceMatrix::Ptr (const DeviceMatrix::Ptr&, const DeviceMatrix::Ptr& ) >
        ("pwabsdot_cuda",pwabsdot_cuda);
	
	def<DeviceMatrixCL::Ptr (const DeviceMatrixCL::Ptr&, const DeviceMatrixCL::Ptr& ) >
	("pwchisq_cl",pwchisq_cl);
	
	def<DeviceMatrixCL::Ptr (const DeviceMatrixCL::Ptr&, const DeviceMatrixCL::Ptr& ) >
	("pwdist_cl",pwdist_cl);
	
    def<DeviceMatrixCL::Ptr (const DeviceMatrixCL::Ptr&, const DeviceMatrixCL::Ptr& ) >
	("pwcityblock_cl",pwcityblock_cl);
	//
    def<DeviceMatrixCL::Ptr (const DeviceMatrixCL::Ptr&, const DeviceMatrixCL::Ptr& ) >
	("pwdot_cl",pwdot_cl);
	//
   def<DeviceMatrixCL::Ptr (const DeviceMatrixCL::Ptr&, const DeviceMatrixCL::Ptr& ) >
	("pwabsdot_cl",pwabsdot_cl);
	
    def ("pwdist_c", pwdist_c);
    def ("pwcityblock_c", pwcityblock_c);
    def ("pwchisq_c", pwchisq_c);
    def ("pwdot_c", pwdot_c);

    def<DeviceMatrix::Ptr (const DeviceMatrix::Ptr&) >
        ("argmin_cuda", argmin_cuda);
    def<DeviceMatrix::Ptr (const DeviceMatrix::Ptr&) >
        ("argmax_cuda", argmax_cuda);

    def<DeviceMatrix::Ptr (const DeviceMatrix::Ptr&) >
        ("min_cuda", min_cuda);
    def<DeviceMatrix::Ptr (const DeviceMatrix::Ptr&) >
        ("max_cuda", max_cuda);
	
	
	def<DeviceMatrixCL::Ptr (const DeviceMatrixCL::Ptr&) >
	("argmin_cl", argmin_cl);
   def<DeviceMatrixCL::Ptr (const DeviceMatrixCL::Ptr&) >
	("argmax_cl", argmax_cl);
	
    def<DeviceMatrixCL::Ptr (const DeviceMatrixCL::Ptr&) >
	("min_cl", min_cl);
    def<DeviceMatrixCL::Ptr (const DeviceMatrixCL::Ptr&) >
	("max_cl", max_cl);
	
	
	
	
}
