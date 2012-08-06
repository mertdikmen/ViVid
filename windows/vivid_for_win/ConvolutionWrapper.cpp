#include "ConvolutionWrapper.hpp"

#include "ConvolutionMCuda.hpp"

using namespace boost::python;

void export_Convolution()
{
  def<DeviceMatrix3D::Ptr (const DeviceMatrix3D::Ptr&,
                           const DeviceMatrix3D::Ptr&)>
    ("convolve3d", convolve3d);
  def("convolve3d_specific", convolve3d_specific);
  def("convolve3d_mcuda", convolve3d_mcuda);

  def("convolve_complex_t", convolve_complex_t);
  def("convolve_complex_t_specific", convolve_complex_t_specific);

  def("debug_convolution_algorithm_used", debug_convolution_algorithm_used);
	
	
	/** OPENCL**/
	
	def<DeviceMatrixCL3D::Ptr (const DeviceMatrixCL3D::Ptr&,
							 const DeviceMatrixCL3D::Ptr&)>
    ("convolve3d_cl", convolve3d_cl);
	def("convolve3d_specific_cl", convolve3d_specific_cl);
	
	def("convolve3d_m_cl", convolve3d_m_cl);
	def("convolve_complex_t_cl", convolve_complex_t_cl);
	def("convolve_complex_t_specific_cl", convolve_complex_t_specific_cl);
	
	def("debug_convolution_algorithm_used_cl", debug_convolution_algorithm_used_cl);

}
