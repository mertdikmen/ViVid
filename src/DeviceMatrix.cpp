#include <iostream>

#include <cuda_runtime.h>

#include <CL/opencl.h>
#include <CL/cl.h>

#include "DeviceMatrix.hpp"
#include "NumPyWrapper.hpp"
#include "exceptions.hpp"

static void deleteDeviceMatrix(DeviceMatrix* mat)
{
    //printf("cudaFree: %p\n", mat->data);
    //
    CUDA_CALL(cudaFree(mat->data));
    delete mat;
}

boost::shared_ptr<DeviceMatrix> makeDeviceMatrix(size_t height,
        size_t width){
    DeviceMatrix* mat = new DeviceMatrix();
    mat->width = width;
    mat->height = height;
    CUDA_CALL
        (cudaMallocPitch((void**)&mat->data, &mat->pitch,
                         mat->width * sizeof(float),
                         mat->height));

    // I can't imagine getting a pitch that's not a multiple of a float
    assert(mat->pitch % sizeof(float) == 0);
    // We want to express everything in floats
    mat->pitch /= sizeof(float);

    //printf("cudaMalloc: %p\n", mat->data);

    return boost::shared_ptr<DeviceMatrix>(mat, deleteDeviceMatrix);
}

static void deleteDeviceMatrixCL(DeviceMatrixCL* mat)
{
    cl_int err = clReleaseMemObject(mat->dataMatrix);
    if (err)
    {
        printf("Error releasing DeviceMatrixCL object (CL Error: %d", err);
    }

    delete mat;
}

boost::shared_ptr<DeviceMatrixCL> makeDeviceMatrixCL(size_t height, size_t width, cl_mem_flags FLAG)
{
    DeviceMatrixCL* mat = new DeviceMatrixCL();
    mat->width = width;
    mat->height = height;

    TheContext * tc = new TheContext();

    cl_context GPUContext = tc->getMyContext()->getContextCL();
    cl_device_id cdDevice = tc->getMyContext()->getDeviceCL();

    /*The optimal pitch is computed by (1) getting the base address alignment
    preference for your card (CL_DEVICE_MEM_BASE_ADDR_ALIGN property with
    clGetDeviceInfo: note that the returned value is in bits, so you have
    to divide by 8 to get it in bytes);*/

    
	//unsigned long long int loc_size;
	//clGetDeviceInfo(cdDevice, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(loc_size), &loc_size, NULL);
	//printf("device local memory size: %lldd bytes\n", loc_size);
	int buffer;
    cl_int prueba = clGetDeviceInfo(cdDevice, CL_DEVICE_MEM_BASE_ADDR_ALIGN , sizeof(buffer), &buffer, NULL);
    buffer /= 8;
	
	
    int naturalPitch = sizeof(float) * mat->width;

    /*let's call this base (2) find the largest multiple of base
    that is no less than your natural
    data pitch (sizeof(type) times number of columns);*/

    int devicepitch = ceil(float(naturalPitch)/buffer) * buffer;

    printf("Pitch: %d, DevicePitch: %d, Buffer: %d\n", naturalPitch, devicepitch, buffer);

    mat->pitch = devicepitch;

    //You then allocate pitch times number of rows bytes, and pass the pitch information to kernels.

	//std::cout << height << "\t" << devicepitch << std::endl;

    const int mem_size = (mat->height+16) * mat->pitch;
	
    //std::cout << "Mem size: " << mem_size << std::endl;

    int err;

    // mat->dataMatrix = clCreateBuffer(GPUContext, CL_MEM_READ_WRITE, mem_size, NULL, &err);
	mat->dataMatrix = clCreateBuffer(GPUContext, FLAG, mem_size, NULL, &err);
    if(err!=0)
    {
        printf("Error Code create buffer: %d\n",err);
    }

    return boost::shared_ptr<DeviceMatrixCL>(mat, deleteDeviceMatrixCL);
}

void DeviceMatrixCL_copyToDevice(DeviceMatrixCL& self, const float* data)
{
    const int mem_size = self.height * self.pitch;
    TheContext * tc = new TheContext();

    size_t buffer_origin[3] = {0,0,0};
    size_t host_origin[3] = {0,0,0};	
    size_t region[3] = {
        self.width * sizeof(float),
        self.height,
        1};	
	
    int err = clEnqueueWriteBufferRect(
            tc->getMyContext()->cqCommandQueue,
		    self.dataMatrix, CL_TRUE,
            buffer_origin, host_origin, region,
            self.pitch, 0,
            sizeof(float) * self.width, 0,
            data, 0, NULL, NULL);
	
    if (err != 0){
        std::cout << "Error in copyToDevice (CODE: " << err << ")" << std::endl;
    }
}

void DeviceMatrixCL_copyFromDevice(const DeviceMatrixCL& self, float* dst)
{
	if ((self.width > 0) && (self.height > 0)) {
		const int mem_size = self.height * self.pitch;

		TheContext * tc = new TheContext();

		size_t buffer_origin[3] = {0,0,0};
		size_t host_origin[3] = {0,0,0};	
		size_t region[3] = {self.width * sizeof(float),
			self.height,
			1};	

		cl_int err =
			clEnqueueReadBufferRect(
			tc->getMyContext()->cqCommandQueue,
			self.dataMatrix, CL_TRUE,
			buffer_origin, host_origin, region,
			self.pitch, 0,
			self.width * sizeof(float), 0,
			dst,
			0, NULL, NULL);

		if (err != 0){
			std::cout << "Error in copyFromDevice (CODE: " << err << ")" << std::endl;
		}
	}
}

static void deleteDeviceMatrix3D(DeviceMatrix3D* mat)
{
    //printf("cudaFree: %p\n", mat->data);
    CUDA_CALL(cudaFree(mat->data));
    delete mat;
}

DeviceMatrix3D::Ptr makeDeviceMatrix3D(size_t dim_t, size_t dim_y, 
        size_t dim_x){
    DeviceMatrix3D* mat = new DeviceMatrix3D();
    mat->dim_x = dim_x;
    mat->dim_y = dim_y;
    mat->dim_t = dim_t;
    size_t pitch;
    CUDA_CALL
        (cudaMallocPitch((void**)&mat->data, &pitch,
                         dim_x * sizeof(float),
                         dim_y * dim_t));
    // I can't imagine getting a pitch that's not a multiple of a float
    assert(pitch % sizeof(float) == 0);
    // We want to express everything in floats
    pitch /= sizeof(float);

    mat->pitch_y = pitch;
    mat->pitch_t = dim_y*mat->pitch_y;

    //printf("cudaMalloc: %p\n", mat->data);

    return DeviceMatrix3D::Ptr(mat, deleteDeviceMatrix3D);
}

/**
 * This function is useful for generating matrices for use with CUFFT.
 */
DeviceMatrix3D::Ptr makeDeviceMatrix3DPacked(size_t dim_t, size_t dim_y,
        size_t dim_x)
{
    DeviceMatrix3D* mat = new DeviceMatrix3D();
    mat->dim_x = dim_x;
    mat->dim_y = dim_y;
    mat->dim_t = dim_t;
    mat->pitch_y = dim_x;
    mat->pitch_t = dim_y*mat->pitch_y;

    CUDA_CALL
        (cudaMalloc((void**)&mat->data, dim_x * dim_y * dim_t * sizeof(float)));

    return DeviceMatrix3D::Ptr(mat, deleteDeviceMatrix3D);

}

void DeviceMatrix::zero(){
    CUDA_CALL(
            cudaMemset(data, 0, height * width * sizeof(float));
            );
}

void DeviceMatrix3D::zero(){
    //CUDA_CALL
    //   (cudaMemset(data, 2.0f, dim_x * dim_y * dim_t * sizeof(float)));
    //CUDA_CALL
    //(
    //  cudaMemset2D(data + 2, pitch_y * sizeof(float), 5.0f, dim_x * sizeof(float), dim_y * dim_t )
    //);
    CUDA_CALL
        (
         cudaMemset(data, 0, dim_t * pitch_t * sizeof(float));
        );
}


    DeviceMatrix3D::Ptr
cropDeviceMatrix3D(const DeviceMatrix3D::Ptr self,
        size_t new_dim_t, size_t new_dim_y, size_t new_dim_x)
{
    boost::shared_ptr<DeviceMatrix3DView> retval(new DeviceMatrix3DView());
    retval->parent = self;
    retval->dim_x = new_dim_x;
    retval->dim_y = new_dim_y;
    retval->dim_t = new_dim_t;
    retval->pitch_y = self->pitch_y;
    retval->pitch_t = self->pitch_t;
    retval->data = self->data;

    return retval;
}


static void deleteMCudaMatrix3D(DeviceMatrix3D* mat)
{
    delete [] mat->data;
    delete mat;
}

MCudaMatrix3D::Ptr makeMCudaMatrix3D(size_t dim_t, size_t dim_y,
        size_t dim_x)
{
    MCudaMatrix3D* mat = new MCudaMatrix3D();
    mat->dim_x = dim_x;
    mat->dim_y = dim_y;
    mat->dim_t = dim_t;
    mat->pitch_y = dim_x;
    mat->pitch_t = dim_y*mat->pitch_y;

    mat->data = new float[dim_x * dim_y * dim_t];

    return MCudaMatrix3D::Ptr(mat, deleteMCudaMatrix3D);
}





/**
 
 OpenCL 3d MATRIX
 
 **/

static void deleteDeviceMatrixCL3D(DeviceMatrixCL3D* mat)
{
    //printf("cudaFree: %p\n", mat->data);
	//    CUDA_CALL(cudaFree(mat->data));
	//  delete mat;
}


DeviceMatrixCL3D::Ptr makeDeviceMatrixCL3D(size_t dim_t, size_t dim_y, 
									   size_t dim_x){
    DeviceMatrixCL3D* mat = new DeviceMatrixCL3D();
    mat->dim_x = dim_x;
    mat->dim_y = dim_y;
    mat->dim_t = dim_t;
	printf("%d  x %d  x  %d\n",dim_x,dim_y,dim_t);
    size_t pitch;
	
    TheContext * tc = new TheContext();
	
    cl_context GPUContext = tc->getMyContext()->getContextCL();
    cl_device_id cdDevice = tc->getMyContext()->getDeviceCL();
	
    /*The optimal pitch is computed by (1) getting the base address alignment
	 preference for your card (CL_DEVICE_MEM_BASE_ADDR_ALIGN property with
	 clGetDeviceInfo: note that the returned value is in bits, so you have
	 to divide by 8 to get it in bytes);*/
	
    int buffer;
    cl_int prueba = clGetDeviceInfo(cdDevice, CL_DEVICE_MEM_BASE_ADDR_ALIGN , sizeof(buffer), &buffer, NULL);
    buffer /= 8;
	
    int naturalPitch = sizeof(float) * mat->dim_x;
	
    /*let's call this base (2) find the largest multiple of base
	 that is no less than your natural
	 data pitch (sizeof(type) times number of columns);*/
	
    int devicepitch = ceil(float(naturalPitch)/buffer) * buffer;
	
    printf("Pitch: %d, DevicePitch: %d, Buffer: %d\n", naturalPitch, devicepitch, buffer);
	
    mat->pitch_y = devicepitch;
	mat->pitch_t = dim_y*mat->pitch_y;
	
    //You then allocate pitch times number of rows bytes, and pass the pitch information to kernels.
	
    const int mem_size =  mat->dim_t*mat->pitch_t;
	
    std::cout << "Mem size: " << mem_size << std::endl;
	
    int err;
	
    mat->dataMatrix = clCreateBuffer(GPUContext, CL_MEM_READ_WRITE, mem_size, NULL, &err);
    if(err!=0)
    {
        printf("Error Code create buffer: %d\n",err);
    }
	
	
	
	
	
    return DeviceMatrixCL3D::Ptr(mat, deleteDeviceMatrixCL3D);
}





/**
 * This function is useful for generating matrices for use with CUFFT.
 */
DeviceMatrixCL3D::Ptr makeDeviceMatrixCL3DPacked(size_t dim_t, size_t dim_y,
											 size_t dim_x)
{
	DeviceMatrixCL3D* mat = new DeviceMatrixCL3D();
    mat->dim_x = dim_x;
    mat->dim_y = dim_y;
    mat->dim_t = dim_t;
    size_t pitch;
	
    TheContext * tc = new TheContext();
	
    cl_context GPUContext = tc->getMyContext()->getContextCL();
    cl_device_id cdDevice = tc->getMyContext()->getDeviceCL();
	
    	

	
    mat->pitch_y = dim_x;
	mat->pitch_t = dim_y*mat->pitch_y;
	
    
	
    const int mem_size = mat->dim_t * mat->pitch_t;
	
   	
    int err;
	
    mat->dataMatrix = clCreateBuffer(GPUContext, CL_MEM_READ_WRITE, mem_size, NULL, &err);
    if(err!=0)
    {
        printf("Error Code create buffer: %d\n",err);
    }
	
	
	
	
	
    return DeviceMatrixCL3D::Ptr(mat, deleteDeviceMatrixCL3D);	
}

void DeviceMatrixCL::zero(){
  //  CUDA_CALL(
//			  cudaMemset(data, 0, height * width * sizeof(float));
//			  );
}

void DeviceMatrixCL3D::zero(){
    //CUDA_CALL
    //   (cudaMemset(data, 2.0f, dim_x * dim_y * dim_t * sizeof(float)));
    //CUDA_CALL
    //(
    //  cudaMemset2D(data + 2, pitch_y * sizeof(float), 5.0f, dim_x * sizeof(float), dim_y * dim_t )
    //);
 /*   CUDA_CALL
	(
	 cudaMemset(data, 0, dim_t * pitch_t * sizeof(float));
	 );*/
}


DeviceMatrixCL3D::Ptr
cropDeviceMatrixCL3D(const DeviceMatrixCL3D::Ptr self,
				   size_t new_dim_t, size_t new_dim_y, size_t new_dim_x)
{
    boost::shared_ptr<DeviceMatrixCL3DView> retval(new DeviceMatrixCL3DView());
    retval->parent = self;
    retval->dim_x = new_dim_x;
    retval->dim_y = new_dim_y;
    retval->dim_t = new_dim_t;
    retval->pitch_y = self->pitch_y;
    retval->pitch_t = self->pitch_t;
    retval->data = self->data;
	
    return retval;
}


static void deleteMCLMatrix3D(DeviceMatrixCL3D* mat)
{
    delete [] mat->data;
    delete mat;
}

MCLMatrix3D::Ptr makeMCLMatrix3D(size_t dim_t, size_t dim_y,
									 size_t dim_x)
{
    MCLMatrix3D* mat = new MCLMatrix3D();
    mat->dim_x = dim_x;
    mat->dim_y = dim_y;
    mat->dim_t = dim_t;
    mat->pitch_y = dim_x;
    mat->pitch_t = dim_y*mat->pitch_y;
	
    mat->data = new float[dim_x * dim_y * dim_t];
	
    return MCLMatrix3D::Ptr(mat, deleteMCLMatrix3D);
}
