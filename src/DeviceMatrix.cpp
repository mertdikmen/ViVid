#include "DeviceMatrix.hpp"
#include "NumPyWrapper.hpp"
#include "exceptions.hpp"

#include <cuda_runtime.h>

#include <iostream>

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
  mat->pitch_t = dim_y*pitch;

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
