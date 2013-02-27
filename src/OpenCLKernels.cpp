#include "OpenCLKernels.hpp"
#include "cl_exceptions.hpp"

ViVidCLKernels::ViVidCLKernels(cl_context context, vivid::DeviceType device_type)
{
	memset(kernel_ready, 0, sizeof(bool) * NUM_MAX_KERNELS);

	if (device_type == vivid::DEVICE_GPU)
		createKernel(context, "pairwiseDistanceKernel","../../../src/E_PairwiseDistance.cl",0);
	else if (device_type == vivid::DEVICE_CPU)
		createKernel(context, "pairwiseDistanceKernel","../../../src/CPU_PairwiseDistance.cl",0);

	createKernel(context, "argminKernel","../../../src/argminKernel.cl",1);
	createKernel(context, "argmaxKernel","../../../src/argmaxKernel.cl",2);
	createKernel(context, "minKernel","../../../src/minKernel.cl",3);
	createKernel(context, "maxKernel","../../../src/maxKernel.cl",4);

	if (device_type == vivid::DEVICE_GPU)
		createKernel(context, "blockwise_distance_kernel","../../../src/E_blockwise_distance_kernel.cl",5);
	if (device_type == vivid::DEVICE_CPU)
		createKernel(context, "blockwise_distance_kernel","../../../src/CPU_blockwise_distance_kernel.cl",5);

	createKernel(context, "blockwise_filter_kernel","../../../src/blockwise_filter_kernel.cl",6);
	createKernel(context, "cell_histogram_kernel","../../../src/cell_histogram_kernel.cl",7);
	createKernel(context, "cellHistogramKernel1","../../../src/cellHistogramKernel1.cl",8);
	createKernel(context, "cellHistogramKernel2","../../../src/cellHistogramKernel2.cl",9);
	createKernel(context, "cellHistogramKernel3","../../../src/cellHistogramKernel3.cl",10);
}

void ViVidCLKernels::createKernel(cl_context context, const char* kernel, const char* path, int indice)
{
		char full_path[256];
		sprintf(full_path, "%s", path);

		char *program_source = load_program_source(full_path);
		if (program_source == NULL) {
			printf("Error: Failed to read the OpenCL kernel: %s\n",path);
			exit(-1);
		}
		cl_int err;

		program_list[indice] = clCreateProgramWithSource(context, 1, (const char **) &program_source, NULL, &err);
		CHECK_CL_ERROR(err);
		
		// Build the program executable
		const char * options = "-cl-fast-relaxed-math";
		err = clBuildProgram(program_list[indice], 0, NULL, options, NULL, NULL);
		if (err != CL_SUCCESS) {
			size_t len;
			char buffer[10000];

			printf("Error: Failed to build program executable for device %d kernel: (%s)!\n",err,kernel);
			cl_int get_err=clGetProgramBuildInfo(program_list[indice], cdDevice_K, CL_PROGRAM_BUILD_LOG, sizeof (buffer), buffer, &len);
			printf("%d %s\n", get_err, buffer);

		}

		kernel_list[indice] = clCreateKernel(program_list[indice], kernel, &err);
		CHECK_CL_ERROR(err);
}

//theKernels *  MyKernels::My_Kernels=NULL;
//theKernels *  MyKernels::My_Kernels_TMP=NULL;
//
//MyKernels::MyKernels(vivid::ContexOpenCl* context){
//    if (My_Kernels==NULL){
//        My_Kernels = new theKernels(context);
//    }
//}
//
//MyKernels::MyKernels(cl_context Context1,
//        cl_device_id cdDevice1,int cpu){
//    //	printf("Te Context()");
//    if (My_Kernels==NULL){
//        //	printf("Is NULL");
//        My_Kernels = new theKernels(Context1,cdDevice1);
//	}else{
//		if(My_Kernels_TMP==NULL){
//			My_Kernels_TMP = My_Kernels;
//			My_Kernels = new theKernels(Context1,cdDevice1);
//		}else{
//			theKernels *  tmp = My_Kernels_TMP;
//			My_Kernels_TMP = My_Kernels;
//			My_Kernels=tmp;
//		}
//	}
//}
//
//theKernels *  MyKernels_CPU::My_Kernels=NULL;
//
//MyKernels_CPU::MyKernels_CPU(cl_context GPUContext1,
//        cl_device_id cdDevice1){
//    //	printf("Te Context()");
//    if (My_Kernels==NULL){
//        //	printf("Is NULL");
//        My_Kernels = new theKernels(GPUContext1,cdDevice1);
//    }
//}

