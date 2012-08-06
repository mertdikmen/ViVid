__kernel void  do_convolution1(
    __global float* video,
    const int  video_x,
    const int video_y, 
	const int video_t,    
	const int video_pitch_y,
	const int video_pitch_t,                                
	__global float* _kernel,
    const int  kernel_x,
    const int kernel_y, 
	const int kernel_t,    
	const int kernel_pitch_y,
	const int kernel_pitch_t, 
	__global float* output,
    const int  output_x,
    const int output_y, 
	const int output_t,    
	const int output_pitch_y,
	const int output_pitch_t,
	__global float* constant_kernel
	)
{

	const int video_pitch_yf = video_pitch_y / sizeof(float);
	const int video_pitch_tf = video_pitch_t / sizeof(float);
	
	const int kernel_pitch_yf = kernel_pitch_y / sizeof(float);
	const int kernel_pitch_tf = kernel_pitch_t / sizeof(float);
	
	const int output_pitch_yf = output_pitch_y / sizeof(float);
	const int output_pitch_tf = output_pitch_t / sizeof(float);
	
    float sum = 0;
    const unsigned int block_y = get_group_id(0) % output_y;
    const unsigned int block_t = get_group_id(0) / output_y;
	
    for (unsigned int t=0; t < kernel_t; t++) {
        sum +=video[(block_t + t) * video_pitch_tf+
				             (block_y + get_local_id(0)) * video_pitch_yf +
							  get_group_id(1) + get_local_id(1)] 
		* constant_kernel
		[((((kernel_t - t - 1) * kernel_t)
		   + (kernel_y - get_local_id(0) - 1)) * kernel_y)
		 + (kernel_x - get_local_id(1) - 1)];
    }
	
	const int linear_idx = get_local_id(0) * get_local_size(1) + get_local_id(1);
	// HACK: We can't have more than 512 threads...
	__local float buffer[512];
	buffer[linear_idx] = sum;
	
   barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
	
	unsigned int len = get_local_size(1) * get_local_size(0);
	while (len > 1) {
		unsigned int stride =  (len+1)/ 2;
		if (linear_idx + stride < len) {
			buffer[linear_idx] += buffer[linear_idx + stride];
		}
   barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
		len = stride;
	}
	
	if (linear_idx == 0) {
		output[(block_t) * output_pitch_tf+
				             (block_y) * output_pitch_yf + get_group_id(1)]
							  = buffer[0];
		
	}

 
}

