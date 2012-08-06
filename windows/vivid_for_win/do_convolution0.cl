__kernel void  do_convolution0(
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
	
    const unsigned int out_x = 16 * get_group_id(1) + get_local_id(1);
    const unsigned int yt = 16 * get_group_id(0) + get_local_id(0);
    const unsigned int out_y = yt % output_y;
    const unsigned int out_t = yt / output_y;
	
    if (!((out_x < output_x) &&
          (out_y < output_y) &&
          (out_t < output_t))) {
        // Out of bounds -- bail
        return;
    }
	
    float sum = 0;
    for (unsigned int x=0; x < kernel_x; x++) {
        for (unsigned int y=0; y < kernel_y; y++) {
            for (unsigned int t=0; t < kernel_t; t++) {
                sum += video[(out_t + t) * video_pitch_tf+
				             (out_y + y) * video_pitch_yf + out_x + x]
				* constant_kernel
				[((((kernel_t - t - 1) * kernel_t)
				   + (kernel_y - y - 1)) * kernel_y)
				 + (kernel_x - x - 1)];
            }
        }
    }
     output[(out_t) * output_pitch_tf+
				             (out_y) * output_pitch_yf + out_x] = sum;
 
}

