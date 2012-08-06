struct float2 {
   float x, y;
};
__kernel void  do_convolution_complex_t0(
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
	float scale_val
	)
{

	const int video_pitch_yf = video_pitch_y / sizeof(float);
	const int video_pitch_tf = video_pitch_t / sizeof(float);
	
	const int kernel_pitch_yf = kernel_pitch_y / sizeof(float);
	const int kernel_pitch_tf = kernel_pitch_t / sizeof(float);
	
	const int output_pitch_yf = output_pitch_y / sizeof(float);
	const int output_pitch_tf = output_pitch_t / sizeof(float);
	
    
	
		
    const unsigned int x = get_local_size(1) * get_group_id(1) + get_local_id(1);
    const unsigned int y = get_local_size(0) * get_group_id(0) + get_local_id(0);
    if ((x < output_x/2) && (y < output_y)) {
        for (int t=0; t < output_t; t++) {
            float2 sum;
			sum.x = 0;
			sum.y = 0;
            for (int s=0; s < kernel_t; s++) {
			
			
			
               /* increment(sum,
						  complex_multiply(*get2(video, t+s, y, x),
										   *get2(_kernel, kernel_t-s-1, y,x )));*/
				float2 a = *((float2*)(&video[(t+s)*video_pitch_tf
			+y*video_pitch_yf
			+2*x ]));
			
			float2 b = *((float2*)(&_kernel[(kernel_t-s-1)*kernel_pitch_tf
			+y*kernel_pitch_yf
			+2*x ]));
			
				float2 other;
				other.x=a.x * b.x - a.y * b.y;
				other.y=a.x * b.y + a.y * b.x;
														   
				sum.x += other.x;
				sum.y += other.y;
            }
            //scale(sum, scale_val);
			sum.x *= scale_val;
			sum.y *= scale_val;
			*((float2*)(&output[t*output_pitch_tf
			+y*output_pitch_yf
			+2*x ])) = sum;
        }
    }

}
