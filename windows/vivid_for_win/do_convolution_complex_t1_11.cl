struct float2 {
   float x, y;
};
__kernel void  do_convolution_complex_t1_11(
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
	int KERN_SIZE = 11;
    
	
		
	__local float2 kern[11][16];
	
    const unsigned int offset_yx = 16 * get_group_id(1) + get_local_id(1);
    const unsigned int offset_t  = get_local_id(0);
	
    // Abuse the offsets because we have packed matrices.  We
    // essentially drop the problem back to 2D.
    kern[offset_t][get_local_id(1)] = *((float2*)(&_kernel[offset_t*kernel_pitch_tf
			+0*kernel_pitch_yf
			+2*offset_yx ]));
			
    __local float2 data[2][16];
	
    // Flag for ping-ponging
    int read_buffer = 0;
	
    // This variable indexes the kernel as we slide the image through.
    // The values here should range from 0:(KERN_SIZE-1)
    int kern_t = offset_t;
	
    // This tracks the partial sum
    float2 sum;
	sum.x = 0;
	sum.y = 0;
	
    // Prime the buffer
    data[read_buffer][get_local_id(1)] = *((float2*)(&video[0*video_pitch_tf
			+0*video_pitch_yf
			+2*offset_yx ]));
    // Carve a swath in t.
    for (unsigned int t=0; t < video_t; t++) {
        // Hoist the reading up higher.  We will read the value that
        // we want to use next.
        float2 load_value;
        if (get_local_id(0) == 0) {
            if ((t+1) < video_t) {
                load_value = *((float2*)(&video[(t+1)*video_pitch_tf
			+0*video_pitch_yf
			+2*offset_yx ]));

            }
        }
		
     /*   increment(sum, complex_multiply(data[read_buffer][get_local_id(1)],
                                        kern[KERN_SIZE - kern_t -1]
										[get_local_id(1)]));*/
										
			float2 a = data[read_buffer][get_local_id(1)];
			
			float2 b = kern[KERN_SIZE - kern_t -1][get_local_id(1)];
			
				float2 other;
				other.x=a.x * b.x - a.y * b.y;
				other.y=a.x * b.y + a.y * b.x;
														   
				sum.x += other.x;
				sum.y += other.y;
		
        kern_t++;
        if (kern_t == KERN_SIZE) {
            int out_t = t - KERN_SIZE + 1;
            if (0 <= out_t) {
                // Write out the solution
               // scale(sum, scale_val);
			sum.x *= scale_val;
			sum.y *= scale_val;
			*((float2*)(&output[out_t*output_pitch_tf
			+0*output_pitch_yf
			+2*offset_yx ])) = sum;
            }
            sum.x = 0;
			sum.y = 0;
            kern_t = 0;
        }
		
        if (get_local_id(0) == 0) {
            data[!read_buffer][get_local_id(1)] = load_value;
        }
		
        read_buffer = !read_buffer;
		
        // Prevent the next iteration of the loop from scribbling on
        // values we are trying to read
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }
}
