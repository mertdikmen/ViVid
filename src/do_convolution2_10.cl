

__kernel void  do_convolution2_10(
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
	__global float* constant_kernel,
	int KERN_SIZE,
	int SWATH_SIZE
	)
{

	const int video_pitch_yf = video_pitch_y / sizeof(float);
	const int video_pitch_tf = video_pitch_t / sizeof(float);
	
	const int kernel_pitch_yf = kernel_pitch_y / sizeof(float);
	const int kernel_pitch_tf = kernel_pitch_t / sizeof(float);
	
	const int output_pitch_yf = output_pitch_y / sizeof(float);
	const int output_pitch_tf = output_pitch_t / sizeof(float);
	
    const int DATA_SIZE = SWATH_SIZE + KERN_SIZE - 1;
	
	//5+4-1 = DS = 8
	//7+4-1 = DS = 10
	//9+4-1 = DS = 12
	//11 + 5 - 1 = DS = 14
	//SIZE OF DATA = 8 * 8 * 8... etc
	 __local float data[10][10][10];


	
    const unsigned int block_y = get_group_id(0) % (output_y / SWATH_SIZE);
    const unsigned int block_t = get_group_id(0) / (output_y / SWATH_SIZE);
    const unsigned int offset_x = get_group_id(1) * SWATH_SIZE;
    const unsigned int offset_y = block_y * SWATH_SIZE;
    const unsigned int offset_t = block_t * SWATH_SIZE;
	
    for (unsigned int t=0; t < DATA_SIZE; t++)
    {
        for (unsigned int y=0; y < DATA_SIZE; y += KERN_SIZE)
        {
            const unsigned int my_y = y + get_local_id(0);
            if (my_y < DATA_SIZE)
            {
                for (unsigned int x=0; x < DATA_SIZE; x += 16)
                {
                    const unsigned int my_x = x + get_local_id(1);
                    if (my_x < DATA_SIZE) {
                        data[t][my_y][my_x] =
						video[(offset_t + t)*video_pitch_tf
						 + (offset_y + my_y)*video_pitch_yf
						 +offset_x + my_x ];
							                }
                 }
             }
          }
	}

   barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
	
    /**
     * @todo put this whole thing in a loop in case SWATH_SIZE ever
     * gets bigger than 4.  Actually, we count on the fact hat
     * SWATH_SIZE * SWATH_SIZE = 16.
     */
    const int base_x = get_local_id(1) % 4;
    const int base_t = get_local_id(1) / 4;
    const int base_y = get_local_id(0);
	
    /**
     * @warning We depend on blockDim(0) == KERN_SIZE >= SWATH_SIZE
     * @todo Remove dependency between KERN_SIZE and SWATH_SIZE
     */
    if (base_y >= SWATH_SIZE) {
        return;
    }
	
    float sum = 0;
    for (unsigned int t=0; t < KERN_SIZE; t++) {
        for (unsigned int y=0; y < KERN_SIZE; y++) {
            for (unsigned int x=0; x < KERN_SIZE; x++) {
                sum += data[base_t + t][base_y + y][base_x + x] *
				constant_kernel[((((KERN_SIZE-t-1) * KERN_SIZE)
				+ (KERN_SIZE-y-1)) * KERN_SIZE) + (KERN_SIZE-x-1)];
            }
        }
    }
    // Write out the solution
    		 output[( offset_t + base_t)*kernel_pitch_tf
		 + (offset_y + base_y) *output_pitch_yf
		 +offset_x + base_x]= sum;
}