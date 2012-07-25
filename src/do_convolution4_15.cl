/*void do_convolution3_load(__local float **data,
                     float* video,
                     int offset_x, int offset_t,
                     int y,size_t pitch_t,size_t pitch_y,
					 int KERN_SIZE, int DATA_SIZE, int SWATH_SIZE)
{
    // Write the next set of data into the buffer.  We make use of the
    // fact that kernel_y == kernel_t == blockDim(0) to load
    // all t values in parallel.
    data[get_local_id(0)][get_local_id(1)] =video[
	(offset_t + get_local_id(0))*pitch_t
	+(y)*pitch_y
	+offset_x + get_local_id(1)];

	
    // Get the trim
    if (get_local_id(1) < DATA_SIZE - SWATH_SIZE) {
        data[get_local_id(0)][get_local_id(1) + SWATH_SIZE] =
		video[
		(offset_t + get_local_id(0))*pitch_t
		+y*pitch_y
		+offset_x + SWATH_SIZE + get_local_id(1)
		];
    }
}


void  do_convolution4_consume(float data[KERN_SIZE][DATA_SIZE],
                        DeviceMatrixCL3D output,
                        unsigned int offset_x, unsigned int offset_t,
                        int y, int& kern_y, float& sum,int KERN_SIZE, int DATA_SIZE)
{
    if (kern_y >= 0)
    {
        // Read the data we wrote last time and increment partial sums
        for (int t = 0; t < KERN_SIZE; t++)
        {
            for (int x = 0; x < KERN_SIZE; x++)
            {
                sum += data[t][get_local_id(1) + x] *
				convolution_kernel_get(KERN_SIZE-t-1, KERN_SIZE-kern_y-1,
									   KERN_SIZE-x-1);
            }
        }
    }
	
    kern_y++;
	
    if (kern_y == KERN_SIZE)
    {
       
        output[ offset_t*output_pitch_tf + (y-KERN_SIZE+1)*output_pitch_yf+ offset_x + get_local_id(1)]
		= sum;
        sum = 0;
        kern_y = 0;
    }
}


*/


__kernel void  do_convolution4_15(
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
	int KERN_SIZE
	)
{

	const int video_pitch_yf = video_pitch_y / sizeof(float);
	const int video_pitch_tf = video_pitch_t / sizeof(float);
	
	const int kernel_pitch_yf = kernel_pitch_y / sizeof(float);
	const int kernel_pitch_tf = kernel_pitch_t / sizeof(float);
	
	const int output_pitch_yf = output_pitch_y / sizeof(float);
	const int output_pitch_tf = output_pitch_t / sizeof(float);

   const int SWATH_SIZE = 16;
	
    const unsigned int offset_x = SWATH_SIZE * get_group_id(1);
    const unsigned int offset_t = get_group_id(0);
	
    const int DATA_SIZE = SWATH_SIZE + KERN_SIZE - 1;
	
    __local float data[2][15][30];
	
    // This variable indexes the kernel as we slide the image through.
    // We start out in the negative zone to avoid writing out results
    // outside the "valid" zone.  The values here should range from
    // -(KERN_SIZE-1):0
    int kern_y = get_local_id(0) - KERN_SIZE + 1;
	
    // This tracks the partial sum
    float sum = 0;
	
   /* do_convolution3_load<KERN_SIZE, DATA_SIZE, SWATH_SIZE>
	(data[0],
	 video, offset_x, offset_t,
	 0);*/
{
    // Write the next set of data into the buffer.  We make use of the
    // fact that kernel_y == kernel_t == blockDim(0) to load
    // all t values in parallel.
	int y = 0;
    data[0][get_local_id(0)][get_local_id(1)] =video[
	(offset_t + get_local_id(0))*video_pitch_tf
	+(y)*video_pitch_yf
	+offset_x + get_local_id(1)];

	
    // Get the trim
    if (get_local_id(1) < DATA_SIZE - SWATH_SIZE) {
        data[0][get_local_id(0)][get_local_id(1) + SWATH_SIZE] =
		video[
		(offset_t + get_local_id(0))*video_pitch_tf
		+y*video_pitch_yf
		+offset_x + SWATH_SIZE + get_local_id(1)
		];
    }
}	
	
	
	
	
    // Carve a swath in y.
    for (int y=0; y < (video_y-1); y++) {
        //for (int y=0; y < 9; y++) {
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
		
        // Flag for ping-ponging
        int read_buffer = y % 2;
		
       /* do_convolution4_consume<KERN_SIZE, DATA_SIZE>
		(data[read_buffer],
		 output, offset_x, offset_t,
		 y, kern_y, sum);
		*/
		
{
    if (kern_y >= 0)
    {
        // Read the data we wrote last time and increment partial sums
        for (int t = 0; t < KERN_SIZE; t++)
        {
            for (int x = 0; x < KERN_SIZE; x++)
            {
                sum += data[read_buffer][t][get_local_id(1) + x] *
				constant_kernel[((((KERN_SIZE-t-1) * KERN_SIZE) + 
				(KERN_SIZE-kern_y-1)) * KERN_SIZE) + (KERN_SIZE-x-1)];
				
            }
        }
    }
	
    kern_y++;
	
    if (kern_y == KERN_SIZE)
    {
       
        output[ offset_t*output_pitch_tf + (y-KERN_SIZE+1)*output_pitch_yf+ offset_x + get_local_id(1)]
		= sum;
        sum = 0;
        kern_y = 0;
    }
}
		
        // Load the next frame -- hopefully the compiler will move the
        // independent load earlier in the loop.
        /*do_convolution3_load<KERN_SIZE, DATA_SIZE, SWATH_SIZE>
		(data[!read_buffer],
		 video, offset_x, offset_t,
		 y+1);
		 */
		 
{
    // Write the next set of data into the buffer.  We make use of the
    // fact that kernel_y == kernel_t == blockDim(0) to load
    // all t values in parallel.
	int y1 = y+1;
    data[!read_buffer][get_local_id(0)][get_local_id(1)] =video[
	(offset_t + get_local_id(0))*video_pitch_tf
	+(y1)*video_pitch_yf
	+offset_x + get_local_id(1)];

	
    // Get the trim
    if (get_local_id(1) < DATA_SIZE - SWATH_SIZE) {
        data[!read_buffer][get_local_id(0)][get_local_id(1) + SWATH_SIZE] =
		video[
		(offset_t + get_local_id(0))*video_pitch_tf
		+y1*video_pitch_yf
		+offset_x + SWATH_SIZE + get_local_id(1)
		];
    }
}
		 
		 
    }
	
    {
        // One last write
        int y = video_y-1;
        int read_buffer = y % 2;
		
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
/*        do_convolution4_consume<KERN_SIZE, DATA_SIZE>
		(data[read_buffer],
		 output, offset_x, offset_t,
		 y, kern_y, sum);
*/		 
{
    // Write the next set of data into the buffer.  We make use of the
    // fact that kernel_y == kernel_t == blockDim(0) to load
    // all t values in parallel.
	int y1 = y;
    data[!read_buffer][get_local_id(0)][get_local_id(1)] =video[
	(offset_t + get_local_id(0))*video_pitch_tf
	+(y1)*video_pitch_yf
	+offset_x + get_local_id(1)];

	
    // Get the trim
    if (get_local_id(1) < DATA_SIZE - SWATH_SIZE) {
        data[!read_buffer][get_local_id(0)][get_local_id(1) + SWATH_SIZE] =
		video[
		(offset_t + get_local_id(0))*video_pitch_tf
		+y1*video_pitch_yf
		+offset_x + SWATH_SIZE + get_local_id(1)
		];
    }
}
    }
	
}