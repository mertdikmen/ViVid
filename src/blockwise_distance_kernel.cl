__constant int MAX_FILTERBANK_SIZE  = 10000;
__constant int N_MAX_CHANNELS = 10;

__kernel void blockwise_distance_kernel(
    __global float* frame, const int f_width, const int f_height, const int f_pitch,
    __global float* output, const int output_x, const int output_y,
    const int output_t, const int o_pitch_y,const int o_pitch_t,
    const int frame_width, const int frame_height,
    const int FILTER_DIM, const int  BLOCK_MULT,const int BLOCK_SIZE,const int optype,
    __global float * c_FilterBank,
    const int n_filters)
{
	const int f_pitch_f = f_pitch / sizeof(float);
	const int o_pitch_yf = o_pitch_y / sizeof(float);
	const int o_pitch_tf = o_pitch_t / sizeof(float);
	
	const int out_pix_y0 = get_group_id(0) * (BLOCK_SIZE * BLOCK_MULT) + FILTER_DIM / 2;
	
	const int out_pix_x0 = get_group_id(1) * (BLOCK_SIZE * BLOCK_MULT) + FILTER_DIM / 2;

    const int out_pix_y_stop = min(out_pix_y0 + (BLOCK_SIZE * BLOCK_MULT),
                               frame_height - FILTER_DIM / 2);
    
	const int out_pix_x_stop = min(out_pix_x0 + (BLOCK_SIZE * BLOCK_MULT),
                               frame_width - FILTER_DIM / 2);
							   
	const int cache_size = 38;

    __local float image_cache[38*38];

    int read_pix_x, read_pix_y;
    int cache_ind_x, cache_ind_y;

	const int max_im_ind = f_height * f_pitch_f;

    read_pix_y = out_pix_y0 - FILTER_DIM / 2 + get_local_id(0);
    cache_ind_y = get_local_id(0);
    for (int ii = 0;  ii < BLOCK_MULT + 1; ii++){
        read_pix_x = out_pix_x0 - FILTER_DIM / 2 + get_local_id(1);
        cache_ind_x = get_local_id(1);
        for (int jj = 0; jj < BLOCK_MULT + 1; jj++){
            if ((cache_ind_x < cache_size) && (cache_ind_y < cache_size)){
				const int read_ind = read_pix_y * f_pitch_f + read_pix_x;

				if ((read_ind > 0) && (read_ind < max_im_ind))
				{
					image_cache[cache_ind_y*cache_size+cache_ind_x] = frame[read_ind];
				}
				else 
				{
					image_cache[cache_ind_y*cache_size+cache_ind_x] = 0;
				}
            }
            read_pix_x += BLOCK_SIZE;
            cache_ind_x += BLOCK_SIZE;
        }
        read_pix_y += BLOCK_SIZE;
        cache_ind_y += BLOCK_SIZE;
    }

    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

  int out_y = out_pix_y0 + get_local_id(0);
    for (int ii=0; ii<BLOCK_MULT; ii++){
        int out_x = out_pix_x0 + get_local_id(1);       
        for (int jj=0; jj<BLOCK_MULT; jj++){
            float curval = -1e6;
            float curid = -1;
            int fi = 0;
            if ((out_y < out_pix_y_stop) && (out_x < out_pix_x_stop)){ 
                for (int filter_id=0; filter_id<n_filters; filter_id++){
                    float tempval = 0.0f;
                    int cyi = get_local_id(0)+ ii * BLOCK_SIZE;
                    for (int fyi=0; fyi<FILTER_DIM; fyi++){ 
                        int cxi = get_local_id(1) + jj * BLOCK_SIZE;
                        for (int fxi=0; fxi<FILTER_DIM; fxi++){
                            tempval += c_FilterBank[fi++] * image_cache[cyi*cache_size+cxi];
                            cxi++;
                        }
                        cyi++;
                    }
                    if (fabs(tempval) > curval){
                        curid = filter_id;
                        curval = fabs(tempval);
                    }
                }
                const int out_pix_offset = out_y * o_pitch_yf + out_x;

                *(output + out_pix_offset) = curid;
                *(output + o_pitch_tf + out_pix_offset) = curval;
            }
			 
            out_x += BLOCK_SIZE;
        }
        out_y += BLOCK_SIZE;
    }
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
}
