
	__constant int MAX_FILTERBANK_SIZE  = 10000;
	__constant int N_MAX_CHANNELS = 10;




__kernel void blockwise_distance_kernel( __global float* frame, const int f_width, const int f_height, const int f_pitch,
 __global float* output, const int output_x, const int output_y,const int output_t,  const int o_pitch_y,const int o_pitch_t,
                                         const int frame_width, const int frame_height,
                                         const int FILTER_DIM, const int  BLOCK_MULT,const int BLOCK_SIZE,const int optype,
										 __constant float * c_FilterBank)
{
	
	
 	const int dim_t =  output_t;
	
	const int f_pitch_f = f_pitch / sizeof(float);
	const int o_pitch_yf = o_pitch_y / sizeof(float);
	const int o_pitch_tf = o_pitch_t / sizeof(float);
	
	const int out_ry = get_group_id(0) * BLOCK_SIZE + get_local_id(0);
    const int out_cx = get_group_id(1) * BLOCK_SIZE + get_local_id(1);
	
	const int out_pix_y0 = get_group_id(0) * (BLOCK_SIZE * BLOCK_MULT) + FILTER_DIM / 2;
	
	const int out_pix_x0 = get_group_id(1) * (BLOCK_SIZE * BLOCK_MULT) + FILTER_DIM / 2;

    const int out_pix_y1 = min(out_pix_y0 + (BLOCK_SIZE * BLOCK_MULT),
                               frame_height - FILTER_DIM / 2);
    
	const int out_pix_x1 = min(out_pix_x0 + (BLOCK_SIZE * BLOCK_MULT),
                               frame_width - FILTER_DIM / 2);
							   
	const int cache_size = BLOCK_SIZE * BLOCK_MULT + FILTER_DIM - 1;
	 
	 __local float *image_cache;
	

	if(cache_size == 34){
	
	__local float temp[34 * 34];
		image_cache = temp;
	}else if(cache_size == 36){
	
	__local float temp[36 * 36];
		image_cache = temp;
		
	}else if(cache_size == 38){
	
	__local float temp[38 * 38];
		image_cache = temp;
	
		
	}
	
	int read_pix_y = out_pix_y0 - FILTER_DIM / 2 + get_local_id(0);
    int cache_ind_y = get_local_id(0);
    for (int ii=0; ii<BLOCK_MULT+1; ii++){
        int read_pix_x = out_pix_x0 - FILTER_DIM / 2 + get_local_id(1);
        int cache_ind_x = get_local_id(1);
        for (int jj=0; jj<BLOCK_MULT+1; jj++){
            if ((cache_ind_x < cache_size) && (cache_ind_y < cache_size)){
                image_cache[cache_ind_y*cache_size+cache_ind_x] = frame[read_pix_y * f_pitch_f + read_pix_x];
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
            if ((out_y < out_pix_y1) && (out_x < out_pix_x1)){           
                for (int filter_id=0; filter_id<dim_t; filter_id++){
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
