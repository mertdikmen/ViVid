/*static const unsigned int BLOCK_SIZE = 16;*/

__constant int EUCLIDEAN = 0;
 __constant int DOTPRODUCT = 1;
 __constant int CHISQUARED =  2;
 __constant int CITYBLOCK = 3;
  __constant int ABSDOTPRODUCT=  4;
  

/*
 * @note This kernel is based on the blocked matrix multiply.  We
 * expect to be caleld with blockDim(BLOCK_SIZE, BLOCK_SIZE) and a
 * sufficiently large grid to cover all othe output values.
 */
__kernel void E_pairwiseDistanceKernel(
        __global float* a, const int a_width, const int a_height, const int a_pitch,
        __global float* b, const int b_width, const int b_height, const int b_pitch,
        __global float* out, const int o_width, const int o_height, const int o_pitch,
        const int BLOCK_SIZE)
{

    const int a_pitch_f = a_pitch / sizeof(float);
    const int b_pitch_f = b_pitch / sizeof(float);
    const int o_pitch_f = o_pitch / sizeof(float);

    const int out_ry = get_group_id(1) * BLOCK_SIZE + get_local_id(1);
    const int out_cx = get_group_id(0) * BLOCK_SIZE + get_local_id(0);

	const int out_ry_mul = out_ry * a_pitch_f;
	

    const int b_ry = get_group_id(0) * BLOCK_SIZE + get_local_id(1);

	const int b_ry_mul = b_ry * b_pitch_f;

	const int myInd = get_local_id(1) * BLOCK_SIZE + get_local_id(0);

	const int myCoef1 = get_local_id(1) * BLOCK_SIZE;
	const int myCoef0 = get_local_id(1) * BLOCK_SIZE;

    __local float a_cache[16 * 16];
    __local float b_cache[16 * 16];

    float dst = 0;

    int end = 0;

	

    for (unsigned int i=0; i < a_width; i+=BLOCK_SIZE)
    {
        int read_cx = i + get_local_id(0);
        // if (read_cx < a_width) {
        //    if (out_ry < a_height) {
                a_cache[myInd] = 
                    a[out_ry_mul + read_cx];
       //     }
        //    if (b_ry < b_height)
            {
                b_cache[myInd] =
                    b[b_ry_mul + read_cx];
            }
     //   }
        barrier(CLK_LOCAL_MEM_FENCE);
       
	    end = BLOCK_SIZE;
	//	end =  a_width - i;
   //     if (BLOCK_SIZE < (a_width - i))
 //       {
  //          end = BLOCK_SIZE;
  //      }
//        else 
  //      {
    //        end = a_width - i;
    //    }
    
        for (unsigned int k=0; k < end; k++)
        {
                float diff = a_cache[myCoef1 + k]  - b_cache[myCoef0 + k];
                dst += diff * diff;
				// dst = mad(diff, diff, dst);
        }
        //barrier( CLK_LOCAL_MEM_FENCE);
    }

    //if ((out_cx < o_width) && (out_ry < o_height))
    {
        out[out_ry * o_pitch_f + out_cx] = dst;
    }
}
