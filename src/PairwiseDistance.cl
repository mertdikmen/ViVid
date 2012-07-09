/*static const unsigned int BLOCK_SIZE = 16;*/

/*const int EUCLIDEAN 0
  const int DOTPRODUCT 1
  const int CHISQUARED 2
  const int CITYBLOCK 3
  const int ABSDOTPRODUCT 4
*/

/*
 * @note This kernel is based on the blocked matrix multiply.  We
 * expect to be caleld with blockDim(BLOCK_SIZE, BLOCK_SIZE) and a
 * sufficiently large grid to cover all othe output values.
 */
__kernel void pairwiseDistanceKernelGeneric(
        __global float* a, const int a_width, const int a_height, const int a_pitch,
        __global float* b, const int b_width, const int b_height, const int b_pitch,
        __global float* out, const int o_width, const int o_height, const int o_pitch,
        const int type, const int BLOCK_SIZE)
{

    const int a_pitch_f = a_pitch / sizeof(float);
    const int b_pitch_f = b_pitch / sizeof(float);
    const int o_pitch_f = o_pitch / sizeof(float);

    const int out_ry = get_group_id(0) * BLOCK_SIZE + get_local_id(0);
    const int out_cx = get_group_id(1) * BLOCK_SIZE + get_local_id(1);

    const int b_ry = get_group_id(1) * BLOCK_SIZE + get_local_id(0);

    __local float a_cache[16 * 16];
    __local float b_cache[16 * 16];

    float dst = 0;

    int end = 0;

    for (unsigned int i=0; i < a_width; i+=BLOCK_SIZE)
    {
        int read_cx = i + get_local_id(1);
        if (read_cx < a_width) {
            if (out_ry < a_height) {
                a_cache[get_local_id(0) * BLOCK_SIZE + get_local_id(1)] = 
                    a[out_ry * a_pitch_f + read_cx];
            }
            if (b_ry < b_height)
            {
                b_cache[get_local_id(0) * BLOCK_SIZE + get_local_id(1)] =
                    b[b_ry * b_pitch_f + read_cx];
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
        
        if (BLOCK_SIZE < (a_width - i))
        {
            end = BLOCK_SIZE;
        }
        else 
        {
            end = a_width - i;
        }
    
        for (unsigned int k=0; k < end; k++)
        {
            float diff = a_cache[get_local_id(0) * BLOCK_SIZE + k] - b_cache[get_local_id(1) * BLOCK_SIZE + k];
            dst += diff * diff;
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }

    if ((out_cx < o_width) && (out_ry < o_height))
    {
        out[out_ry * o_pitch_f + out_cx] = dst;
    }
}
