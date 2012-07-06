


//static const unsigned int BLOCK_SIZE = 16;



/*const int EUCLIDEAN 0
const int DOTPRODUCT 1
const int CHISQUARED 2
const int CITYBLOCK 3
const int ABSDOTPRODUCT 4
*/

//DeviceMatrixOpenCL a, DeviceMatrixOpenCL b,DeviceMatrixOpenCL out, const int type
											  
/**
 * @note This kernel is based on the blocked matrix multiply.  We
 * expect to be caleld with blockDim(BLOCK_SIZE, BLOCK_SIZE) and a
 * sufficiently large grid to cover all othe output values.
 */
__kernel void pairwiseDistanceKernelGeneric(
__global float * a,const int a_width,const int a_height,const int a_pitch,
__global float * b,const int b_width,const int b_height,const int b_pitch,
__global float * out,const int o_width,const int o_height,const int o_pitch, const int type,const int BLOCK_SIZE)
{
    int i_coordinate = get_global_id(1);
	
	int j_coordinate = get_global_id(0);
	
	
	
	const int out_ry = get_group_id(1)  * BLOCK_SIZE +  get_local_id(1);
	
    const int out_cx = get_group_id(0) * BLOCK_SIZE + get_local_id(0);
	
	const int b_ry = get_group_id(0) * BLOCK_SIZE + get_local_id(0);
	
	
	
	
	
	

	if(j_coordinate<a_width && i_coordinate<a_height){
			//index for the row of each matrix. We doo row + j_coordinate to access to
			// the position which we need.
			int row_a = ((i_coordinate)/((float) a_width-1))*a_pitch ;
			int row_b = ((i_coordinate)/((float) b_width-1))*b_pitch ;
			int row_o = ((i_coordinate)/((float) o_width-1))*o_pitch ;
			
		//	out[row+j_coordinate]=j_coordinate;	
	
	
	__local float a_cache[16][16];
	__local float b_cache[16][16];
	
	float dst = 0;
    for ( int i=0; i < a_width; i+=BLOCK_SIZE) {
       
        if (j_coordinate < a_width) {
            if (i_coordinate < a_height) {
                a_cache[get_local_id(1)][get_local_id(0)] =
					a[row_a+j_coordinate];
                              }
            if (i_coordinate < b_height) {
                b_cache[get_local_id(1)][get_local_id(0)] =
                          b[row_b+j_coordinate];
						}
        }
		
		barrier(CLK_GLOBAL_MEM_FENCE ) ;
		
		
		
		
		int end = min(BLOCK_SIZE,(a_width - i));
        for (int k=0; k < end; k++) {
            if (type == 0){
                float diff = a_cache[get_local_id(1)][k] - b_cache[get_local_id(0)][k];
                dst += diff * diff;
            }
            else if (type == 1){
                dst += a_cache[get_local_id(1)][k] * b_cache[get_local_id(0)][k];
            }
            else if (type == 4){
                dst += a_cache[get_local_id(1)][k] * b_cache[get_local_id(0)][k];
            }
            else if (type == 2){
                float diff, sum;
                diff = a_cache[get_local_id(1)][k] - b_cache[get_local_id(0)][k];
                sum  = a_cache[get_local_id(1)][k] + b_cache[get_local_id(0)][k];
                dst += diff * diff / sum;
            }
            else if (type == 3){
                dst += fabs(a_cache[get_local_id(1)][k] - b_cache[get_local_id(0)][k]);
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE ) ;
		
		
		if ((out_cx < o_width) && (out_ry < o_height)) {
			if (type == 4){
				out[row_o+j_coordinate] = fabs(dst);
			}else {
				out[row_o+j_coordinate] = dst;
			}
		}


	}
	
	
	}
	
	/*const int out_ry = blockIdx.y * BLOCK_SIZE + threadIdx.y;
	
    const int out_cx = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    // Index used for reading b.  We use the fact that our thread
    // blocks are square here.
    const int b_ry = blockIdx.x * BLOCK_SIZE + threadIdx.y;

    __shared__ float a_cache[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float b_cache[BLOCK_SIZE][BLOCK_SIZE];

    float dst = 0;
    for (unsigned int i=0; i < a.width; i+=BLOCK_SIZE) {
        int read_cx = i + threadIdx.x;
        if (read_cx < a.width) {
            if (out_ry < a.height) {
                a_cache[threadIdx.y][threadIdx.x] =
                       *getPtr(a, out_ry, read_cx);
            }
            if (b_ry < b.height) {
                b_cache[threadIdx.y][threadIdx.x] =
                       *getPtr(b, b_ry, read_cx);
            }
        }
        __syncthreads();
        int end = min(BLOCK_SIZE, (unsigned int)(a.width - i));
        for (int k=0; k < end; k++) {
            if (type == EUCLIDEAN){
                float diff = a_cache[threadIdx.y][k] - b_cache[threadIdx.x][k];
                dst += diff * diff;
            }
            else if (type == DOTPRODUCT){
                dst += a_cache[threadIdx.y][k] * b_cache[threadIdx.x][k];
            }
            else if (type == ABSDOTPRODUCT){
                dst += a_cache[threadIdx.y][k] * b_cache[threadIdx.x][k];
            }
            else if (type == CHISQUARED){
                float diff, sum;
                diff = a_cache[threadIdx.y][k] - b_cache[threadIdx.x][k];
                sum  = a_cache[threadIdx.y][k] + b_cache[threadIdx.x][k];
                dst += diff * diff / sum;
            }
            else if (type == CITYBLOCK){
                dst += fabs(a_cache[threadIdx.y][k] - b_cache[threadIdx.x][k]);
            }
        }
        __syncthreads();
    }

    if ((out_cx < out.width) & (out_ry < out.height)) {
        if (type == ABSDOTPRODUCT){
            *getPtr(out, out_ry, out_cx) = abs(dst);
        }
        else {
            *getPtr(out, out_ry, out_cx) = dst;
        }
    }
	*/
}