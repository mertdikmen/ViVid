#include "FlexibleFilter.hpp"

#define MAX_FILTERBANK_SIZE 10000
#define N_MAX_
#define N_MAX_CHANNELS 10
//We are just allocating max possible filterbank ahead of time
//whatever gets filled gets filled
//right now not expecting a filterbank larger than 10000 floats
//we'll see how that pans out
__device__ __constant__ float c_FilterBank[MAX_FILTERBANK_SIZE];

static __device__ float* getPtr(const DeviceMatrix mat, 
                         unsigned int r, unsigned int c)
{
    return mat.data + r*mat.pitch + c;
}

static __device__ float* getPtr(const DeviceMatrix3D mat, 
                         unsigned int t, unsigned int y, unsigned int x)
{
    return mat.data + t*mat.pitch_t + y*mat.pitch_y + x;
}

#define HIST_CACHE_SIZE 10
static const unsigned int MAX_HISTOGRAM_SIZE = 512;
static const unsigned int MAX_FOLDS = MAX_HISTOGRAM_SIZE / (BLOCK_8 * BLOCK_8);
__global__ void cell_histogram_kernel(DeviceMatrix3D input,
                                      DeviceMatrix3D output,
                                      const int cell_size,
                                      const int offset_y,
                                      const int offset_x,
                                      const int max_bin){

    const int out_y = blockIdx.y ;
    const int out_x = blockIdx.x ;

    const int n_threads = BLOCK_8 * BLOCK_8;
    const int thread_id =  threadIdx.y * BLOCK_8 + threadIdx.x ;

    __shared__ int   id_cache[BLOCK_8*BLOCK_8];
    __shared__ float wt_cache[BLOCK_8*BLOCK_8];

    const int base_y = offset_y + blockIdx.y * cell_size;
    const int base_x = offset_x + blockIdx.x * cell_size;

    const int cell_area = cell_size * cell_size;
    const int n_folds = (max_bin - 1) / n_threads + 1;

    float local_hist[MAX_FOLDS];

    //read the cell into the shared memory
    if ((threadIdx.y < cell_size) && (threadIdx.x < cell_size)){
        id_cache[threadIdx.y*cell_size + threadIdx.x] = 
            *getPtr(input, 0, base_y + threadIdx.y, base_x + threadIdx.x);

        wt_cache[threadIdx.y*cell_size + threadIdx.x] = 
            *getPtr(input, 1, base_y + threadIdx.y, base_x + threadIdx.x);
    }

    for (int i=0;i<MAX_FOLDS;i++){
        local_hist[i] = 0;
    }

    __syncthreads();

    //loop over the cell pixels and increment the histogram bin if the id matches
    for (int i=0; i<cell_area; i++){
        const int cur_id = id_cache[i];
        const float cur_wt = wt_cache[i];
        int hist_i = thread_id;
        for (int fi=0; fi<n_folds; fi++){
            if (cur_id == hist_i){
                local_hist[fi] += cur_wt; 
            }
            hist_i += n_threads;
        }
    }

    __syncthreads();

    //write out the histogram
    if ( (out_y < output.dim_t) && (out_x < output.dim_y) ){
        for (int fi=0; fi<n_folds; fi++){
            int hist_i = n_threads * fi + thread_id;
            if (hist_i < max_bin){
                *getPtr(output, out_y, out_x, hist_i) = local_hist[fi];
            }
        }
    }

    __syncthreads();
}

/* FOR COMPUTE CAPABILITY 2.0 AND ABOVE 
__global__ void cell_histogram_kernel(DeviceMatrix3D input,
                                      DeviceMatrix3D output,
                                      const int cell_size,
                                      const int offset_y,
                                      const int offset_x,
                                      const int max_bin){
  

    __shared__ float cell_histogram[MAX_HISTOGRAM_SIZE];

    int n_folds =  (max_bin - 1) / (BLOCK_SIZE * BLOCK_SIZE) + 1;

    for (int i=0; i<n_folds; i++){
        int hist_i = BLOCK_SIZE * BLOCK_SIZE * i + threadIdx.y * BLOCK_SIZE + threadIdx.x ;
        if (hist_i < max_bin){
            cell_histogram[hist_i] = 0;
        }
    }
    __syncthreads();

    int dim_folds = (cell_size - 1) / BLOCK_SIZE + 1;

    int base_y = offset_y + blockIdx.y * cell_size;
    int base_x = offset_x + blockIdx.x * cell_size;

    for (int i=0; i<dim_folds;i++){
        for (int j=0; j<dim_folds;j++){
            int thread_offset_y =  (BLOCK_SIZE * i) + threadIdx.y;
            int thread_offset_x =  (BLOCK_SIZE * j) + threadIdx.x;

            if ((thread_offset_y < cell_size) && (thread_offset_x < cell_size)){
                int id       = (int) *getPtr(input,0, base_y + thread_offset_y, base_x + thread_offset_x);
                float weight = *getPtr(input,1, base_y + thread_offset_y, base_x + thread_offset_x);

                //cell_histogram[id]+=weight;
                atomicAdd((float*)cell_histogram + id, weight);
            }
        }
    }

    __syncthreads();

    if ((threadIdx.x == 0)&&(threadIdx.y == 0)){
        for (int i=0;i<max_bin;i++){
            *getPtr(output, blockIdx.y, blockIdx.x, i) = 1;
        }
    }


    if ( (blockIdx.y < output.dim_t) && (blockIdx.x < output.dim_y) ){
        for (int i=0; i<n_folds;i++){
           int out_bin = threadIdx.y * (BLOCK_SIZE*BLOCK_SIZE) + threadIdx.x;
           if (out_bin < max_bin){
               *getPtr(output, blockIdx.y, blockIdx.x, out_bin) = 1;//cell_histogram[out_bin];
            }
        }
    }

    __syncthreads();
}
*/

__global__ void blockwise_filter_kernel(DeviceMatrix frame,
                                         DeviceMatrix3D output,
                                         const int frame_width, const int frame_height,
                                         const int apron_lo_y, const int apron_lo_x,
                                         const int apron_hi_y, const int apron_hi_x,
                                         const int dim_t, const int dim_y, const int dim_x,
                                         const int nchannels,
                                         const int optype)
{
    const int pix_y = blockIdx.y * (BLOCK_SIZE-dim_y+1) + threadIdx.y;
    const int pix_x = blockIdx.x * (BLOCK_SIZE-dim_x+1) + threadIdx.x;

    __shared__ float image_cache[BLOCK_SIZE][BLOCK_SIZE];

    __syncthreads();

    const int min_y = apron_hi_y;
    const int min_x = apron_hi_x;
    const int max_y = frame_height - apron_hi_y ;
    const int max_x = frame_width  - apron_hi_x ;

    for (int di=0; di<dim_t;di++){
        float tempval = 0;
        int fi_base = di * dim_y * dim_x * nchannels;
        for (int chan_id=0; chan_id<nchannels; chan_id++){
            int fi = chan_id + fi_base;

            image_cache[threadIdx.y][threadIdx.x] = 0;
            //load the particular channel
            if ( (pix_y >= 0) && (pix_y < frame_height) && (pix_x >= 0) && (pix_x < frame_width) ){
                image_cache[threadIdx.y][threadIdx.x] = *getPtr(frame, pix_y, pix_x * nchannels + chan_id);
            }

            __syncthreads();

            if (optype==FF_OPTYPE_EUCLIDEAN){
                for (int fyi=-apron_lo_y; fyi <= apron_hi_y; fyi++){
                    for (int fxi=-apron_lo_x; fxi <= apron_hi_x; fxi++){
                        float diff = image_cache[threadIdx.y+fyi][threadIdx.x+fxi] - c_FilterBank[fi];
                        tempval += diff*diff;
                        fi+=nchannels;
                    }
                }

            }
            else if (optype==FF_OPTYPE_COSINE){
                for (int fyi=-apron_lo_y; fyi <= apron_hi_y; fyi++){
                    for (int fxi=-apron_lo_x; fxi <= apron_hi_x; fxi++){
                        tempval += image_cache[threadIdx.y+fyi][threadIdx.x+fxi] * c_FilterBank[fi];
                        fi+=nchannels;
                    }
                }
            }
            __syncthreads();
        }

        if ( (pix_y >= min_y) && (pix_y < max_y) && (pix_x >= min_x) && (pix_x < max_x) ){
            if ( (threadIdx.y >= apron_lo_y) && (threadIdx.y < BLOCK_SIZE-apron_hi_y) && 
                 (threadIdx.x >= apron_lo_x) && (threadIdx.x < BLOCK_SIZE-apron_hi_x) ){
                *getPtr(output, pix_y, pix_x, di) = tempval;
            }
        }
        else if ( (pix_y >= 0) && (pix_y < frame_height) && (pix_x >= 0) && (pix_x < frame_width) ){
            *getPtr(output, pix_y, pix_x, di) = -1;
        }
    
    }

    __syncthreads();

}

/*
template <int FILTER_DIM>
__global__ void blockwise_distance_kernel(DeviceMatrix frame,
                                         DeviceMatrix3D output,
                                         const int frame_width, const int frame_height,
                                         const int dim_t,
                                         const int optype)
{
    const int pix_y = blockIdx.y * (BLOCK_SIZE-FILTER_DIM+1) + threadIdx.y;
    const int pix_x = blockIdx.x * (BLOCK_SIZE-FILTER_DIM+1) + threadIdx.x;

    const int out_pix_y = pix_y + FILTER_DIM / 2;
    const int out_pix_x = pix_x + FILTER_DIM / 2;

    const int out_pix_offset = out_pix_y*output.pitch_y + out_pix_x;

    __shared__ float image_cache[BLOCK_SIZE][BLOCK_SIZE];
    image_cache[threadIdx.y][threadIdx.x] = *(frame.data + pix_y*frame.pitch + pix_x);
    __syncthreads();

    float curval = -1e6;
    float curid = -1;
    float tempval;

    int fi=0;

    if ( (threadIdx.y < BLOCK_SIZE-FILTER_DIM+1) && (threadIdx.x < BLOCK_SIZE-FILTER_DIM+1) ){
        for (int filter_id=0; filter_id<dim_t; filter_id++){
            tempval = 0;

            if (optype==FF_OPTYPE_EUCLIDEAN) {
                for (int fyi=0; fyi<FILTER_DIM; fyi++){for (int fxi=0; fxi<FILTER_DIM; fxi++){
                    float diff = image_cache[threadIdx.y+fyi][threadIdx.x+fxi] - c_FilterBank[fi++];
                    tempval += diff * diff;
                }}
            }
            else { // optype==FF_OPTYPE_COSINE 
                for (int fyi=0; fyi<FILTER_DIM; fyi++){ for (int fxi=0; fxi<FILTER_DIM; fxi++){
                    tempval += c_FilterBank[fi++] * image_cache[threadIdx.y+fyi][threadIdx.x+fxi];
                }}
            }

            if (optype==FF_OPTYPE_EUCLIDEAN){
               if (tempval < curval){
                   curid = filter_id;
                   curval = tempval;
               }
            }
            else { //(optype==FF_OPTYPE_COSINE){
                if (abs(tempval) > curval){
                    curid = filter_id;
                    curval = abs(tempval);
                }
            }
        }
    }

    if ( (out_pix_y < frame_height) && (out_pix_x < frame_width) &&
         (threadIdx.y < BLOCK_SIZE - FILTER_DIM + 1) && (threadIdx.x < BLOCK_SIZE - FILTER_DIM + 1) ){
        *(output.data + out_pix_offset) = curid;
        *(output.data + output.pitch_t + out_pix_offset) = curval;
    }
}
*/

//this updates the filterbank saved in the constant device memory
int update_filter_bank_internal(float* new_filter, int filter_size){

    cudaError_t cet;

    if (filter_size > MAX_FILTERBANK_SIZE){
        printf("ERROR: Filterbank too large\n");
        return 1;
    }
    else {
        //printf("Value in: %05f\n",new_filter[0]);
        cet = cudaMemcpyToSymbol(c_FilterBank, new_filter, sizeof(float) * filter_size,0, cudaMemcpyHostToDevice);
        //printf("err: %d\n", cet);
        if (cet){
            printf("Some error happened while updating the filterbank\n");
            printf("Error code: %d\n",cet);
            return cet;
        }
        //printf("Value out:%04f\n",c_FilterBank[0]);
        return 0;
    }

}


#define BLOCK_MULT 2
template<int FILTER_DIM>
__global__ void blockwise_distance_kernel(DeviceMatrix frame,
                                         DeviceMatrix3D output,
                                         const int frame_width, const int frame_height,
                                         const int dim_t,
                                         const int optype)
{
    const int out_pix_y0 = blockIdx.y * (BLOCK_SIZE * BLOCK_MULT) + FILTER_DIM / 2;
    const int out_pix_x0 = blockIdx.x * (BLOCK_SIZE * BLOCK_MULT) + FILTER_DIM / 2;

    const int out_pix_y1 = min(out_pix_y0 + (BLOCK_SIZE * BLOCK_MULT),
                               frame_height - FILTER_DIM / 2);
    const int out_pix_x1 = min(out_pix_x0 + (BLOCK_SIZE * BLOCK_MULT),
                               frame_width - FILTER_DIM / 2);

    const int cache_size = BLOCK_SIZE * BLOCK_MULT + FILTER_DIM - 1;

    __shared__ float image_cache[cache_size][cache_size];

    int read_pix_y = out_pix_y0 - FILTER_DIM / 2 + threadIdx.y;
    int cache_ind_y = threadIdx.y;
    for (int ii=0; ii<BLOCK_MULT+1; ii++){
        int read_pix_x = out_pix_x0 - FILTER_DIM / 2 + threadIdx.x;
        int cache_ind_x = threadIdx.x;
        for (int jj=0; jj<BLOCK_MULT+1; jj++){
            if ((cache_ind_x < cache_size) && (cache_ind_y < cache_size)){
                image_cache[cache_ind_y][cache_ind_x] = *(frame.data + read_pix_y * frame.pitch + read_pix_x);
            }
            read_pix_x += BLOCK_SIZE;
            cache_ind_x += BLOCK_SIZE;
        }
        read_pix_y += BLOCK_SIZE;
        cache_ind_y += BLOCK_SIZE;
    }

    __syncthreads();

    int out_y = out_pix_y0 + threadIdx.y;
    for (int ii=0; ii<BLOCK_MULT; ii++){
        int out_x = out_pix_x0 + threadIdx.x;       
        for (int jj=0; jj<BLOCK_MULT; jj++){
            float curval = -1e6;
            float curid = -1;
            int fi = 0;
            if ((out_y < out_pix_y1) && (out_x < out_pix_x1)){           
                for (int filter_id=0; filter_id<dim_t; filter_id++){
                    float tempval = 0.0f;
                    int cyi = threadIdx.y + ii * BLOCK_SIZE;
                    for (int fyi=0; fyi<FILTER_DIM; fyi++){ 
                        int cxi = threadIdx.x + jj * BLOCK_SIZE;
                        for (int fxi=0; fxi<FILTER_DIM; fxi++){
                            tempval += c_FilterBank[fi++] * image_cache[cyi][cxi];
                            cxi++;
                        }
                        cyi++;
                    }
                    if (abs(tempval) > curval){
                        curid = filter_id;
                        curval = abs(tempval);
                    }
                }
           
                const int out_pix_offset = out_y * output.pitch_y + out_x;
                *(output.data + out_pix_offset) = curid;
                *(output.data + output.pitch_t + out_pix_offset) = curval;
            }
            out_x += BLOCK_SIZE;
        }
        out_y += BLOCK_SIZE;
    }

    __syncthreads();
}

void dist_filter2_d3(const DeviceMatrix* frame,
                  const int dim_t, const int nchannels,
                  DeviceMatrix3D* output,
                  const int optype)
{
    const int frame_width = int(frame->width);
    const int frame_height = int(frame->height);

    const int valid_region_h = frame_height - 3 + 1;
    const int valid_region_w = frame_width - 3 + 1;

    int grid_ry = valid_region_h / (BLOCK_SIZE * BLOCK_MULT) + 1;
    int grid_cx = valid_region_w / (BLOCK_SIZE * BLOCK_MULT) + 1;

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    dim3 dimGrid(grid_cx, grid_ry);

    blockwise_distance_kernel<3><<<dimGrid, dimBlock>>>(*frame,
                                                    *output,
                                                    frame_width, frame_height,
                                                    dim_t,
                                                    optype);
}




/**CUDA***/
void dist_filter2_d5(const DeviceMatrix* frame,
                  const int dim_t, const int nchannels,
                  DeviceMatrix3D* output,
                  const int optype)
{
    const int frame_width = int(frame->width);
    const int frame_height = int(frame->height);

    const int valid_region_h = frame_height - 5 + 1;
    const int valid_region_w = frame_width - 5 + 1;

    int grid_ry = valid_region_h / (BLOCK_SIZE * BLOCK_MULT) + 1;
    int grid_cx = valid_region_w / (BLOCK_SIZE * BLOCK_MULT) + 1;

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    dim3 dimGrid(grid_cx, grid_ry);

    blockwise_distance_kernel<5><<<dimGrid, dimBlock>>>(*frame,
                                                    *output,
                                                    frame_width, frame_height,
                                                    dim_t,
                                                    optype);

}
void dist_filter2_d7(const DeviceMatrix* frame,
                  const int dim_t, const int nchannels,
                  DeviceMatrix3D* output,
                  const int optype)
{
    const int frame_width = int(frame->width);
    const int frame_height = int(frame->height);

    const int valid_region_h = frame_height - 7 + 1;
    const int valid_region_w = frame_width - 7 + 1;

    int grid_ry = valid_region_h / (BLOCK_SIZE * BLOCK_MULT) + 1;
    int grid_cx = valid_region_w / (BLOCK_SIZE * BLOCK_MULT) + 1;

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    dim3 dimGrid(grid_cx, grid_ry);

    blockwise_distance_kernel<7><<<dimGrid, dimBlock>>>(*frame,
                                                    *output,
                                                    frame_width, frame_height,
                                                    dim_t,
                                                    optype);
}
/*
//new dist filter implementation
template<int FILTER_DIM>
void dist_filter2(const DeviceMatrix* frame,
                  const int dim_t, const int nchannels,
                  DeviceMatrix3D* output,
                  const int optype)
{
    const int frame_width = float(frame->width) / (nchannels);
    const int frame_height = float(frame->height);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    int grid_ry = (frame_height) / (dimBlock.y-FILTER_DIM+1) + 1;
    int grid_cx = (frame_width ) / (dimBlock.x-FILTER_DIM+1) + 1;
    dim3 dimGrid(grid_cx, grid_ry);

    blockwise_distance_kernel<FILTER_DIM><<<dimGrid, dimBlock>>>(*frame,
                                                    *output,
                                                    frame_width, frame_height,
                                                    dim_t,
                                                    optype);
   

}
*/

//new dist filter implementation
void dist_filter_noargmin(const DeviceMatrix* frame,
                  const int dim_t, const int dim_y, const int dim_x, const int nchannels,
                  DeviceMatrix3D* output,
                  const int optype)
{
    const int frame_width = frame->width / nchannels;
    const int frame_height = frame->height;

    const int apron_hi_y = dim_y / 2;
    const int apron_hi_x = dim_x / 2;

    const int apron_lo_y = dim_y - apron_hi_y - 1;
    const int apron_lo_x = dim_x - apron_hi_x - 1;

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    int grid_ry = (frame_height) / (dimBlock.y-dim_y+1) + 1;
    int grid_cx = (frame_width ) / (dimBlock.x-dim_x+1) + 1;
    dim3 dimGrid(grid_cx, grid_ry);

    blockwise_filter_kernel<<<dimGrid, dimBlock>>>(*frame,
                                                    *output,
                                                    frame_width, frame_height,
                                                    apron_lo_y, apron_lo_x,
                                                    apron_hi_y, apron_hi_x,
                                                    dim_t, dim_y, dim_x,
                                                    nchannels, optype);

}

void hist_all_cells(const DeviceMatrix3D* inds_and_weights,
                    DeviceMatrix3D* output,
                    const int cell_size,
                    const int offset_y,
                    const int offset_x,
                    const int max_bin){

    const int frame_height = inds_and_weights->dim_y;
    const int frame_width = inds_and_weights->dim_x;

    dim3 dimBlock(BLOCK_8, BLOCK_8);

    int grid_ry = (frame_height - offset_y) / cell_size + 1 ;
    int grid_cx = (frame_width  - offset_x) / cell_size + 1 ;
   
    dim3 dimGrid(grid_cx, grid_ry);
    
    cell_histogram_kernel<<<dimGrid, dimBlock>>>(*inds_and_weights,*output,cell_size,offset_y, offset_x, max_bin);

    cudaThreadSynchronize();
}


/*obsolete helper kernels. leaving for reference delete at will.
__global__ void copy_kernel(DeviceMatrix3D outmat,
                           DeviceMatrix newmat){
    const int pix_y = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    const int pix_x = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    if ( (pix_y < newmat.height) && (pix_x < newmat.width) ){
        float new_val = *getPtr(newmat, pix_y, pix_x);
        *getPtr(outmat, 0, pix_y, pix_x) = new_val;
        *getPtr(outmat, 1, pix_y, pix_x) = 0;
    }
    __syncthreads();
}

__global__ void min_kernel(DeviceMatrix3D outmat,
                           DeviceMatrix newmat,
                           const int new_label){

    const int pix_y = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    const int pix_x = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    if ( (pix_y < newmat.height) && (pix_x < newmat.width) ){
        float new_val = *getPtr(newmat, pix_y, pix_x);
        float out_val = *getPtr(outmat, 0, pix_y, pix_x);
        if (new_val < out_val){
            *getPtr(outmat, 0, pix_y, pix_x) = new_val;
            *getPtr(outmat, 1, pix_y, pix_x) = new_label;
        }
    }
    __syncthreads();

}

__global__ void max_kernel(DeviceMatrix3D outmat,
                           DeviceMatrix newmat,
                           const int new_label){

    const int pix_y = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    const int pix_x = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    if ( (pix_y < newmat.height) && (pix_x < newmat.width) ){
        float new_val = *getPtr(newmat, pix_y, pix_x);
        float out_val = *getPtr(outmat, 0, pix_y, pix_x);
        if (new_val > out_val){
            *getPtr(outmat, 0, pix_y, pix_x) = new_val;
            *getPtr(outmat, 1, pix_y, pix_x) = new_label;
        }
    }
    __syncthreads();

}

*/

/* obsolete euclid kernel called by the dist_filter
__global__ void blockwise_euclid_kernel(DeviceMatrix frame,
                                        DeviceMatrix3D filter_bank,
                                        DeviceMatrix output,
                                        const int filter_ind,
                                        const int start_y, const int start_x,
                                            const int end_y,   const int end_x)
{

    // Load the filter
    __shared__ float filter_cache[BLOCK_SIZE][BLOCK_SIZE];

    if ((threadIdx.y < filter_bank.dim_y) &&
        (threadIdx.x < filter_bank.dim_x) )
    {
        filter_cache[threadIdx.y][threadIdx.x] = *getPtr(filter_bank,
                                                         filter_ind,
                                                         threadIdx.y,
                                                         threadIdx.x);
    }

    __syncthreads();

    const int pix_y = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    const int pix_x = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    float out_val = 0;
    float temp_val = 0;
    if ((pix_y >= start_y)&&(pix_y <= end_y)&&
        (pix_x >= start_x)&&(pix_x <= end_x)){
        for (int i = -start_y; i<=start_y; i++) {
            for (int j = -start_x; j<=start_x; j++){
                temp_val = *getPtr(frame, pix_y+i, pix_x+j) - filter_cache[i+start_y][j+start_x];
                out_val += temp_val * temp_val;
                __syncthreads();
            }
        }
        *getPtr(output, pix_y, pix_x) = out_val;
    }

}
*/

/*obsolete cosine kernel called by dist_filter
__global__ void blockwise_cosine_kernel(DeviceMatrix frame,
                                        DeviceMatrix3D filter_bank,
                                        DeviceMatrix output,
                                        const int filter_ind,
                                        const int start_y, const int start_x,
                                        const int end_y,   const int end_x)
{

    // Load the filter
    __shared__ float filter_cache[BLOCK_SIZE][BLOCK_SIZE];

    if ((threadIdx.y < filter_bank.dim_y) &&
        (threadIdx.x < filter_bank.dim_x) )
    {
        filter_cache[threadIdx.y][threadIdx.x] = *getPtr(filter_bank,
                                                         filter_ind,
                                                         threadIdx.y,
                                                         threadIdx.x);
    }

    __syncthreads();

    const int pix_y = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    const int pix_x = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    float out_val = 0;
    //float sq_sum = 0;
    float pix_val = 0;
    if ((pix_y >= start_y)&&(pix_y <= end_y)&&
        (pix_x >= start_x)&&(pix_x <= end_x)){
        for (int i = -start_y; i<=start_y; i++) {
            for (int j = -start_x; j<=start_x; j++){
                pix_val = *getPtr(frame, pix_y+i, pix_x+j);
                out_val +=  pix_val * filter_cache[i+start_y][j+start_x];
                //sq_sum += pix_val * pix_val;
                __syncthreads();
            }
        }
        *getPtr(output, pix_y, pix_x) = out_val;// / sqrt(sq_sum);
    }

}
*/
/* obsoloete kernel for setting the edges of an image to -1
__global__ void clip_edges( const DeviceMatrix3D output,
                            const int start_y, const int start_x,
                            const int end_y, const int end_x)
{
    const int pix_y = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    const int pix_x = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    if ((pix_y < output.dim_y) && (pix_x < output.dim_x)){
        if ((pix_y < start_y) || (pix_x < start_x) ){
            *getPtr(output, 0, pix_y, pix_x) = -1;
            *getPtr(output, 1, pix_y, pix_x) = -1;
        }
        else if ((pix_y > end_y) || (pix_x > end_x) ) {
            *getPtr(output, 0, pix_y, pix_x) = -1;
            *getPtr(output, 1, pix_y, pix_x) = -1;
        }
    }

    __syncthreads();
}
*/



/* Obsolete implementation
void dist_filter( const DeviceMatrix* frame,
                  const int dim_t, const int dim_y, const int dim_x,
                  const DeviceMatrix3D* filter_bank,
                  DeviceMatrix3D* output,
                  const int optype)
{
    int half_filter_height = filter_bank->dim_y / 2;
    int half_filter_width  = filter_bank->dim_x / 2;

    int start_y = half_filter_height;
    int start_x = half_filter_width;

    int end_y = frame->height - half_filter_height - 1;
    int end_x = frame->width  - half_filter_width - 1;

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    int grid_ry = (frame->height-1) / dimBlock.y + 1;
    int grid_cx = (frame->width -1) / dimBlock.x + 1;

    dim3 dimGrid(grid_cx, grid_ry);

    DeviceMatrix::Ptr temp_result = makeDeviceMatrix(frame->height, frame->width);

    for (int filter_ind = 0; filter_ind < filter_bank->dim_t; filter_ind++){
        temp_result->zero();
        if (optype == FF_OPTYPE_EUCLIDEAN){
            blockwise_euclid_kernel<<<dimGrid, dimBlock>>>(*frame,
                                                       *filter_bank,
                                                       *temp_result,
                                                       filter_ind,
                                                       start_y, start_x,
                                                       end_y, end_x);
        }
        else if (optype == FF_OPTYPE_COSINE){
            blockwise_cosine_kernel<<<dimGrid, dimBlock>>>(*frame,
                                                       *filter_bank,
                                                       *temp_result,
                                                       filter_ind,
                                                       start_y, start_x,
                                                       end_y, end_x);
        }
        if (filter_ind == 0){
            copy_kernel<<<dimGrid, dimBlock>>>(*output, *temp_result);
        }
        else {
            if (optype == FF_OPTYPE_EUCLIDEAN){
                min_kernel<<<dimGrid, dimBlock>>>(*output, *temp_result, filter_ind);
            }
            else if (optype == FF_OPTYPE_COSINE){
                max_kernel<<<dimGrid, dimBlock>>>(*output, *temp_result, filter_ind);
            }
        }
    }

    clip_edges<<<dimGrid, dimBlock>>>(*output, start_y, start_x, end_y, end_x);
}
*/
