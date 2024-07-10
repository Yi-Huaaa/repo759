#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "gamer.cuh"
#define rev_idx(dir_n, idx) abs(dir_n-1-idx)


__global__ void record_path () {
}

/**
    prefix_sum (also prefix_min)
    Algorithm: Harris-Sengupta-Owen
    Note: 
    (1) #thread = len/2

    * Todo: 
    (1) padding (for those n != pow(2))

    * Todo: optimization: 
    (1) warp level synchronization 
    (2) (ai,bi) = ((offset*(2*t_idx+1)-1), offset*(2*t_idx+2)-1) --> bad data alocality: (1) coalesced and (2) properly aligned
    (3) __device__ __forceinline__/__inline__
    (4) fast version for compared Big/small number for prefix min ---> PTX
    
    * (Me) Done optimization: 
    (1) load/store input/output into shared memory, original program did not consider the (1) coalesced and (2) properly aligned

    * (slide) Done optimization: 
    (1) reduce half number of threads 
*/
__global__ void prefix_sum (int *output, int *input, int n) {
    extern volatile __shared__ int temp[];  // allocated on invocation
    int t_idx = threadIdx.x;
    int b_idx = blockDim.x;
    int offset = 1;
    temp[t_idx] = input[t_idx]; // load input into shared memory
    temp[b_idx + t_idx] = input[b_idx + t_idx];

    for (int d = n >> 1; d > 0; d >>= 1){ // build sum in place up the tree
        __syncthreads();
        if (t_idx < d) {
            int ai = offset*(2*t_idx+1)-1;
            int bi = offset*(2*t_idx+2)-1;
             temp[bi] += temp[ai];
        }
        offset <<= 1; // multiply by 2 implemented as bitwise operation
    }
    if (t_idx == 0) {
        temp[n - 1] = 0;
    } // clear the last element
    for (int d = 1; d < n; d *= 2){ // traverse down tree & build scan
        offset >>= 1;
        __syncthreads();
        if (t_idx < d){
            int ai = offset*(2*t_idx+1)-1;
            int bi = offset*(2*t_idx+2)-1;
            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();

    if (t_idx == 0) {
        output[2*b_idx-1] = temp[2*b_idx-1]+input[2*b_idx-1];  // the last element
    } else {
        output[t_idx-1] = temp[t_idx]; // write results to device memory
    }
    output[b_idx+t_idx-1] = temp[b_idx+t_idx];
}

__global__ void prefix_min (int *output, int *input, int n) {
    extern volatile __shared__ int temp[];  // allocated on invocation
    int t_idx = threadIdx.x;
    int b_idx = blockDim.x;
    int offset = 1;
    temp[t_idx] = input[t_idx]; // load input into shared memory
    temp[b_idx + t_idx] = input[b_idx + t_idx];

    for (int d = n >> 1; d > 0; d >>= 1){ // build sum in place up the tree
        __syncthreads();
        if (t_idx < d) {
            int ai = offset*(2*t_idx+1)-1;
            int bi = offset*(2*t_idx+2)-1;
            temp[bi] = (temp[ai] < temp[bi]) ? (temp[ai]) : (temp[bi]);
        }
        offset <<= 1; // multiply by 2 implemented as bitwise operation
    }
    if (t_idx == 0) {
        temp[n - 1] = MAX_INF;
    } // clear the last element
    for (int d = 1; d < n; d *= 2){ // traverse down tree & build scan
        offset >>= 1;
        __syncthreads();
        if (t_idx < d){
            int ai = offset*(2*t_idx+1)-1;
            int bi = offset*(2*t_idx+2)-1;
            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] = (t < temp[bi]) ? (t) : (temp[bi]);
        }
    }
    __syncthreads();

    if (t_idx == 0) {
        output[2*b_idx-1] = temp[2*b_idx-1]+input[2*b_idx-1];  // the last element
        output[2*b_idx-1] = (temp[2*b_idx-1] < input[2*b_idx-1]) ? (temp[2*b_idx-1]) : (input[2*b_idx-1]);
    } else {
        output[t_idx-1] = temp[t_idx]; // write results to device memory
    }
    output[b_idx+t_idx-1] = temp[b_idx+t_idx];
}

/**
    d_min_s (also t_add_s)
    Algorithm: X
    Note: 
    (1) #thread = len

    * Todo: optimization: 
    (1) merge the function into other function --> too garbage wasting for kernel launching time    

*/

__device__ __forceinline__ void prefix_min_device (int *output, int *input, volatile int *temp, int n, int t_idx, int b_idx, int dir) {
    // note: for negative direction of `prefix_min_device`, both R/W need to reverse
    // load into shared memory
    int r_idx, w_idx;
    r_idx = t_idx; 
    w_idx = t_idx;
    if (dir == 1) r_idx = rev_idx(n,r_idx); 
    temp[w_idx] = input[r_idx];

    r_idx = t_idx+(n>>1); 
    w_idx = w_idx+(n>>1);
    if (dir == 1) r_idx = rev_idx(n,r_idx);
    temp[w_idx] = input[r_idx]; 


    // prefix_min 
    int offset = 1;
    for (int d = n >> 1; d > 0; d >>= 1){ // build sum in place up the tree
        __syncthreads();
        if (t_idx < d) {
            int ai = offset*(2*t_idx+1)-1;
            int bi = offset*(2*t_idx+2)-1;
            temp[bi] = (temp[ai] < temp[bi]) ? (temp[ai]) : (temp[bi]);
        }
        offset <<= 1; // multiply by 2 implemented as bitwise operation
    }
    if (t_idx == 0) {
        temp[n - 1] = MAX_INF;
    } // clear the last element
    for (int d = 1; d < n; d *= 2){ // traverse down tree & build scan
        offset >>= 1;
        __syncthreads();
        if (t_idx < d){
            int ai = offset*(2*t_idx+1)-1;
            int bi = offset*(2*t_idx+2)-1;
            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] = (t < temp[bi]) ? (t) : (temp[bi]);
        }
    }
    __syncthreads();
    
    // store back to global memory
    // need to shift left back for 1 space 
    if (t_idx == 0) {
        int pre_element = temp[n-1];
        r_idx = (dir == 1) ? (0) : (n-1);
        int ori_element = input[r_idx];
        w_idx = n-1; if (dir == 1) w_idx = rev_idx(n,w_idx); 
        output[w_idx] = (pre_element > ori_element) ? (ori_element) : (pre_element);
    } else {
        w_idx = t_idx-1; if (dir == 1) w_idx = rev_idx(n,w_idx); 
        output[w_idx] = temp[t_idx];
    }
    w_idx = b_idx+t_idx-1; if (dir == 1) w_idx = rev_idx(n,w_idx); 
    output[w_idx] = temp[b_idx+t_idx];
}
__global__ void gamer_x (int *gcell_3D_devc, int *weight_x_post, int *weight_x_nega, int x_dir, int y_dir, int z_dir, int *s_devc, int *t_devc, int *path_trace_devc, int BATCH) {
    extern volatile __shared__ int temp[];  // allocated on invocation
    int z = 0;
    // todo: reduce redundant __syncthreads();
    int t_idx = threadIdx.x;
    // idx_accum: used for update the index accumulation for different blocks
    int idx_accum = blockIdx.x*x_dir;
    int idx, tmp;
    
    // STEP 1. s(i) = sigma(c(i)) ---> prefix_sum -> already compute outside the kerenl
    // prepare weight index array
    int *w_array[2];
    w_array[0] = weight_x_post;
    w_array[1] = weight_x_nega;
    // (Positive, Negative) = (dir = 0, dir = 1)
    for (auto dir = 0; dir < 2; dir ++) {
        int *gcell_arr, *w_array_dir;
        for (auto batch = 0; batch < (BATCH); batch++) {
            unsigned int idx_accum_x = IDX23D((x_dir),(y_dir),(z_dir),0,((y_dir/BATCH)*batch),z);

            // update index accumulation 
            gcell_arr = gcell_3D_devc + idx_accum_x;
            w_array_dir = w_array[dir] + idx_accum_x;
                    
            // STEP 2. temp(i) = d(i) - s(i) ---> parallel minus
            idx = idx_accum+t_idx;
            t_devc[idx] = gcell_arr[idx] - w_array_dir[idx]; 
            idx += (x_dir>>1);
            t_devc[idx] = gcell_arr[idx] - w_array_dir[idx]; 
            __syncthreads();

            // STEP 3. temp (i) = min (temp(j)), for all j <= i ---> prefix_min, use s_devc to store
            prefix_min_device(s_devc+(idx_accum), t_devc+(idx_accum), temp+(idx_accum), x_dir, t_idx, blockDim.x, dir);
            __syncthreads();

            // STEP 4. d(i) = temp(i) + s(i) ---> parallel addition
            idx = idx_accum+t_idx;
            tmp = s_devc[idx] + w_array_dir[idx];
            if (gcell_arr[idx] != tmp) {
                path_trace_devc[idx] = (dir+1); // x+: 1, x-: 2
                gcell_arr[idx] = tmp;
            }

            idx += (x_dir>>1);
            tmp = s_devc[idx] + w_array_dir[idx];
            if (gcell_arr[idx] != tmp) {
                path_trace_devc[idx] = (dir+1); // x+: 1, x-: 2
                gcell_arr[idx] = tmp;
            }        
            __syncthreads(); // todo: only need sync warp
            // printf("gamer_x\n");
        }
    }


}
__global__ void gamer_y (int *gcell_3D_devc, int *weight_y_post, int *weight_y_nega, int x_dir, int y_dir, int z_dir, int *s_devc, int *t_devc, int *path_trace_devc, int BATCH) {
    extern volatile __shared__ int temp[];  // allocated on invocation
    int z = 0;
    // todo: reduce redundant __syncthreads();
    int t_idx = threadIdx.x;
    // idx_accum_y_blk, idx_accum_x_blk: used for update the index accumulation for different blocks
    int idx_accum_y_blk = ((blockIdx.x)/x_dir)*(x_dir*y_dir) + (blockIdx.x%x_dir); // index 要改
    int idx_accum_x_blk = blockIdx.x*x_dir;
    int r_idx, w_idx, tmp;

    // STEP 1. s(i) = sigma(c(i)) ---> prefix_sum -> already compute outside the kerenl
    // prepare weight index array
    int *w_array[2];
    w_array[0] = weight_y_post;
    w_array[1] = weight_y_nega;
    // Positive, Negative  
    for (auto dir = 0; dir < 2; dir ++) {
        int *gcell_arr, *w_array_dir;
        for (auto batch = 0; batch < (BATCH); batch++) {
            // idx_accum_x and idx_accum_y: used to update the index of tile
            unsigned int idx_accum_x = IDX23D((x_dir),(y_dir),(z_dir),0,((y_dir/BATCH)*batch),z);
            unsigned int idx_accum_y = IDX23D((x_dir),(y_dir),(z_dir),((x_dir/BATCH)*batch),0,z);
            // update index accumulation
            gcell_arr = gcell_3D_devc + idx_accum_y;
            w_array_dir = w_array[dir] + idx_accum_x;

            // STEP 2. temp(i) = d(i) - s(i) ---> parallel minus
            // r_idx: here means the vertical reading index
            r_idx = idx_accum_y_blk+(t_idx*x_dir); 
            w_idx = idx_accum_x_blk+t_idx;
            t_devc[w_idx] = gcell_arr[r_idx] - w_array_dir[w_idx]; 
            r_idx += (y_dir>>1)*x_dir; 
            w_idx += (y_dir>>1);
            t_devc[w_idx] = gcell_arr[r_idx] - w_array_dir[w_idx]; 
            __syncthreads();
            
            // STEP 3. temp (i) = min (temp(j)), for all j <= i ---> prefix_min, use s_devc to store
            prefix_min_device(s_devc+(idx_accum_x_blk), t_devc+(idx_accum_x_blk), temp+(idx_accum_x_blk), y_dir, t_idx, blockDim.x, dir);
            __syncthreads();

            // STEP 4. d(i) = temp(i) + s(i) ---> parallel addition
            r_idx = idx_accum_x_blk+t_idx;
            w_idx = idx_accum_y_blk+(t_idx*x_dir); 
            tmp = s_devc[r_idx] + w_array_dir[r_idx];
            if (gcell_arr[w_idx] != tmp) {
                path_trace_devc[w_idx] = (dir+3); // y+: 3, y-: 4
                gcell_arr[w_idx] = tmp;
            }
            // w_idx += (x_dir>>1);
            r_idx += (y_dir>>1);
            w_idx += (y_dir>>1)*x_dir; 
            tmp = s_devc[r_idx] + w_array_dir[r_idx];
            if (gcell_arr[w_idx] != tmp) {
                path_trace_devc[w_idx] = (dir+3); // y+: 3, y-: 4
                gcell_arr[w_idx] = tmp;
            }
            __syncthreads(); // todo: only need sync warp
        }
    }
    // printf("Inside, gamer_y\n");
}
__global__ void gamer_z (int *gcell_3D_devc, int *weight_x_post, int *weight_x_nega, int x_dir, int y_dir, int z_dir, int *s_devc, int *t_devc, int *path_trace_devc, int BATCH) {
}

__device__ __forceinline__ void prefix_sum_device (int *output, int *input, volatile int *temp, int n, int t_idx, int b_idx) {
    int offset = 1;
    for (int d = n >> 1; d > 0; d >>= 1){ // build sum in place up the tree
        __syncthreads();
        if (t_idx < d) {
            int ai = offset*(2*t_idx+1)-1;
            int bi = offset*(2*t_idx+2)-1;
             temp[bi] += temp[ai];
        }
        offset <<= 1; // multiply by 2 implemented as bitwise operation
    }
    if (t_idx == 0) {
        temp[n - 1] = 0;
    } // clear the last element
    for (int d = 1; d < n; d *= 2){ // traverse down tree & build scan
        offset >>= 1;
        __syncthreads();
        if (t_idx < d){
            int ai = offset*(2*t_idx+1)-1;
            int bi = offset*(2*t_idx+2)-1;
            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();

    output[t_idx] = temp[t_idx];
    output[b_idx+t_idx] = temp[b_idx+t_idx];
}

__global__ void compute_weight (int *output_pos, int *output_neg, int *input, int dir_n, int x_dir, int y_dir, int z_dir, int BATCH) {
    extern volatile __shared__ int temp[];  // allocated on invocation
    int z = 0;
    int *ot_pos, *ot_neg, *inp;
    for (auto batch = 0; batch < (BATCH); batch++) {
        unsigned int idx_accum_x = IDX23D((x_dir),(y_dir),(z_dir),0,((y_dir/BATCH)*batch),z);

        // update index accumulation
        ot_pos = output_pos + idx_accum_x;
        ot_neg = output_neg + idx_accum_x;
        inp = input + idx_accum_x;

        // tiling compute_weight
        int t_idx = threadIdx.x;
        int b_idx = blockDim.x; // number of team
        int r_idx, w_idx;
        int idx_accum = blockIdx.x*dir_n;
    
        // positive direction
        // load inp into shared memory 
        r_idx = idx_accum+t_idx; 
        w_idx = idx_accum+t_idx;
        temp[w_idx] = inp[r_idx];
        r_idx += (dir_n>>1); 
        w_idx += (dir_n>>1);
        temp[w_idx] = inp[r_idx]; 
        prefix_sum_device(ot_pos+(idx_accum), inp+(idx_accum), temp+(idx_accum), dir_n, t_idx, b_idx);
        __syncthreads();
    
        // negative direction
        int lane_max = ot_pos[idx_accum+dir_n-1];
        r_idx = idx_accum+t_idx; 
        w_idx = idx_accum+t_idx;    
        ot_neg[w_idx] = lane_max - ot_pos[r_idx];
        r_idx += (dir_n>>1); 
        w_idx += (dir_n>>1);
        ot_neg[w_idx] = lane_max - ot_pos[r_idx];
    }
}

__global__ void find_shortest_pin_set_source (int *gcell_3D_devc, pins *pin_devc, int *set_pin_record, int x_dir, int y_dir, int z_dir, int n_pin, int p_out, int *path_trace_devc) {
    // find the pin with shortest path
    // todo: here now is brutal force, it can be optimized
    int set_pin = 0; // shortest pin
    int tmp_path_distance = MAX_INF;
    int z = 0;
    int s_idx = IDX23D((x_dir),(y_dir),(z_dir),pin_devc[set_pin].x,pin_devc[set_pin].y,z);
    int tmp_idx = 0;

    if (p_out == 0) {
        tmp_idx = IDX23D((x_dir),(y_dir),(z_dir),pin_devc[0].x,pin_devc[0].y,z);
        gcell_3D_devc[tmp_idx] = 0;
        path_trace_devc[s_idx] = 0;        
        set_pin_record[0] = 1;
        // printf("First pin, p_out = %d, connected pin = %d\n", p_out, set_pin);
    } else {
        for (auto p = 1; p < n_pin; p++) {;
            tmp_idx = IDX23D((x_dir),(y_dir),(z_dir),pin_devc[p].x,pin_devc[p].y,z);
            // printf("\tp = %d, pin_devc[p].x = %d, pin_devc[p].y = %d, tmp_idx = %d\n", p, pin_devc[p].x,pin_devc[p].y, tmp_idx);
            if ((gcell_3D_devc[tmp_idx] < tmp_path_distance) && (set_pin_record[p] == 0)) { 
                set_pin = p;
                tmp_path_distance = gcell_3D_devc[tmp_idx];
                s_idx = tmp_idx;
                // printf("insides, s_idx = %d\n", s_idx);
            }
        }
        // // set the pin's distance = 0
        gcell_3D_devc[s_idx] = 0;
        set_pin_record[set_pin] = 1;
        // printf("p_out = %d, connected pin = %d (set_pin_record[%d] = %d, gcell_3D_devc[%d] = %d)\n\n", p_out, set_pin, set_pin, set_pin_record[set_pin], s_idx, gcell_3D_devc[s_idx]);
        // printf("-----\n");
    } 
    // set all the pins on the shortest path's distance = 0
    tmp_idx = s_idx;
    int end_val = 1;
    // todo: add z+, z- direction to the cases
    while (end_val) {
        if (path_trace_devc[tmp_idx] == 1) { // from x+
            gcell_3D_devc[tmp_idx] = 0;
            path_trace_devc[tmp_idx] = 0;
            tmp_idx -= 1;
            
        } else if (path_trace_devc[tmp_idx] == 2) {  // from x-
            gcell_3D_devc[tmp_idx] = 0;
            path_trace_devc[tmp_idx] = 0;
            tmp_idx += 1;

        } else if (path_trace_devc[tmp_idx] == 3) {  // from y+
            gcell_3D_devc[tmp_idx] = 0;
            path_trace_devc[tmp_idx] = 0;
            tmp_idx -= x_dir;

        } else if (path_trace_devc[tmp_idx] == 4) {  // from y-
            gcell_3D_devc[tmp_idx] = 0;
            path_trace_devc[tmp_idx] = 0;
            tmp_idx += x_dir;

        } else { 
            end_val = 0; // arrive a source pin
        }
        
    }
}

