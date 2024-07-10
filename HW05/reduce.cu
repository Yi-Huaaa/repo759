#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <assert.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "reduce.cuh"
// #define check_load

int pad_exp_2 (unsigned int threads_per_block) {
    int tpb = 1; 
    unsigned int exp2_arr [11] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
    for (int i = 0; i < 11; i++) {
        if (exp2_arr[i] >= threads_per_block) {
            tpb = int(exp2_arr[i]);
            break;
        }
    }
    return tpb;
}

__device__ void print_arr (float *arr, int len) {
    for (auto i = 0; i < len; i++) {
        printf("%.2lf, ", arr[i]);
    } printf("]\n");
}

__device__ void load_data (float *smem, float *g_idata, int tpb, int process_gdata, int t_idx, int n) {
    float pre_data = (process_gdata < n) ? (g_idata[process_gdata]) : (0);
    float post_data = ((process_gdata+tpb) < n) ? (g_idata[process_gdata+tpb]) : (0);

    smem[t_idx] = pre_data+post_data;
    // printf("t_idx = %d (b_idx = %d), pre_data = %.2lf, post_data = %.2lf, smem[%d] = %.2lf\n", t_idx, blockIdx.x, pre_data, post_data, t_idx, smem[t_idx]);
}


__global__ void reduce_kernel(float *g_idata, float *g_odata, unsigned int n) {
    extern __shared__ float smem[];

    int t_idx = threadIdx.x;// b_idx = blockIdx.x;
    int tpb = blockDim.x; // threads_per_block
    int process_gdata = ((2*tpb*blockIdx.x)) + threadIdx.x; // one block can deal with '((2*tpb*blockIdx.x))' data
    int int_n = int(n);
    
    // reduction 4 version: First Add During Load
    // totally needed to load 2*threads_per_block elements before 'First Add During Load,' out of boundary -> set as 0
    load_data (smem, g_idata, tpb, process_gdata, t_idx, int_n);
    __syncthreads();

#ifdef check_load
    if (t_idx == 0 && blockIdx.x == 0) {
        printf("check_load, smem = [");
        print_arr(smem, 2*tpb);
    }
#endif 

    // reduction 3 version: Sequential Addressing
    // can start with 'tpb' is due to setting the out of bound smem == 0
    for (int s = tpb/2; s > 0; s /= 2) {
        // if (t_idx == 0 && blockIdx.x == 0){
        //     printf("smem[%d] = %.2lf, smem[%d] = %.2lf\n", t_idx, smem[t_idx], t_idx + s, smem[t_idx + s]);
        // }
        if (t_idx < s) {
            smem[t_idx] += smem[t_idx + s];
        }
        // if (t_idx == 0 && blockIdx.x == 0){
        //     printf("post: smem[%d] = %.2lf, smem[%d] = %.2lf\n", t_idx, smem[t_idx], t_idx + s, smem[t_idx + s]);
        // }
        __syncthreads();

    }
    
    // write back to global memory
    if (t_idx == 0) {
        g_odata[blockIdx.x] = smem[0];
        g_idata[blockIdx.x] = smem[0]; // since the final answer needed to be stored in the input array 
        // if (t_idx == 0 && blockIdx.x == 0){
        //     printf("g_odata[blockIdx.x] = %.2lf\n", g_odata[blockIdx.x]);
        // }
    }
}

// TODO: the function should end in a call to cudaDeviceSynchronize for timing purposes
__host__ void reduce(float **input, float **output, unsigned int N, unsigned int threads_per_block) {
    // Note that 100,000= 390*256 + 160, therefore [(N+255)/256]=391 blocks needed
    int tpb = pad_exp_2(threads_per_block);
    
    unsigned int deal_num_per_block = 2*tpb; // since one thread processes 2 number simultaneously
    unsigned int num_block = (N > deal_num_per_block) ? (((N+deal_num_per_block-1)/deal_num_per_block)) : (1);
    unsigned int smem_sz = deal_num_per_block*sizeof(float);
    
    float *in1 = *input;
    float *in2 = *output;
    float *tmp;
    
    // printf("num_block = %d\n", num_block);
    if (num_block > 1) {
        unsigned int deal_num_data = N;
        
        while(num_block > 1) {
            // printf("num_block = %u, deal_num_data = %u\n",num_block, deal_num_data);
            reduce_kernel <<< num_block, tpb, smem_sz >>> (in1, in2, deal_num_data);
            deal_num_data = num_block;
            num_block = (deal_num_data > deal_num_per_block) ? (((deal_num_data+deal_num_per_block-1)/deal_num_per_block)) : (1);
            // exchange *input, *output pointer
            tmp = in1;
            in1 = in2;
            in2 = tmp;
        }
        
        // last round
        // printf("last round, num_block = %u\n", num_block);
        reduce_kernel <<< num_block, tpb, smem_sz >>> (in1, in2, deal_num_data);
    } else {
        reduce_kernel <<< num_block, tpb, smem_sz >>> (in1, in2, N);
    }
}
