#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <assert.h>
#include <cuda_runtime.h>

#include "scan.cuh"

#define check

// ERROR: ./task2 32768 510

void  check_cal_answer (const float *input, float *output, int out_index, int total_data) {
    float ans = (0+total_data-1)*total_data/2;
    // printf("ans = %.0lf\n", ans);
    // // assert(output[out_index] == ans);
    // if (output[out_index] != ans)
    //     printf("out_index = %d\n", out_index);
}


__global__ void copy_add_data (float *input, float *tmp, int tpb) {
    int index = (blockIdx.x+1)*tpb-1; // take the last number of each block
    tmp[blockIdx.x] = input[index];
}

// the second layer of scan, coopertating by CPU and GPU
__global__ void accum_val (float *g_odata, int total_data, unsigned int offset_block, float *tmp) {
    int t_idx = threadIdx.x;
    
    // all the data index before "blockDim.x*offset_block"'s work have been done
    int start_idx = blockDim.x*(offset_block+blockIdx.x);
    int read_idx = start_idx + t_idx; // the data that the thread is going to tackle with, also the WB index
    
    int added_block = (blockIdx.x);
    float add_val = tmp[added_block];

    if (read_idx < total_data) {
        g_odata[read_idx] += add_val;
    }
}

// Ref: lecture 14
__global__ void scan_kernel (const float *g_idata, float *g_odata, int n, int total_data) {
    extern volatile __shared__ float smem[]; // allocated on invocation
    int t_idx = threadIdx.x;
    int pout = 0, pin = 1;
    // load input into shared memory. note: load the "black" line data 
    int start_idx = blockIdx.x*n;
    int read_idx = start_idx + t_idx;
    
    // judge - out of bound
    smem[t_idx] = (read_idx < total_data) ? (g_idata[read_idx]) : (0);
    __syncthreads();

    // use two shared memory buffer 
    for (int offset = 1; offset < n; offset *= 2) {
        // pout = 1, pin = 0 at the first round
        pout = 1 - pout; // swap double buffer indices
        pin = 1 - pout;

        if (t_idx >= offset) {
            smem[pout*n+t_idx] = smem[pin*n+t_idx] + smem[pin*n+t_idx - offset];
        } else {
            smem[pout*n+t_idx] = smem[pin*n+t_idx];
        }
        __syncthreads(); // I need this here before I start next iteration
    }

    int out_idx = start_idx + t_idx;   
    if ((out_idx) < total_data) {
        g_odata[out_idx] = smem[pout*n+t_idx]; // write output
    }
}

// Performs an *inclusive scan* on the array input and writes the results to the array output.
// The scan should be computed by making calls to your kernel hillis_steele with
// threads_per_block threads per block in a 1D configuration.
// input and output are arrays of length n allocated as managed memory.
//
// Assumptions:
// - n <= threads_per_block * threads_per_block

__host__ void scan(const float* input, float* output, unsigned int n, unsigned int threads_per_block) {
    unsigned int smem_sz = 2*threads_per_block*sizeof(float);
    int int_n = int(n);
    int tpb = int(threads_per_block);

    unsigned int num_per_block = (int_n > tpb) ? ((int_n+tpb-1)/tpb) : (1); // ceiling
    int deal_data = tpb; // we can only deal with tpb data at a time    

    // printf("num_per_block = %u\n", num_per_block);
    scan_kernel <<< num_per_block, threads_per_block, smem_sz >>> (input, output, deal_data, int_n);

    // copy addition data to tmp memory, since the input array is const, therefore we can only use a new array for the second layer's perfix sum
    float *tmp;
    cudaMallocManaged(&tmp, num_per_block*sizeof(float));
    

    for (unsigned int offset = 1; offset < num_per_block; offset *= 2) {
        copy_add_data <<< (num_per_block-1) , 1 >>> (output, tmp, tpb); // copy addition data to tmp memory

        unsigned int tmp_num_per_block = num_per_block - offset;
        accum_val <<< tmp_num_per_block, threads_per_block >>> (output, int_n, offset, tmp);
    }  
}