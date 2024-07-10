#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "stencil.cuh"

__global__ void stencil_kernel(const float* image, const float* mask, float* output, unsigned int n, unsigned int R) {
    int int_n = int(n);
    int int_R = int(R);
    int t_idx = threadIdx.x;
    int tpb = blockDim.x;
    int g_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_of_blocks = (int_n - 1) / tpb + 1;
    int is_last_block = (blockIdx.x == num_of_blocks - 1);

    extern __shared__ float shared_mem[];
    float *shared_image, *shared_output, *shared_mask;
    shared_image = shared_mem;
    shared_output = &shared_mem[tpb + 2*int_R];
    shared_mask = &shared_mem[tpb + 2*int_R + tpb];

    // load image (central)
    if (g_idx < int_n) {
        shared_image[int_R + t_idx] = image[g_idx];
    }

    // load image (left side)
    if (t_idx < int_R) {
        shared_image[t_idx] = (g_idx - int_R >= 0) ? (image[g_idx - int_R]) : (1);
    }
    
    // load image (right side)
    if (t_idx < int_R) {
        int central_size = 0;
        if (is_last_block && (int_n % tpb)) {
            central_size = int_n % tpb;
        } else {
            central_size = tpb;
        }
        shared_image[int_R + central_size + t_idx] = (g_idx + central_size < int_n) ? (image[g_idx + central_size]) : (1);
    }
    
    // load mask
    // NOTE: it's guaranteed that tpb >= 2 * R + 1, so each elements in the mask will be handled by one thread
    if (t_idx < 2 * int_R + 1) {
        shared_mask[t_idx] = mask[t_idx];
    }

    // set output to 0
    if (g_idx < int_n) {
        shared_output[t_idx] = 0.0;
    }

    __syncthreads();
    
    // convolution
    if (g_idx < int_n) {
        for (int j = -int_R; j <= int_R; j++) {
            float image_val = shared_image[int_R + t_idx + j];
            shared_output[t_idx] += image_val * shared_mask[j + int_R];
        }
        output[g_idx] = shared_output[t_idx];
    }
}

__host__ void stencil(const float* image, const float* mask, float* output, unsigned int n, unsigned int R, unsigned int threads_per_block) {
    int tpb = threads_per_block;
    int smem_sz = (tpb + 2 * R) + (tpb) + (2 * R + 1);
    int num_of_blocks = (n - 1) / tpb + 1;
    stencil_kernel <<< num_of_blocks, tpb, (smem_sz * sizeof(float)) >>> (image, mask, output, n, R);
}