#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "matmul.cuh"
// #define oneblock

__device__ void print_arr_gpu (const float *arr, int len) {
    for (auto i = 0; i < len; i++)
        printf("%.2lf ", arr[i]);
    printf("]\n");
}

// __global__ void matmul_kernel(const float* A, const float* B, float* C, size_t n) {
//     int threads_per_block = blockDim.x;
//     int thread = threadIdx.x;
//     int int_n = int(n);
// #ifdef oneblock
//     int iteration = int_n/threads_per_block; // implies (n > threads_per_block)
// #endif 

//     if (int_n >= threads_per_block) { // large cases
//         int row = thread + threads_per_block*blockIdx.x; 
        
//         float tmp = 0.;
//         for(int j = 0; j < int_n; j++) {
//             for(int k = 0; k < int_n; k++){
//                 tmp += A[row*int_n+k]*B[k*int_n+j];   
//             }
//             C[row*int_n+j] = tmp; tmp = 0.;
//         }
//     } else { // small cases
//         if (thread < n) {
//             int row = thread;

//             float tmp = 0.;
//             for(int j = 0; j < int_n; j++) {
//                 for(int k = 0; k < int_n; k++){
//                     tmp += A[row*int_n+k]*B[k*int_n+j];   
//                 }
//                 C[row*int_n+j] = tmp; tmp = 0.;
//             }
//         } 
//     }
// }

// void matmul(const float* A, const float* B, float* C, size_t n, unsigned int threads_per_block) {
// #ifdef oneblock
//     matmul_kernel <<< 1, threads_per_block >>> (A, B, C, n);
// #else 
//     // todo: 這裡的 num_per_blocks 要處理除不盡的狀況，因為 n 不一定是 exp of 2
//     int num_per_blocks = (n > threads_per_block) ? (int(n/threads_per_block)) : (1); 
//     matmul_kernel <<< num_per_blocks, threads_per_block >>> (A, B, C, n);
// #endif
// }


__global__ void matmul_kernel(const float* A, const float* B, float* C, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = int(n);
    
    if (idx >= N * N) {
        return;
    }

    int r = idx / N, c = idx % N;
    float sum = 0.0;
    for (int k = 0; k < N; k++) {
        sum += A[r * N + k] * B[k * N + c]; 
    }
    C[idx] = sum;
}

void matmul(const float* A, const float* B, float* C, size_t n, unsigned int threads_per_block) {
    int num_of_blocks = (n * n - 1) / threads_per_block + 1;
    matmul_kernel <<< num_of_blocks, threads_per_block >>> (A, B, C, n);
}