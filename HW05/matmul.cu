#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "matmul.cuh"


template<typename T>
__global__ void matmul_kernel (const T* A, const T* B, T* C, unsigned int n) {
    int BLOCK_SIZE = blockDim.x;
    int int_n = int(n);

    // Block index
    int bx = blockIdx.x; //the B (and C) matrix sub-block column index
    int by = blockIdx.y; //the A (and C) matrix sub-block row index
    // Thread index
    int tx = threadIdx.x; //the column index in the sub-block
    int ty = threadIdx.y; //the row index in the sub-block
    
    int aBegin = (int_n * BLOCK_SIZE * by); // Index of the first sub-matrix of A processed by the block
    int aEnd = (aBegin + int_n - 1); // Index of the last sub-matrix of A processed by the block
    int aStep = (BLOCK_SIZE); // Step size used to iterate through the sub-matrices of A
    int bBegin = (BLOCK_SIZE * bx); // Index of the first sub-matrix of B processed by the block
    int bStep = (BLOCK_SIZE * int_n); // Step size used to iterate through the sub-matrices of B
    T Csub = 0; // The element of the block sub-matrix that is computed by the thread

    extern __shared__ __align__(sizeof(T)) unsigned char smem[];
    T *shared_mem = reinterpret_cast<T *>(smem);
    T *As, *Bs;
    As = shared_mem;
    Bs = &shared_mem[BLOCK_SIZE*BLOCK_SIZE];

    // Loop over all the sub-matrices (tiles) of A and B required to
    // compute the block sub-matrix; moving in A left to right in
    // a row, and in B from top to bottom in a column
    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
        // Load tiles from global memory into shared memory; each
        // thread loads one element of the two tiles from A & B
        unsigned int a_idx = (a+int_n*ty+tx), b_idx = (b+int_n*ty+tx);
        As[ty*BLOCK_SIZE+tx] = (a_idx < n*n) ? (A[a_idx]) : (0);
        Bs[ty*BLOCK_SIZE+tx] = (b_idx < n*n) ? (B[b_idx]) : (0);
        // Synchronize to make sure the matrices are loaded
        __syncthreads();
        // Each thread in this block computes one element
        // of the block sub-matrix (tile). Thread with indexes
        // ty and tx computes in this tile the entry [ty][tx].
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Csub += As[ty*BLOCK_SIZE+k] * Bs[k*BLOCK_SIZE+tx];
        }
        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }
    // Write the block sub-matrix to global memory;
    // each thread writes one element
    int c = int_n * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    // Note: only write back the data located in valid memory
    int c_idx = c+int_n*ty+tx;
    if (c_idx < n*n) {
        C[c_idx] = Csub;
    }
}

__host__ void matmul_1(const int *A, const int *B, int *C, unsigned int n, unsigned int block_dim) {
    int num_of_blocks = (n > block_dim) ? ((n+block_dim-1)/block_dim) : (1); // ceiling

    dim3 dimBlock(block_dim, block_dim);
    dim3 dimGrid(num_of_blocks, num_of_blocks);
    unsigned int smem_sz = 2*block_dim*block_dim*sizeof(int);
    matmul_kernel<int> <<< dimGrid, dimBlock, smem_sz >>>(A, B, C, n);
}

__host__ void matmul_2(const float *A, const float *B, float *C, unsigned int n, unsigned int block_dim) {
    int num_of_blocks = (n > block_dim) ? ((n+block_dim-1)/block_dim) : (1);

    dim3 dimBlock(block_dim, block_dim);
    dim3 dimGrid(num_of_blocks, num_of_blocks);
    unsigned int smem_sz = 2*block_dim*block_dim*sizeof(float);
    matmul_kernel<float> <<< dimGrid, dimBlock, smem_sz >>>(A, B, C, n);      
}

__host__ void matmul_3(const double *A, const double *B, double *C, unsigned int n, unsigned int block_dim) {
    int num_of_blocks = (n > block_dim) ? ((n+block_dim-1)/block_dim) : (1);

    dim3 dimBlock(block_dim, block_dim);
    dim3 dimGrid(num_of_blocks, num_of_blocks);
    unsigned int smem_sz = 2*block_dim*block_dim*sizeof(double);
    matmul_kernel<double> <<< dimGrid, dimBlock, smem_sz >>>(A, B, C, n);    
}


