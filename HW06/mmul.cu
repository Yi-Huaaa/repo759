#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "mmul.h"

void mmul(cublasHandle_t handle, const float* A, const float* B, float* C, int n) {
    
    /* format: cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, alpha, A, lda, B, ldb, beta, C, ldc);
    CUBLAS_OP_N: non-transport, 
    lda: leading dimension of A, B, C
    C = (alpha)*A*B + (beta)*C
    */
    // cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, 1, A, n, B, n, 0, C, n);
    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, A, n, B, n, &beta, C, n);
}