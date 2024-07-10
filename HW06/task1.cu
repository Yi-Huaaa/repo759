#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "mmul.h"
#define IDX2C(i,j,ld) (((j)*(ld))+(i)) // for column major
// #define check 

void init (float *a, float *b, float *c, const int n) {
	auto idx = 0;
    for (auto i = 0; i < n; i++){
		for (auto j = 0; j < n; j++){
            idx = IDX2C(i,j,n);
            a[idx] = (((float)rand()/RAND_MAX)*2)-1; // range [-1, 1];
            b[idx] = (((float)rand()/RAND_MAX)*2)-1; // range [-1, 1];
            c[idx] = 0;
        }
	}
}

void init_check (float *a, float *b, float *c, const int n) {
	auto idx = 0;
    for (auto i = 0; i < n; i++){
		for (auto j = 0; j < n; j++){
            idx = IDX2C(i,j,n);
            a[idx] = 1;
            b[idx] = j;
            c[idx] = 0;
        }
	}
}

void print_arr (const float *arr, int n) {
    for (auto i = 0; i < n; i++) {
        for (auto j = 0; j < n; j++) {
            int indx = IDX2C(i,j,n);
            printf("%.2lf, ", arr[indx]);
        } printf("\n");
    } printf("]\n");
}


int main(int argc, char* argv[]){
	int n = atoi(argv[1]);
    int n_tests = atoi(argv[2]);

    // Creates three n×n matrices, A, B, and C, stored in column-major order in managed memory 
    // with random float numbers in the range [-1, 1], where n is the first command line argument as below.
    float *A, *B, *C;
    cudaMallocManaged(&A, n*n*sizeof(float));
    cudaMallocManaged(&B, n*n*sizeof(float));
    cudaMallocManaged(&C, n*n*sizeof(float));

    srand(time(NULL));
#ifdef check 
    init_check(A, B, C, n);
#else     
    init (A, B, C, n);
#endif 

    cublasHandle_t cublasHandle;
    cudaEvent_t start;
    cudaEvent_t stop;
    float ms; 
    cublasCreate(&cublasHandle);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
        
    for (auto ii = 0; ii < n_tests; ii++)
        mmul(cublasHandle, A, B, C, n);    
    
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&ms, start, stop);

    // Prints the average time taken by a single call to mmul in milliseconds using CUDA events.
    printf("%.2lf\n", (ms/n_tests));

    cublasDestroy(cublasHandle);
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

	return 0;
}
