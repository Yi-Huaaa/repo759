#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "matmul.cuh"

// #define check

void init (float *a, float *b, const int n) {
	for (int i = 0; i < n*n; i++){
		a[i] = (((float)rand()/RAND_MAX)*2)-1; // range [-1, 1];
		b[i] = (((float)rand()/RAND_MAX)*2)-1; // range [-1, 1];
	}
}

void init_check (float *a, float *b, const int n) {
	for (int i = 0; i < n*n; i++){
		a[i] = 1;
		b[i] = 1;
	}
    for (auto i = 0; i < int(n*n); i++) 
        assert(a[i] == b[i]);
}

void print_arr (const float *arr, int len) {
    printf("arr = [");
    for (auto i = 0; i < len; i++) {
        printf("%.2lf, ", arr[i]);
    } printf("]\n");
}


int main(int argc, char* argv[]){
	int n = atoi(argv[1]);
    unsigned int threads_per_block = atoi(argv[2]);

    float *A = (float*)malloc(n*n*sizeof(float)); // Create matrices (as 1D row major arrays) A and B of size n × n on the host.
    float *B = (float*)malloc(n*n*sizeof(float)); 
    float *C = (float*)malloc(n*n*sizeof(float)); memset(C, 0, n*n*sizeof(float));
    float *A_gpu; cudaMalloc((void**)&A_gpu, n*n*sizeof(float));
    float *B_gpu; cudaMalloc((void**)&B_gpu, n*n*sizeof(float));
    float *C_gpu; cudaMalloc((void**)&C_gpu, n*n*sizeof(float));
    
    srand(time(NULL));
#ifdef check 
    init_check(A, B, n);
#else     
    init (A, B, n);
#endif 

    cudaMemcpy(A_gpu, A, n*n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_gpu, B, n*n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(C_gpu, C, n*n*sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    cudaEvent_t start;
    cudaEvent_t stop;
    float ms; 
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
        
        matmul(A_gpu, B_gpu, C_gpu, n, threads_per_block);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    cudaMemcpy(C, C_gpu, n*n*sizeof(float), cudaMemcpyDeviceToHost);
    
// #ifdef check    
//     cudaDeviceSynchronize();
//     print_arr(C, n*n);
// #endif

    printf("%.2lf\n", C[n*n-1]); // Print the last element of the resulting matrix.
    printf("%.2lf\n", ms); // Print the time taken to execute your matmul function in milliseconds using CUDA events.

    
    free (A);
    free (B);
    free (C);
    cudaFree(A_gpu);
    cudaFree(B_gpu);
    cudaFree(C_gpu);

	return 0;
}
