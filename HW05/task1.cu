#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>

#include "matmul.cuh"

// #define check
// #define check_answer

double one_min_one_zero (double input) {
    if (input < 0.333) {
        return 1.0;
    } else if (input < 0.666) {
        return 0.0;
    } else {
        return -1.0;
    }
}

void init_rand (int *a_i, int *b_i, float *a_f, float *b_f, double *a_d, double *b_d, unsigned int n) {
	double t1 = 0., t2 = 0., f1 = 0., f2 = 0.;
    for (unsigned int i = 0; i < n*n; i++){
		f1 = (((double)rand()/RAND_MAX)*2)-1; // range [-1, 1];
        f2 = (((double)rand()/RAND_MAX)*2)-1; // range [-1, 1];
        t1 = one_min_one_zero(f1);
        t2 = one_min_one_zero(f2);
        a_i[i] = int(t1);
		b_i[i] = int(t2);
        a_f[i] = float(t1);
		b_f[i] = float(t2); 
        a_d[i] = double(t1);
		b_d[i] = double(t2);  
	}
}


// Creates and fills however you like row-major representations of n×n matrices A, B, and C, 
// where n is the first command line argument as below.
void init_with_one (int *a_i, int *b_i, float *a_f, float *b_f, double *a_d, double *b_d, unsigned int n) {
	for (unsigned int i = 0; i < n*n; i++){
        a_i[i] = 1;
		b_i[i] = 1;
        a_f[i] = 1;
		b_f[i] = 1; 
        a_d[i] = 1;
		b_d[i] = 1;  
	}
}
 
template<typename T>
void print_arr (T *arr, unsigned int len) {
    printf("arr = [");
    for (unsigned int i = 0; i < len; i++) {
        printf("%.2lf, ", arr[i]);
    } printf("]\n");
}

template<typename T>
void ask_mem (T *ptrs[6], unsigned int n) {
    T *A = (T*)malloc(n*n*sizeof(T)); // Create matrices (as 1D row major arrays) A and B of size n × n on the host.
    T *B = (T*)malloc(n*n*sizeof(T)); 
    T *C = (T*)malloc(n*n*sizeof(T)); memset(C, 0, n*n*sizeof(T));
    T *A_gpu; cudaMalloc((void**)&A_gpu, n*n*sizeof(T));
    T *B_gpu; cudaMalloc((void**)&B_gpu, n*n*sizeof(T));
    T *C_gpu; cudaMalloc((void**)&C_gpu, n*n*sizeof(T));
    ptrs[0] = A; 
    ptrs[1] = B; 
    ptrs[2] = C;
    ptrs[3] = A_gpu; 
    ptrs[4] = B_gpu; 
    ptrs[5] = C_gpu;
}

template<typename T>
void free_mem (T *ptrs[6]) {
    free(ptrs[0]); 
    free(ptrs[1]); 
    free(ptrs[2]);
    cudaFree(ptrs[3]); 
    cudaFree(ptrs[4]); 
    cudaFree(ptrs[5]); 
}

template<typename T>
void mem_copy (T *ptrs[6], unsigned int n) {
    cudaMemcpy(ptrs[3], ptrs[0], n*n*sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(ptrs[4], ptrs[1], n*n*sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(ptrs[5], ptrs[2], n*n*sizeof(T), cudaMemcpyHostToDevice);
}


int main(int argc, char* argv[]){
    unsigned int n = atoi(argv[1]);
    unsigned int block_dim = atoi(argv[2]); // block_dim: threads_per_block

    int *ptrs_int [6];       ask_mem<int>(ptrs_int, n);
    float *ptrs_float [6];   ask_mem<float>(ptrs_float, n);
    double *ptrs_double [6]; ask_mem<double>(ptrs_double, n);
    
    
    srand(time(NULL));
#ifdef check    
    init_with_one(ptrs_int[0], ptrs_int[1], ptrs_float[0], ptrs_float[1], ptrs_double[0], ptrs_double[1], n);
#else
    init_rand(ptrs_int[0], ptrs_int[1], ptrs_float[0], ptrs_float[1], ptrs_double[0], ptrs_double[1], n);
#endif 

    mem_copy<int>(ptrs_int, n);
    mem_copy<float>(ptrs_float, n);
    mem_copy<double>(ptrs_double, n);
    cudaDeviceSynchronize();

    cudaEvent_t start;
    cudaEvent_t stop;
    float ms_int = 0., ms_float = 0., ms_double = 0.; 
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
// Compute datatype: int
    cudaEventRecord(start);
        
        matmul_1(ptrs_int[3], ptrs_int[4], ptrs_int[5], n, block_dim);

    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&ms_int, start, stop);
    cudaMemcpy(ptrs_int[2], ptrs_int[5], n*n*sizeof(int), cudaMemcpyDeviceToHost);
    
// Compute datatype: float
    cudaEventRecord(start);
        
        matmul_2(ptrs_float[3], ptrs_float[4], ptrs_float[5], n, block_dim);

    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&ms_float, start, stop);
    cudaMemcpy(ptrs_float[2], ptrs_float[5], n*n*sizeof(float), cudaMemcpyDeviceToHost);

// Compute datatype: double
    cudaEventRecord(start);
            
        matmul_3(ptrs_double[3], ptrs_double[4], ptrs_double[5], n, block_dim);

    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&ms_double, start, stop);
    cudaMemcpy(ptrs_double[2], ptrs_double[5], n*n*sizeof(double), cudaMemcpyDeviceToHost);


    // #ifdef check_answer 
    //     cudaDeviceSynchronize();
    //     printf("Datatype, int:\n");
    //     print_arr<int>(ptrs_int[2], n*n);
    // #endif

    // print answer
    int *C_int = ptrs_int[2];
    printf("%d\n", C_int[0]); //Prints the first element of the resulting C.
    printf("%d\n", C_int[n*n-1]); // Prints the last element of the resulting C.
    printf("%.2lf\n", ms_int); //  Prints the time taken to run the matrix multiplication in milliseconds using CUDA events.
    
    float *C_float = ptrs_float[2];
    printf("%.2lf\n", C_float[0]); //Prints the first element of the resulting C.
    printf("%.2lf\n", C_float[n*n-1]); // Prints the last element of the resulting C.
    printf("%.2lf\n", ms_float); //  Prints the time taken to run the matrix multiplication in milliseconds using CUDA events.
    
    double *C_double = ptrs_double[2];
    printf("%.2lf\n", C_double[0]); //Prints the first element of the resulting C.
    printf("%.2lf\n", C_double[n*n-1]); // Prints the last element of the resulting C.
    printf("%.2lf\n", ms_double); //  Prints the time taken to run the matrix multiplication in milliseconds using CUDA events.
     

    free_mem<int>(ptrs_int);
    free_mem<float>(ptrs_float);
    free_mem<double>(ptrs_double);

	return 0;
}
